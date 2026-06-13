#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/gaia_eval.py — GAIA 外部基准评测 (v5.1.0)

加载 HuggingFace `gaia-benchmark/GAIA` 数据集（config="2023_all", split="validation"），
对每个问题：
  1. G-Core：使用 DeepSeek API 回答
  2. D-Core：与标准答案做精确匹配 + 模糊匹配
  3. Φ-Gate：若启用 δ-mem 管道，将 Q&A 更新入 δ-mem S 矩阵并触发 flush
  4. 记录每次推演的 CV / drift / S 矩阵 norm 等指标

GAIA 数据格式：task_id, Question, Level, Final answer, file_name, file_path, Annotator Metadata

输出：results/gaia_v510.json

用法：
    python scripts/gaia_eval.py --limit 50         # 跑前 50 题
    python scripts/gaia_eval.py --no-delta         # 禁用 δ-mem
    python scripts/gaia_eval.py                   # 默认前 50 题 + δ-mem
    python scripts/gaia_eval.py --limit 0          # 跑全部 165 题

v5.1.0 (2026-06-13): 集成到 v5.1 外部基准评测 (Taiji-OS v5.1.0)

Author: Taiji OS Team (寇豆码)
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# ── 路径设置：让脚本独立运行 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 导入核心模块（δ-mem 管道） ─────────────────────────────────────────
from core.world_model import WorldModel
from core.self_consistency_loop import SelfConsistencyLoop
from core.delta_fusion import DeltaFusion
from core.embedding_adapter import auto_detect_dim
from core.drift_detector import DriftDetector, HyperParamAdapter

# ── DeepSeek API ────────────────────────────────────────────────────────
from hal.llm_router import LLMRouter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s"
)
logger = logging.getLogger("gaia_eval")


# ────────────────────────────────────────────────────────────────────────────
# GAIA 数据加载
# ────────────────────────────────────────────────────────────────────────────


def load_gaia(limit: Optional[int] = None) -> list[dict]:
    """从 HuggingFace 加载 GAIA 数据集。

    使用 `datasets` 库加载 `gaia-benchmark/GAIA` 的 2023_all 配置 validation split。
    HuggingFace GAIA 实际字段名为：task_id, question, level, answer, file_name,
    file_path, annotator_meta 等。本脚本兼容多种大小写变体。

    Args:
        limit: 限制加载的问题数；None 表示加载全部（默认 165 题 validation）。

    Returns:
        标准化后的实例字典列表，每个包含：
          - task_id, question, level, answer, file_name, file_path
    """
    from datasets import load_dataset

    logger.info("Loading GAIA 2023_all validation from HuggingFace (streaming) ...")
    try:
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load GAIA: {e}")
        raise

    instances = []
    # Streaming mode: iterate to collect items
    collected = []
    for item in ds:
        collected.append(item)
        if limit and len(collected) >= limit:
            break
    n = len(collected)
    logger.info(f"  dataset size: {n}, using first {n} instances")

    # GAIA HF schema (v1+): task_id, question, level, answer, file_name, file_path
    # Annotator Metadata is the legacy name; current HF dataset may use annotator_meta
    for i in range(n):
        row = collected[i]
        # 兼容字段大小写
        task_id = (
            row.get("task_id")
            or row.get("Task_id")
            or row.get("id")
            or f"unknown_{i}"
        )
        question = (
            row.get("question")
            or row.get("Question")
            or ""
        )
        level = (
            row.get("level")
            or row.get("Level")
            or ""
        )
        answer = (
            row.get("answer")
            or row.get("Final answer")
            or row.get("final_answer")
            or row.get("Answer")
            or ""
        )
        file_name = row.get("file_name") or ""
        file_path = row.get("file_path") or ""
        annotator_meta = (
            row.get("annotator_meta")
            or row.get("Annotator Metadata")
            or row.get("annotator")
            or {}
        )

        instances.append({
            "task_id": task_id,
            "question": question,
            "level": str(level),
            "answer": str(answer).strip() if answer else "",
            "file_name": file_name,
            "file_path": file_path,
            "annotator_meta": annotator_meta if isinstance(annotator_meta, dict) else {},
        })

    return instances


# ────────────────────────────────────────────────────────────────────────────
# 答案匹配 (D-Core)
# ────────────────────────────────────────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    """归一化答案字符串以便匹配。

    策略：
      - 转为小写
      - 去除首尾空白
      - 去除标点
      - 合并空白
    """
    if not answer:
        return ""
    s = str(answer).lower().strip()
    # 去除常见标点
    s = re.sub(r"[\.,!?;:'\"`()\[\]{}]", " ", s)
    # 合并空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(predicted: str, gold: str) -> bool:
    """精确匹配（归一化后）。"""
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    if not p and not g:
        return True
    if not p or not g:
        return False
    return p == g


def contains_match(predicted: str, gold: str) -> bool:
    """包含匹配：gold 是 predicted 的子串（归一化后）。"""
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    if not g:
        return False
    return g in p


def fuzzy_match(predicted: str, gold: str, threshold: float = 0.75) -> bool:
    """模糊匹配：SequenceMatcher 相似度 ≥ 阈值。"""
    if not predicted or not gold:
        return exact_match(predicted, gold)
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    sm = difflib.SequenceMatcher(None, p, g)
    return sm.ratio() >= threshold


def dcore_judge_answer(predicted: str, gold: str) -> dict:
    """D-Core 答案判定：精确 + 包含 + 模糊三层匹配。

    Args:
        predicted: LLM 生成的答案字符串。
        gold: GAIA 标准答案字符串。

    Returns:
        dict 包含 exact, contains, fuzzy, fuzzy_score, passed, level。
    """
    if not predicted or not gold:
        return {
            "exact": False,
            "contains": False,
            "fuzzy": False,
            "fuzzy_score": 0.0,
            "passed": False,
        }

    em = exact_match(predicted, gold)
    cm = contains_match(predicted, gold)
    p_n = normalize_answer(predicted)
    g_n = normalize_answer(gold)
    score = difflib.SequenceMatcher(None, p_n, g_n).ratio() if (p_n and g_n) else 0.0
    fm = score >= 0.75
    # 任何一层匹配即视为通过
    passed = em or cm or fm
    return {
        "exact": bool(em),
        "contains": bool(cm),
        "fuzzy": bool(fm),
        "fuzzy_score": round(float(score), 4),
        "passed": bool(passed),
    }


# ────────────────────────────────────────────────────────────────────────────
# 答案生成 (G-Core)
# ────────────────────────────────────────────────────────────────────────────


ANSWER_PROMPT = """You are a precise question-answering assistant. Answer the following question with a CONCISE, DIRECT answer.

Question: {question}

Additional context (if any):
{context}

IMPORTANT INSTRUCTIONS:
- Give a SHORT, DIRECT answer (1-5 words or a single number/phrase)
- Do not provide explanations unless specifically asked
- If the question requires a number, give just the number
- If the question requires a name, give just the name
- If unsure, give your best guess

Answer:"""


def generate_answer(llm: LLMRouter, instance: dict) -> str:
    """调用 DeepSeek API 生成候选答案 (G-Core)。

    Args:
        llm: LLMRouter 实例。
        instance: GAIA 实例字典。

    Returns:
        LLM 生成的答案字符串。
    """
    question = instance.get("question", "")
    # 简化的 context：仅包含 file_name
    context_parts = []
    if instance.get("file_name"):
        context_parts.append(f"File referenced: {instance['file_name']}")
    context = "\n".join(context_parts) if context_parts else "(none)"

    prompt = ANSWER_PROMPT.format(question=question[:2000], context=context)
    try:
        response = llm.complete(prompt)
    except Exception as e:
        logger.warning(f"LLM call failed for {instance['task_id']}: {e}")
        response = f"[Error] {e}"
    return response


# ────────────────────────────────────────────────────────────────────────────
# δ-mem 管道初始化与执行
# ────────────────────────────────────────────────────────────────────────────


def init_delta_pipeline(use_delta: bool = True) -> dict:
    """初始化 δ-mem 管道：WorldModel + SelfConsistencyLoop + DeltaFusion。

    Args:
        use_delta: 是否启用 δ-mem L1/L2 融合。

    Returns:
        dict 包含 llm, wm, loop, fusion 四个组件。
    """
    llm = LLMRouter()
    wm = WorldModel(dim=1536, config_path=str(PROJECT_ROOT / "config.yaml"))
    auto_detect_dim(wm)

    fusion = DeltaFusion() if use_delta else None
    loop = SelfConsistencyLoop(
        llm, wm,
        dcore_mode="semantic",
        delta_fusion=fusion,
    )
    loop.phi.base_threshold = 0.05
    loop.phi._current_threshold = 0.05

    # v5.1: 启用 HyperParamAdapter
    loop.drift_detector.adapter = HyperParamAdapter()

    return {"llm": llm, "wm": wm, "loop": loop, "fusion": fusion}


def run_step_with_delta(components: dict, instance: dict, predicted_answer: str) -> dict:
    """通过 SelfConsistencyLoop 跑一步推演，触发 δ-mem S 更新。

    Returns:
        dict 包含 phi, cv, drift, output, accepted, S_norm。
    """
    loop = components["loop"]
    fusion = components["fusion"]
    dd = loop.drift_detector

    # 构造 env
    env = {
        "intent": f"gaia_level={instance.get('level', '')}",
        "task_id": instance.get("task_id", ""),
    }

    # 用问题触发推演（这样 S 矩阵学习到问题的语义模式）
    output, reason = loop.step(env, instance.get("question", ""))
    phi = float(safe_wm_phi(components["wm"]))
    cv = float(dd.current_cv) if dd.count >= 3 else 0.0
    is_drifting = bool(dd.is_drifting())

    s_norm = 0.0
    if fusion is not None:
        try:
            s_norm = float(np.linalg.norm(fusion.delta_layer.smatrix.S, "fro"))
        except Exception:
            s_norm = 0.0

    return {
        "phi": round(phi, 4),
        "cv": round(cv, 4),
        "is_drifting": is_drifting,
        "output_len": len(output) if output else 0,
        "accepted": output is not None,
        "reason": reason if output is None else None,
        "S_fro_norm": round(s_norm, 4),
    }


def safe_wm_phi(wm) -> float:
    """安全获取 WorldModel 的当前 Φ 值。"""
    try:
        if hasattr(wm, "current_phi"):
            return float(wm.current_phi)
        if hasattr(wm, "last_phi"):
            return float(wm.last_phi)
        if hasattr(wm, "psi") and wm.psi is not None:
            psi = wm.psi
            norm = float(np.linalg.norm(psi))
            if norm < 1e-8:
                return 0.0
            return min(1.0, norm / (np.sqrt(psi.size) + 1e-8))
    except Exception:
        pass
    return 0.0


# ────────────────────────────────────────────────────────────────────────────
# 主评测循环
# ────────────────────────────────────────────────────────────────────────────


def run_evaluation(limit: int = 50, use_delta: bool = True) -> dict:
    """运行 GAIA 评测。

    Args:
        limit: 评测问题数（默认 50）。
        use_delta: 是否集成 δ-mem 管道（默认 True）。

    Returns:
        完整评测结果字典。
    """
    print("=" * 72)
    print(f"GAIA Evaluation — Taiji-OS v5.1.0")
    print(f"  limit:    {limit}")
    print(f"  δ-mem:    {'ENABLED' if use_delta else 'DISABLED'}")
    print("=" * 72)

    # ── 1) 加载数据 ────────────────────────────────────────────────────
    instances = load_gaia(limit=limit)
    print(f"Loaded {len(instances)} questions")
    if instances:
        print(f"  Sample task_id:   {instances[0]['task_id']}")
        print(f"  Sample level:     {instances[0]['level']}")
        print(f"  Sample question:  {instances[0]['question'][:80]}...")

    # ── 2) 初始化 δ-mem 管道 ──────────────────────────────────────────
    print("\n[1] Initializing δ-mem pipeline ...")
    components = init_delta_pipeline(use_delta=use_delta)
    fusion = components["fusion"]
    if fusion is not None:
        print(f"    S matrix rank: {fusion.delta_layer.smatrix.r}")
        print(f"    L1/L2 fusion: ENABLED")
    else:
        print(f"    L1/L2 fusion: DISABLED (--no-delta)")
    print(f"    HyperParamAdapter: {'ENABLED' if components['loop'].drift_detector.adapter else 'DISABLED'}")

    # ── 3) 主评测循环 ──────────────────────────────────────────────────
    print(f"\n[2] Running evaluation on {len(instances)} questions ...")
    print("-" * 72)

    results = []
    n_passed = 0
    phi_vals = []
    cv_vals = []
    drift_count = 0
    level_stats = {}  # level -> {total, passed}

    start_time = time.time()

    for i, instance in enumerate(instances, 1):
        tid = instance["task_id"]
        level = instance["level"]
        try:
            # G-Core: 生成答案
            predicted = generate_answer(components["llm"], instance)

            # D-Core: 答案匹配
            judge = dcore_judge_answer(predicted, instance["answer"])

            # δ-mem: 推演一步
            delta_info = {}
            if use_delta and fusion is not None:
                delta_info = run_step_with_delta(components, instance, predicted)
                phi_vals.append(delta_info["phi"])
                cv_vals.append(delta_info["cv"])
                if delta_info["is_drifting"]:
                    drift_count += 1

            # 累计指标
            if judge["passed"]:
                n_passed += 1
            if level not in level_stats:
                level_stats[level] = {"total": 0, "passed": 0}
            level_stats[level]["total"] += 1
            if judge["passed"]:
                level_stats[level]["passed"] += 1

            # 记录
            result = {
                "task_id": tid,
                "level": level,
                "question": instance["question"][:200],
                "gold_answer": instance["answer"],
                "predicted_answer": predicted[:200],
                "judge": judge,
                "passed": judge["passed"],
                "delta_info": delta_info,
            }
            results.append(result)

            # 进度输出
            status = "✓" if judge["passed"] else "✗"
            elapsed = time.time() - start_time
            avg = elapsed / i
            eta = avg * (len(instances) - i)
            print(
                f"  [{i:3d}/{len(instances)}] {status} L{level} "
                f"sim={judge['fuzzy_score']:.2f} "
                f"cv={delta_info.get('cv', 0):.3f} "
                f"Φ={delta_info.get('phi', 0):.3f} "
                f"[{elapsed:.0f}s, eta {eta:.0f}s] {tid[:32]}"
            )

        except Exception as e:
            logger.error(f"Error on {tid}: {e}")
            logger.debug(traceback.format_exc())
            results.append({
                "task_id": tid,
                "level": level,
                "error": str(e),
                "passed": False,
                "judge": {"passed": False, "fuzzy_score": 0.0},
            })

    elapsed_total = time.time() - start_time

    # ── 4) 汇总指标 ──────────────────────────────────────────────────
    n_total = len(instances)
    accuracy = n_passed / n_total if n_total > 0 else 0.0

    # Level breakdown
    level_accuracy = {}
    for lvl, st in sorted(level_stats.items()):
        lvl_acc = st["passed"] / st["total"] if st["total"] > 0 else 0.0
        level_accuracy[str(lvl)] = {
            "total": st["total"],
            "passed": st["passed"],
            "accuracy": round(lvl_acc, 4),
        }

    # δ-mem 汇总
    phi_mean = float(np.mean(phi_vals)) if phi_vals else 0.0
    cv_mean = float(np.mean(cv_vals)) if cv_vals else 0.0
    cv_max = float(np.max(cv_vals)) if cv_vals else 0.0

    print("\n" + "=" * 72)
    print("RESULTS:")
    print(f"  Total questions : {n_total}")
    print(f"  Passed          : {n_passed}")
    print(f"  Accuracy        : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Elapsed         : {elapsed_total:.1f}s ({elapsed_total/n_total:.1f}s/question)")
    print()
    print("  By Level:")
    for lvl, st in sorted(level_accuracy.items()):
        print(f"    L{lvl}: {st['passed']:3d}/{st['total']:3d} = {st['accuracy']:.3f}")
    if use_delta:
        print()
        print(f"  δ-mem:")
        print(f"    Φ mean         : {phi_mean:.4f}")
        print(f"    CV mean / max  : {cv_mean:.4f} / {cv_max:.4f}")
        print(f"    Drift rounds   : {drift_count}/{n_total}")
        adapter = components["loop"].drift_detector.adapter
        if adapter is not None:
            ast = adapter.stats()
            print(f"    Adapter pushes : {ast.get('history_len', 0)}")
            if ast.get("last_adapted"):
                la = ast["last_adapted"]
                print(f"    Last adapted   : cv_mid={la['cv_mid']:.3f}, "
                      f"γ_max={la['gamma_max']:.3f}, γ_min={la['gamma_min']:.3f}")

    # ── 5) 写入结果 ──────────────────────────────────────────────────
    report = {
        "experiment": "GAIA_v5.1.0",
        "model": "deepseek-chat (via LLMRouter)",
        "n_questions": n_total,
        "n_passed": n_passed,
        "accuracy": round(accuracy, 4),
        "elapsed_sec": round(elapsed_total, 2),
        "level_accuracy": level_accuracy,
        "delta_mem_enabled": use_delta,
        "delta_mem_stats": {
            "phi_mean": round(phi_mean, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_max": round(cv_max, 4),
            "drift_rounds": drift_count,
        },
        "results": results,
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gaia_v510.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="GAIA 外部基准评测 (Taiji-OS v5.1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--limit", type=int, default=50,
        help="评测问题数（默认 50；设为 0 表示跑全部 165 题）",
    )
    p.add_argument(
        "--no-delta", action="store_true",
        help="禁用 δ-mem L1/L2 融合（仅评测 LLM 自身能力）",
    )
    return p.parse_args()


def main():
    """主入口。"""
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    use_delta = not args.no_delta
    run_evaluation(limit=limit, use_delta=use_delta)


if __name__ == "__main__":
    main()

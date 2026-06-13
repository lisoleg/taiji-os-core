"""
swebench_eval.py — SWE-bench Lite 基准评测脚本 (v5.1)

使用 DeepSeek API 对 SWE-bench Lite（300 题，11 个 Python 仓库）进行评测。
数据来自 HuggingFace `princeton-nlp/SWE-bench_Lite`。
评测方式：API 生成 patch → 与标准答案做模糊匹配（prompt-based）。

用法:
    python scripts/swebench_eval.py --limit 50          # 评测前 50 题
    python scripts/swebench_eval.py --limit 0           # 全部 300 题
    python scripts/swebench_eval.py --repo django/django # 仅某仓库

Author: Taiji OS Team
Version: v5.1.0 (2026-06-13)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("swebench_eval")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── DeepSeek API 调用 ───────────────────────────────────────────────────

def call_deepseek(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """调用 DeepSeek Chat API 生成回答。"""
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个高级软件工程师，擅长代码修复。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        logger.error(f"  API HTTP {e.code}: {e.reason}")
        return f"ERROR: HTTP {e.code}"
    except Exception as e:
        logger.error(f"  API 错误: {e}")
        return f"ERROR: {e}"


# ── SWE-bench Prompt ────────────────────────────────────────────────────

SWEBENCH_PROMPT = """你是一个高级软件工程师，需要修复以下 GitHub issue。

仓库：{repo}
Issue 描述：
{problem_statement}

请生成一个 diff patch 来修复此问题。输出格式：
```diff
<your patch here>
```

只修复 issue 描述的问题，不要做无关改动。

提示：{hints}
"""


# ── Patch 解析与比较 ────────────────────────────────────────────────────

def extract_patch(response: str) -> str:
    """从 LLM 响应中提取 diff patch。"""
    diff_match = re.search(r'```diff\s*\n(.*?)```', response, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()
    patch_match = re.search(r'```patch\s*\n(.*?)```', response, re.DOTALL)
    if patch_match:
        return patch_match.group(1).strip()
    return response.strip()


def compute_patch_similarity(predicted_patch: str, gold_patch: str) -> float:
    """计算两个 patch 的相似度。

    综合评分: 35% SequenceMatcher + 35% 行级匹配 + 30% 关键符号匹配
    """
    if not predicted_patch or not gold_patch:
        return 0.0

    pred_lines = [l.strip() for l in predicted_patch.split('\n') if l.strip()]
    gold_lines = [l.strip() for l in gold_patch.split('\n') if l.strip()]

    if not pred_lines or not gold_lines:
        return 0.0

    seq_sim = SequenceMatcher(None, predicted_patch, gold_patch).ratio()
    line_sim = SequenceMatcher(None, '\n'.join(pred_lines), '\n'.join(gold_lines)).ratio()

    def extract_keys(lines):
        keys = set()
        for l in lines:
            keys.update(re.findall(r'[a-zA-Z0-9_/.-]+\.py', l))
            keys.update(re.findall(r'(?:def|class)\s+(\w+)', l))
        return keys

    pred_keys = extract_keys(pred_lines)
    gold_keys = extract_keys(gold_lines)
    key_sim = len(pred_keys & gold_keys) / max(len(gold_keys), 1) if gold_keys else 1.0

    similarity = 0.35 * seq_sim + 0.35 * line_sim + 0.30 * key_sim
    return min(1.0, similarity)


# ── 主评测 ──────────────────────────────────────────────────────────────

def load_swebench_data(limit: int = 0, repo_filter: str = None) -> list[dict]:
    """从 HuggingFace 加载 SWE-bench Lite 数据集。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        questions = []
        for item in ds:
            if repo_filter and repo_filter not in str(item["repo"]):
                continue
            ps = str(item["problem_statement"])
            title = ps.split('\n')[0].strip()[:100] if ps else ""
            questions.append({
                "instance_id": str(item["instance_id"]),
                "repo": str(item["repo"]),
                "base_commit": str(item.get("base_commit", "")),
                "problem_statement": ps,
                "hints": str(item.get("hints_text", "")),
                "patch": str(item.get("patch", "")),
                "issue_title": title,
            })
        if limit > 0:
            questions = questions[:limit]
        logger.info(f"加载 SWE-bench Lite: {len(questions)} 题 (filter={repo_filter or '全部'})")
        return questions
    except Exception as e:
        logger.error(f"加载 SWE-bench 失败: {e}")
        raise


def run_swebench_evaluation(questions: list[dict]) -> dict:
    """运行 SWE-bench Lite 评测。

    Returns:
        评测报告字典。
    """
    results = []
    repo_stats = defaultdict(lambda: {"total": 0, "resolved": 0, "similarities": []})
    total_time = 0.0

    for i, q in enumerate(questions):
        instance_id = q["instance_id"]
        repo = q["repo"]

        logger.info(f"[{i+1}/{len(questions)}] {instance_id} ({repo})")

        t0 = time.time()

        prompt = SWEBENCH_PROMPT.format(
            repo=repo,
            problem_statement=q["problem_statement"][:2000],
            hints=q["hints"] or "无额外提示",
        )

        response = call_deepseek(prompt)
        predicted_patch = extract_patch(response)

        elapsed = time.time() - t0
        total_time += elapsed

        gold_patch = q["patch"]
        similarity = compute_patch_similarity(predicted_patch, gold_patch)
        resolved = similarity >= 0.60

        repo_stats[repo]["total"] += 1
        if resolved:
            repo_stats[repo]["resolved"] += 1
        repo_stats[repo]["similarities"].append(similarity)

        results.append({
            "instance_id": instance_id,
            "repo": repo,
            "similarity": round(similarity, 4),
            "resolved": resolved,
            "patch_length": len(predicted_patch),
            "time_s": round(elapsed, 2),
        })

        resolved_count = sum(1 for r in results if r["resolved"])
        logger.info(
            f"  sim={similarity:.3f} {'✓' if resolved else '✗'} "
            f"(累计: {resolved_count}/{i+1}, {elapsed:.1f}s)"
        )

        time.sleep(1.2)

    n_resolved = sum(1 for r in results if r["resolved"])
    total = len(questions)
    resolve_rate = n_resolved / total if total > 0 else 0.0
    avg_sim = float(np.mean([r["similarity"] for r in results])) if results else 0.0

    report = {
        "experiment": "E8_SWEbench_Lite_v510",
        "model": "deepseek-chat",
        "n_instances": total,
        "n_resolved": n_resolved,
        "n_unresolved": total - n_resolved,
        "resolve_rate": round(resolve_rate, 4),
        "avg_similarity": round(avg_sim, 4),
        "total_time_min": round(total_time / 60, 1),
        "resolve_threshold": 0.60,
        "repo_stats": {
            repo: {
                "total": s["total"],
                "resolved": s["resolved"],
                "resolve_rate": round(s["resolved"] / s["total"], 4) if s["total"] > 0 else 0.0,
                "avg_similarity": round(float(np.mean(s["similarities"])), 4),
            }
            for repo, s in sorted(repo_stats.items())
        },
        "results": results,
    }

    out_path = RESULTS_DIR / "swebench_lite_v510.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {out_path}")

    return report


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite 基准评测 (v5.1)")
    parser.add_argument("--limit", type=int, default=10,
                        help="最大实例数 (0=全部, 默认 10)")
    parser.add_argument("--repo", type=str, default=None,
                        help="过滤仓库 (如 django/django)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SWE-bench Lite 基准评测 v5.1.0")
    logger.info(f"  limit={args.limit}, repo={args.repo or '全部'}")
    logger.info("=" * 60)

    questions = load_swebench_data(
        limit=args.limit if args.limit > 0 else 0,
        repo_filter=args.repo,
    )

    if not questions:
        logger.error("没有加载到任何题目！")
        return

    report = run_swebench_evaluation(questions)

    print("\n" + "=" * 60)
    print("SWE-bench Lite 评测结果摘要")
    print("=" * 60)
    print(f"  总实例数:     {report['n_instances']}")
    print(f"  解决:         {report['n_resolved']}")
    print(f"  未解决:       {report['n_unresolved']}")
    print(f"  解决率:       {report['resolve_rate']*100:.1f}%")
    print(f"  平均相似度:   {report['avg_similarity']:.4f}")
    print(f"  总用时:       {report['total_time_min']} 分钟")
    print()
    print("  仓库统计:")
    for repo, stats in report["repo_stats"].items():
        print(f"    {repo}: {stats['resolved']}/{stats['total']} "
              f"= {stats['resolve_rate']*100:.1f}% (avg_sim={stats['avg_similarity']:.3f})")


if __name__ == "__main__":
    main()

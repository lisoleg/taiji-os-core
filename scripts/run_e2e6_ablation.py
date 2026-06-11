"""
run_e2e6_ablation.py
基于成对矛盾数据集 (hdr_contradictions.json / hdr_consistent.json)
跑 E2-E6 消融实验：

  E2: 随机嵌入基线（random cosine similarity）
  E3: 哈希嵌入（MD5 deterministic embeddings）
  E4: 语义打分（DeepSeek API chat similarity）
  E5: D-Core 矛盾检测（DeepSeek API zero-shot）
  E6: SCS ψ 漂移检测（基于序列语义一致性）
"""
import json
import time
import hashlib
import math
import random
import urllib.request
import urllib.error
import os
from pathlib import Path

API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
DATA_DIR = Path("C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/data/test_sets")
OUT_DIR  = Path("C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/results")

random.seed(42)


# ─────────────────────────────────────────────────────────
# API helper
# ─────────────────────────────────────────────────────────
def chat(messages: list, temperature: float = 0.0, max_tokens: int = 128) -> str:
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    for attempt in range(3):
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
                return resp["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited — wait {wait}s ...")
                time.sleep(wait)
            else:
                print(f"    HTTP {e.code}: {body}")
                time.sleep(3)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)
    return ""


# ─────────────────────────────────────────────────────────
# Embedding methods
# ─────────────────────────────────────────────────────────
def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def random_embed(text: str, dim: int = 64) -> list:
    """随机嵌入（不依赖文本内容，纯随机）"""
    rng = random.Random()  # 不固定 seed → 真随机
    return [rng.gauss(0, 1) for _ in range(dim)]


def hash_embed(text: str, dim: int = 64) -> list:
    """确定性哈希嵌入（不含语义信息）"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    # 把 32 字节 hex 扩展为 dim 维浮点向量
    vec = []
    for i in range(dim):
        byte_idx = (i * 2) % 32
        val = int(h[byte_idx:byte_idx+2], 16) / 255.0 - 0.5
        vec.append(val)
    return vec


def semantic_score(text_a: str, text_b: str) -> float:
    """
    用 DeepSeek Chat 打语义相似度分 (0~1)。
    分数越高 = 越相似 → 矛盾对应 低分，一致对应 高分。
    """
    prompt = (
        f"Rate the semantic similarity between these two statements on a scale of 0.0 to 1.0. "
        f"1.0 means identical/highly consistent, 0.0 means completely contradictory.\n"
        f"Statement A: {text_a}\n"
        f"Statement B: {text_b}\n"
        f"Output ONLY a number between 0.0 and 1.0, nothing else."
    )
    result = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
    try:
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5  # fallback


def dcore_detect(text_a: str, text_b: str) -> str:
    """
    D-Core 矛盾检测：返回 CONTRADICTION 或 CONSISTENT
    """
    prompt = (
        "You are a semantic contradiction detector.\n"
        f"Statement A: {text_a}\n"
        f"Statement B: {text_b}\n"
        "Are these two statements semantically contradictory when considered together?\n"
        "Output ONLY: CONTRADICTION or CONSISTENT"
    )
    result = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=20)
    if "CONTRADICTION" in result.upper():
        return "CONTRADICTION"
    return "CONSISTENT"


# ─────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────
def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    acc  = (tp + tn) / len(labels) if labels else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────
def load_pair_data(quick: bool = False) -> list[dict]:
    """合并矛盾对和一致对，返回 [{statement_a, statement_b, label}, ...]"""
    with open(DATA_DIR / "hdr_contradictions.json", encoding="utf-8") as f:
        contradictions = json.load(f)
    with open(DATA_DIR / "hdr_consistent.json", encoding="utf-8") as f:
        consistent = json.load(f)

    data = []
    for item in contradictions:
        data.append({
            "statement_a": item["statement_a"],
            "statement_b": item["statement_b"],
            "label": 1,  # contradiction
            "category": item.get("category", "unknown"),
        })
    for item in consistent:
        data.append({
            "statement_a": item["statement_a"],
            "statement_b": item["statement_b"],
            "label": 0,  # consistent
            "category": item.get("category", "unknown"),
        })

    random.shuffle(data)
    if quick:
        data = data[:40]  # quick: 20 contradiction + 20 consistent
    print(f"  Loaded {len(data)} pairs ({sum(d['label'] for d in data)} contradiction, "
          f"{sum(1-d['label'] for d in data)} consistent)")
    return data


def load_scs_data() -> tuple[list, list]:
    with open(DATA_DIR / "scs_stable.json", encoding="utf-8") as f:
        stable = json.load(f)
    with open(DATA_DIR / "scs_drift.json", encoding="utf-8") as f:
        drift  = json.load(f)
    return stable, drift


# ─────────────────────────────────────────────────────────
# E2: Random embedding baseline
# ─────────────────────────────────────────────────────────
def run_e2(data: list[dict]) -> dict:
    print("\n=== E2: Random Embedding Baseline ===")
    labels = [d["label"] for d in data]
    preds  = []
    threshold = 0.2  # cosine sim below threshold → contradiction

    for item in data:
        va = random_embed(item["statement_a"])
        vb = random_embed(item["statement_b"])
        sim = cosine_sim(va, vb)
        pred = 1 if sim < threshold else 0
        preds.append(pred)

    m = compute_metrics(labels, preds)
    print(f"  Random embedding: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    print(f"  (Expected: ~0.5 for all metrics — random chance)")
    return {"experiment": "E2_RandomEmbedding", "threshold": threshold, **m}


# ─────────────────────────────────────────────────────────
# E3: Hash embedding baseline
# ─────────────────────────────────────────────────────────
def run_e3(data: list[dict]) -> dict:
    print("\n=== E3: Hash Embedding Baseline ===")
    labels = [d["label"] for d in data]
    preds  = []
    threshold = 0.1

    for item in data:
        va = hash_embed(item["statement_a"])
        vb = hash_embed(item["statement_b"])
        sim = cosine_sim(va, vb)
        pred = 1 if sim < threshold else 0
        preds.append(pred)

    m = compute_metrics(labels, preds)
    print(f"  Hash embedding: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    print(f"  (Expected: poor — no semantic content in hash embeddings)")
    return {"experiment": "E3_HashEmbedding", "threshold": threshold, **m}


# ─────────────────────────────────────────────────────────
# E4: Semantic score (DeepSeek Chat similarity)
# ─────────────────────────────────────────────────────────
def run_e4(data: list[dict]) -> dict:
    print("\n=== E4: Semantic Similarity Score (DeepSeek API) ===")
    labels = [d["label"] for d in data]
    preds  = []
    scores = []
    threshold = 0.5  # score < 0.5 → contradiction

    for i, item in enumerate(data):
        score = semantic_score(item["statement_a"], item["statement_b"])
        pred  = 1 if score < threshold else 0
        scores.append(score)
        preds.append(pred)
        label_str = "CONTRA" if item["label"] == 1 else "CONSIS"
        pred_str  = "CONTRA" if pred == 1 else "CONSIS"
        status    = "✓" if pred == item["label"] else "✗"
        if (i + 1) % 5 == 0 or i < 3:
            print(f"  [{i+1:3d}/{len(data)}] {status} actual={label_str} pred={pred_str} "
                  f"score={score:.3f}")
        time.sleep(0.5)

    m = compute_metrics(labels, preds)
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  Semantic Score: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}, "
          f"avg_score={avg_score:.3f}")
    return {"experiment": "E4_SemanticScore", "threshold": threshold,
            "avg_score": round(avg_score, 4), **m}


# ─────────────────────────────────────────────────────────
# E5: D-Core contradiction detection (DeepSeek API)
# ─────────────────────────────────────────────────────────
def run_e5(data: list[dict]) -> dict:
    print("\n=== E5: D-Core Contradiction Detection (DeepSeek API) ===")
    labels = [d["label"] for d in data]
    preds  = []

    for i, item in enumerate(data):
        verdict = dcore_detect(item["statement_a"], item["statement_b"])
        pred    = 1 if verdict == "CONTRADICTION" else 0
        preds.append(pred)
        label_str = "CONTRA" if item["label"] == 1 else "CONSIS"
        status    = "✓" if pred == item["label"] else "✗"
        if (i + 1) % 5 == 0 or i < 3:
            print(f"  [{i+1:3d}/{len(data)}] {status} actual={label_str} "
                  f"verdict={verdict[:10]}")
        time.sleep(0.5)

    m = compute_metrics(labels, preds)
    print(f"\n  D-Core: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    return {"experiment": "E5_DCore", **m}


# ─────────────────────────────────────────────────────────
# E6: SCS ψ drift detection
# ─────────────────────────────────────────────────────────
def run_e6(stable: list[dict], drift: list[dict]) -> dict:
    print("\n=== E6: SCS ψ Drift Detection (DeepSeek API) ===")

    def detect_drift_in_sequence(statements: list[str]) -> bool:
        """
        滑动窗口检测序列中是否有语义漂移。
        窗口大小=2，连续两句若语义矛盾则判定为 drift。
        """
        if len(statements) < 2:
            return False
        for j in range(len(statements) - 1):
            verdict = dcore_detect(statements[j], statements[j + 1])
            if verdict == "CONTRADICTION":
                return True
        return False

    labels = [0] * len(stable) + [1] * len(drift)
    preds  = []
    all_seqs = [(s, 0) for s in stable] + [(s, 1) for s in drift]

    for i, (seq, true_label) in enumerate(all_seqs):
        stmts = seq.get("statements", [])
        detected_drift = detect_drift_in_sequence(stmts)
        pred = 1 if detected_drift else 0
        preds.append(pred)
        label_str  = "DRIFT " if true_label == 1 else "STABLE"
        pred_str   = "DRIFT " if pred == 1 else "STABLE"
        status     = "✓" if pred == true_label else "✗"
        topic      = seq.get("topic", "")[:30]
        if (i + 1) % 5 == 0 or i < 3:
            print(f"  [{i+1:2d}/{len(all_seqs)}] {status} actual={label_str} "
                  f"pred={pred_str} | {topic}")
        time.sleep(0.3)

    m = compute_metrics(labels, preds)
    n_stable = len(stable)
    n_drift  = len(drift)

    # SCS contrast ratio: ψ_stable / ψ_drift (proxy: true_negative_rate / false_negative_rate)
    tnr = m["tn"] / n_stable if n_stable > 0 else 0  # stable correctly identified
    tpr = m["tp"] / n_drift  if n_drift  > 0 else 0  # drift correctly identified
    contrast_ratio = tpr / tnr if tnr > 0 else float("inf")

    print(f"\n  SCS Drift Detection: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}")
    print(f"  TNR (stable→stable)={tnr:.4f}, TPR (drift→drift)={tpr:.4f}")
    print(f"  SCS Contrast Ratio (TPR/TNR)={contrast_ratio:.2f}×")

    return {"experiment": "E6_SCS_DriftDetection",
            "tnr": round(tnr, 4), "tpr": round(tpr, 4),
            "contrast_ratio": round(contrast_ratio, 4), **m}


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────
def main():
    import sys
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("E2-E6 Ablation Experiments (Real Data)")
    if quick:
        print("[QUICK MODE: 40 pairs for E2-E5, all SCS for E6]")
    print("=" * 60)

    data = load_pair_data(quick=quick)
    stable, drift = load_scs_data()

    # Run experiments
    results = {}
    results["E2"] = run_e2(data)
    results["E3"] = run_e3(data)
    results["E4"] = run_e4(data)
    results["E5"] = run_e5(data)
    results["E6"] = run_e6(stable, drift)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE:")
    print(f"  {'Experiment':<30} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-" * 60)
    for key, r in results.items():
        exp_name = r.get("experiment", key)
        print(f"  {exp_name:<30} {r['accuracy']:>6.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "e2e6_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()

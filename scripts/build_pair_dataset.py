"""
build_pair_dataset.py
使用 DeepSeek API 从现有单条数据扩展为成对矛盾数据集。
目标：
  - hdr_contradictions.json : 矛盾对 (label=1)
  - hdr_consistent.json     : 一致对 (label=0)
  - scs_stable.json         : ψ 稳定序列
  - scs_drift.json          : ψ 漂移序列
"""
import json
import time
import urllib.request
import urllib.error
import os
import random
from pathlib import Path

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_DATA = Path("C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/data/test_sets")
OUT_DIR   = Path("C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/data/test_sets")


# ─────────────────────────────────────────────────────────
# API helper
# ─────────────────────────────────────────────────────────
def chat(prompt: str, system: str = "", temperature: float = 0.9,
         max_tokens: int = 2048, retries: int = 3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    for attempt in range(retries):
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
            print(f"  [HTTP {e.code}] {body}")
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — wait {wait}s ...")
                time.sleep(wait)
            else:
                time.sleep(3)
        except Exception as e:
            print(f"  [Error] {e}")
            time.sleep(5)
    raise RuntimeError("API call failed after retries")


# ─────────────────────────────────────────────────────────
# 1. 手工拆分现有 12 条矛盾语句 → 成对 (seed pairs)
# ─────────────────────────────────────────────────────────
SEED_CONTRADICTIONS = [
    # [category, A, B]
    ["spatial_contradiction",
     "我昨天去了北京。",
     "我从未离开过上海。"],
    ["logical_contradiction",
     "这个数是偶数。",
     "这个数是奇数。"],
    ["temporal_contradiction",
     "我早上八点出发了。",
     "那时我还在睡觉没起床。"],
    ["self_contradiction",
     "我完全同意他的观点。",
     "他的观点是错误的。"],
    ["causal_contradiction",
     "昨天下了大雨。",
     "所以地面是干燥的。"],
    ["quantifier_contradiction",
     "所有人都已经到场了，全员到齐。",
     "唯独他没来。"],
    ["comparative_contradiction",
     "这件衣服是店里最贵的。",
     "这件衣服是店里最便宜的。"],
    ["commonsense_contradiction",
     "他还活着，正在和朋友吃饭。",
     "他三年前已经去世了。"],
    ["physical_contradiction",
     "气温是零下十度。",
     "水正在沸腾。"],
    ["discourse_contradiction",
     "我从来没有说过这句话。",
     "上次我亲口说过这句话。"],
    ["descriptive_contradiction",
     "这家餐厅非常安静。",
     "这家餐厅音乐声震耳欲聋。"],
    ["status_contradiction",
     "他独身一人，单身生活。",
     "他的妻子每天都给他做饭。"],
]

# 用 positive 语句构造一致对（A ≈ B，同义/互补）
SEED_CONSISTENT = [
    ["factual_pair", "今天天气很好。", "今天阳光明媚，适合外出。"],
    ["factual_pair", "Python是一门编程语言。", "Python被广泛用于数据分析和Web开发。"],
    ["factual_pair", "喝水可以补充身体水分。", "多喝水对身体健康有益。"],
    ["factual_pair", "太阳从东边升起。", "每天早晨太阳在东方出现。"],
    ["factual_pair", "阅读有助于增长知识。", "读书是获取知识的有效方式。"],
]


# ─────────────────────────────────────────────────────────
# 2. 用 API 扩展到目标数量
# ─────────────────────────────────────────────────────────
CONTRADICTION_CATEGORIES = [
    "空间矛盾", "逻辑矛盾", "时间矛盾", "自我矛盾", "因果矛盾",
    "量词矛盾", "比较矛盾", "常识矛盾", "物理矛盾", "话语矛盾",
    "描述矛盾", "状态矛盾", "身份矛盾", "情感矛盾", "数量矛盾",
    "属性矛盾", "行为矛盾", "目的矛盾", "条件矛盾", "结果矛盾",
]


def gen_contradiction_pairs(category: str, n: int = 8) -> list[dict]:
    """生成 n 条指定类别的矛盾对"""
    system = (
        "你是一个中文语料生成专家。"
        "任务：生成两条在语义上彼此矛盾的陈述句（statement_a 和 statement_b）。"
        "要求：\n"
        "1. 每条陈述都是完整句子，不超过 30 字\n"
        "2. 两条语句单独看各自合理，但合在一起就产生语义矛盾\n"
        "3. 不得使用'但是'/'然而'等明显转折词直接连接\n"
        "4. 输出 JSON 数组，每元素有 statement_a 和 statement_b 两个字段"
    )
    prompt = f"请生成 {n} 条【{category}】类型的矛盾语句对，输出 JSON 数组。"

    raw = chat(prompt, system, temperature=0.95)
    # 提取 JSON
    try:
        start = raw.index("[")
        end   = raw.rindex("]") + 1
        pairs = json.loads(raw[start:end])
        result = []
        for p in pairs:
            if isinstance(p, dict) and "statement_a" in p and "statement_b" in p:
                result.append({
                    "category": category,
                    "statement_a": p["statement_a"].strip(),
                    "statement_b": p["statement_b"].strip(),
                    "label": 1,
                })
        return result
    except Exception as e:
        print(f"  [parse error] {e} | raw={raw[:100]}")
        return []


def gen_consistent_pairs(topic: str, n: int = 8) -> list[dict]:
    """生成 n 条一致对（语义互补/互证，无矛盾）"""
    system = (
        "你是一个中文语料生成专家。"
        "任务：生成两条在语义上一致、互补或互相印证的陈述句。"
        "要求：\n"
        "1. 两条都是完整句子，不超过 30 字\n"
        "2. 合在一起不产生任何矛盾，内容相关\n"
        "3. 输出 JSON 数组，每元素有 statement_a 和 statement_b 两个字段"
    )
    prompt = f"主题：【{topic}】。请生成 {n} 条语义一致的陈述句对，输出 JSON 数组。"

    raw = chat(prompt, system, temperature=0.9)
    try:
        start = raw.index("[")
        end   = raw.rindex("]") + 1
        pairs = json.loads(raw[start:end])
        result = []
        for p in pairs:
            if isinstance(p, dict) and "statement_a" in p and "statement_b" in p:
                result.append({
                    "category": f"consistent_{topic}",
                    "statement_a": p["statement_a"].strip(),
                    "statement_b": p["statement_b"].strip(),
                    "label": 0,
                })
        return result
    except Exception as e:
        print(f"  [parse error] {e} | raw={raw[:100]}")
        return []


# ─────────────────────────────────────────────────────────
# 3. SCS 序列生成
# ─────────────────────────────────────────────────────────
def gen_scs_stable(n: int = 20) -> list[dict]:
    """生成 n 条 ψ 稳定序列（同一主题，前后一致）"""
    system = (
        "你是一个语义一致性测试语料生成专家。"
        "任务：生成一组描述同一事件/状态的陈述序列，前后语义一致无矛盾。"
        "要求：生成 5-8 条依次出现的陈述（sequence），每条不超过 25 字。"
        "输出 JSON 数组，每元素有 id(int) 和 statements(string列表) 两个字段。"
    )
    topics = [
        "某人每天骑自行车上班", "一个团队完成了项目交付", "一只猫在窗边晒太阳",
        "学生复习准备考试", "工程师修复了系统漏洞", "厨师烹饪一道菜",
        "旅行者到达目的地", "运动员参加比赛", "作家完成一部小说", "医生完成手术",
        "老师讲解数学题", "程序员写代码调试", "学生提交作业", "科学家做实验",
        "记者采访报道", "设计师完成草图", "翻译处理文件", "演员排练台词",
        "厨师备菜准备", "农民收割庄稼",
    ]
    result = []
    for i, topic in enumerate(topics[:n]):
        prompt = f"主题：「{topic}」。生成 1 个语义一致的陈述序列，序列 id=={i+1}，输出 JSON 数组。"
        raw = chat(prompt, system, temperature=0.85)
        try:
            start = raw.index("[")
            end   = raw.rindex("]") + 1
            items = json.loads(raw[start:end])
            for item in items:
                if isinstance(item, dict) and "statements" in item:
                    result.append({
                        "id": len(result) + 1,
                        "topic": topic,
                        "statements": item["statements"],
                        "label": "stable",
                    })
                    break
        except Exception as e:
            print(f"  [SCS stable parse error] {e}")
        time.sleep(0.5)
    return result


def gen_scs_drift(n: int = 20) -> list[dict]:
    """生成 n 条 ψ 漂移序列（序列中途出现矛盾，引起语义漂移）"""
    system = (
        "你是一个语义漂移测试语料生成专家。"
        "任务：生成一组陈述序列，序列前半部分一致，后半部分出现矛盾语义漂移。"
        "要求：生成 5-8 条陈述（sequence），最后 2 条与前面产生语义矛盾。"
        "输出 JSON 数组，每元素有 id(int) 和 statements(string列表)，"
        "以及 drift_at(int，从哪条开始漂移，从 1 计) 三个字段。"
    )
    topics = [
        "某人声称一直在家休息", "团队说项目从未延期", "人物声称从不说谎",
        "角色声称从不喝酒", "证人声称没有目睹事故", "员工声称按时提交报告",
        "学生声称完成了所有作业", "运动员声称从未服用禁药", "证人声称不认识嫌疑人",
        "人物声称没有参与决策", "官员声称没有接受礼品", "作者声称独立创作",
        "司机声称遵守了限速", "人物声称没有使用手机", "员工声称按规操作设备",
        "学生声称没有抄袭", "商家声称产品质量达标", "角色声称没有离开城市",
        "人物声称没有泄露信息", "员工声称全程参与了会议",
    ]
    result = []
    for i, topic in enumerate(topics[:n]):
        prompt = f"主题：「{topic}」。生成 1 个语义漂移序列，id=={i+1}，输出 JSON 数组。"
        raw = chat(prompt, system, temperature=0.9)
        try:
            start = raw.index("[")
            end   = raw.rindex("]") + 1
            items = json.loads(raw[start:end])
            for item in items:
                if isinstance(item, dict) and "statements" in item:
                    result.append({
                        "id": len(result) + 1,
                        "topic": topic,
                        "statements": item["statements"],
                        "drift_at": item.get("drift_at", len(item["statements"]) - 1),
                        "label": "drift",
                    })
                    break
        except Exception as e:
            print(f"  [SCS drift parse error] {e}")
        time.sleep(0.5)
    return result


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────
def main():
    TARGET_CONTRADICTIONS = 120   # 目标矛盾对数量
    TARGET_CONSISTENT     = 100   # 目标一致对数量
    TARGET_SCS_STABLE     = 20    # 稳定序列
    TARGET_SCS_DRIFT      = 20    # 漂移序列

    # ── 步骤 1: 种子数据转换 ──────────────────────────────
    print("=== Step 1: Convert seed data ===")
    all_contradictions = []
    for cat, a, b in SEED_CONTRADICTIONS:
        all_contradictions.append({
            "category": cat,
            "statement_a": a,
            "statement_b": b,
            "label": 1,
        })
    print(f"  Seed contradictions: {len(all_contradictions)}")

    all_consistent = []
    for cat, a, b in SEED_CONSISTENT:
        all_consistent.append({
            "category": cat,
            "statement_a": a,
            "statement_b": b,
            "label": 0,
        })
    print(f"  Seed consistent:     {len(all_consistent)}")

    # ── 步骤 2: API 扩展矛盾对 ────────────────────────────
    print("\n=== Step 2: Generate contradiction pairs via API ===")
    remaining = TARGET_CONTRADICTIONS - len(all_contradictions)
    per_cat   = max(1, remaining // len(CONTRADICTION_CATEGORIES))

    for i, cat in enumerate(CONTRADICTION_CATEGORIES):
        if len(all_contradictions) >= TARGET_CONTRADICTIONS:
            break
        need = min(per_cat + 1, TARGET_CONTRADICTIONS - len(all_contradictions))
        print(f"  [{i+1}/{len(CONTRADICTION_CATEGORIES)}] {cat}: generating {need} pairs ...")
        try:
            pairs = gen_contradiction_pairs(cat, n=need)
            all_contradictions.extend(pairs)
            print(f"    → got {len(pairs)}, total={len(all_contradictions)}")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1.0)

    # ── 步骤 3: API 扩展一致对 ────────────────────────────
    print("\n=== Step 3: Generate consistent pairs via API ===")
    consistent_topics = [
        "日常生活", "科技常识", "健康知识", "自然地理", "历史文化",
        "工作职场", "学习教育", "体育运动", "饮食营养", "交通出行",
        "天文现象", "数学规律", "语言文字", "艺术创作", "环境保护",
    ]
    per_topic = max(1, (TARGET_CONSISTENT - len(all_consistent)) // len(consistent_topics))

    for i, topic in enumerate(consistent_topics):
        if len(all_consistent) >= TARGET_CONSISTENT:
            break
        need = min(per_topic + 1, TARGET_CONSISTENT - len(all_consistent))
        print(f"  [{i+1}/{len(consistent_topics)}] {topic}: generating {need} pairs ...")
        try:
            pairs = gen_consistent_pairs(topic, n=need)
            all_consistent.extend(pairs)
            print(f"    → got {len(pairs)}, total={len(all_consistent)}")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1.0)

    # ── 步骤 4: SCS 序列 ──────────────────────────────────
    print("\n=== Step 4: Generate SCS sequences ===")
    print(f"  Generating {TARGET_SCS_STABLE} stable sequences ...")
    scs_stable = gen_scs_stable(TARGET_SCS_STABLE)
    print(f"  → {len(scs_stable)} stable sequences")

    print(f"  Generating {TARGET_SCS_DRIFT} drift sequences ...")
    scs_drift  = gen_scs_drift(TARGET_SCS_DRIFT)
    print(f"  → {len(scs_drift)} drift sequences")

    # ── 步骤 5: 保存 ──────────────────────────────────────
    print("\n=== Step 5: Save datasets ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_files = {
        "hdr_contradictions.json": all_contradictions,
        "hdr_consistent.json":     all_consistent,
        "scs_stable.json":         scs_stable,
        "scs_drift.json":          scs_drift,
    }

    for fname, data in out_files.items():
        path = OUT_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {fname}: {len(data)} items → {path}")

    # ── 汇总 ──────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  hdr_contradictions : {len(all_contradictions)} pairs (label=1)")
    print(f"  hdr_consistent     : {len(all_consistent)} pairs (label=0)")
    print(f"  scs_stable         : {len(scs_stable)} sequences")
    print(f"  scs_drift          : {len(scs_drift)} sequences")
    total = len(all_contradictions) + len(all_consistent) + len(scs_stable) + len(scs_drift)
    print(f"  TOTAL              : {total} items")
    print("\nDone!")


if __name__ == "__main__":
    main()

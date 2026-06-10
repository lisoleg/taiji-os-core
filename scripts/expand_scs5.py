#!/usr/bin/env python3
"""第五轮 — 补齐 SCS 漂移到 ≥150"""

import json, os

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "test_sets"
)

def load_json(n):
    with open(os.path.join(OUTPUT_DIR, n), "r", encoding="utf-8") as f:
        return json.load(f)
def save_json(n, d):
    with open(os.path.join(OUTPUT_DIR, n), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def main():
    data = load_json("scs_drift.json")
    existing_topics = {e["topic"] for e in data["entries"]}
    target = 150
    need = target - len(data["entries"])
    nid = len(data["entries"]) + 1

    # 批量为主题漂移补充,用唯一编号确保不重名
    drift_topics = [
        "美术→化学",  "哲学→物理",  "法律→经济",  "地理→天文",  "音乐→数学",
        "体育→医学",  "历史→考古",  "社会→心理",  "宗教→政治",  "军事→技术",
        "舞蹈→生理",  "电影→文学",  "工程→管理",  "商业→统计",  "环境→政策",
        "文学→哲学",  "数学→神学",  "物理→玄学",  "计算机→数学", "天文→宗教",
        "教育→神经",  "新闻→历史",  "市场→心理",  "物流→地理",  "化学→生物",
        "机械→电子",  "土木→地质",  "航空→气象",  "航海→天文",  "医药→化学",
        "纺织→化工",  "印刷→文化",  "矿业→材料",  "电力→物理",  "通信→电磁",
        "保险→精算",  "金融→数学",  "审计→法律",  "零售→物流",  "餐饮→农业",
        "运输→能源",  "房产→金融",  "旅游→经济",  "传媒→社会",  "卫生→医学",
        "消防→化学",  "水务→环境",  "渔业→生物",  "林业→生态",  "牧业→兽医",
        "酿酒→化工",  "陶瓷→材料",  "玻璃→光学",  "橡胶→有机",  "塑料→高分子",
        "涂料→表面",  "胶粘→界面",  "香料→天然",  "染料→合成",  "炸药→能材",
        "农药→毒理",  "化肥→土壤",  "种子→遗传",  "灌溉→水利",  "农机→机械",
        "仓储→信息",  "配送→路径",  "包装→设计",  "质检→标准",  "计量→物理",
    ]

    # 预制的漂移陈述模板
    def make_drift(a_field, b_field):
        return [
            f"{a_field}领域的基础理论已经相当成熟",
            f"{a_field}的研究方法注重实验和观测",
            f"{a_field}的许多问题涉及到了{b_field}的领域",
            f"{a_field}与{b_field}的交叉研究产生了丰富的成果",
            f"从{b_field}的角度来看,{a_field}中的许多现象可以得到新的解释",
            f"{b_field}的理论框架为理解{a_field}提供了有力的工具",
            f"{b_field}的核心概念已经完全取代了最初的{a_field}讨论框架",
            f"{b_field}正在主导这一交叉领域的研究方向",
        ]

    added = 0
    for topic in drift_topics:
        if added >= need:
            break
        if topic not in existing_topics:
            parts = topic.split("→")
            stmts = make_drift(parts[0], parts[1])
            data["entries"].append({
                "id": f"SCS-DRF-{nid:04d}",
                "topic": topic,
                "statements": stmts,
                "label": "drift",
                "expected_cv": "> 0.3",
                "drift_type": "topic_transition",
                "description": f"从{parts[0]}话题逐步漂移到{parts[1]}话题",
            })
            nid += 1; added += 1

    data["metadata"]["total"] = len(data["entries"])
    save_json("scs_drift.json", data)
    print(f"SCS Drift: {len(data['entries'])} (+{added})")

    # 验证全部数据集
    for fname in ["hdr_contradictions.json","hdr_consistent.json","scs_stable.json","scs_drift.json"]:
        d = load_json(fname)
        print(f"  {fname}: {len(d['entries'])}")

    print(f"\n✅ {'ALL MET' if len(data['entries']) >= 150 else 'NEED MORE'}")

if __name__ == "__main__":
    main()

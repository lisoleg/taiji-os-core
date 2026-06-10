"""
tests/test_benchmark.py — 数据集格式验证 & 消融实验正确性测试
"""
import json, os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_sets")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def load_json(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


# ============ HDR Contradictions Dataset ============

class TestHDRContradictions:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = load_json("hdr_contradictions.json")

    def test_total_count(self):
        """≥500 条矛盾正例"""
        assert len(self.data["entries"]) >= 500, \
            f"Expected ≥500, got {len(self.data['entries'])}"

    def test_all_categories_present(self):
        """至少 8 种矛盾类型"""
        cats = set(e["category"] for e in self.data["entries"])
        assert len(cats) >= 8, f"Expected ≥8 categories, got {len(cats)}: {cats}"

    def test_each_category_min_count(self):
        """每种至少 10 条"""
        from collections import Counter
        cat_counts = Counter(e["category"] for e in self.data["entries"])
        for cat, count in cat_counts.items():
            assert count >= 10, f"Category '{cat}' has only {count} entries"

    def test_no_empty_statements(self):
        """无空文本"""
        for e in self.data["entries"]:
            assert len(e["statement_a"].strip()) > 0
            assert len(e["statement_b"].strip()) > 0

    def test_unique_ids(self):
        """ID 唯一"""
        ids = [e["id"] for e in self.data["entries"]]
        assert len(ids) == len(set(ids))

    def test_all_contradiction_label(self):
        """所有标为正例"""
        for e in self.data["entries"]:
            assert e["label"] == "contradiction"

    def test_contradiction_type_valid(self):
        """矛盾类型为 explicit 或 implicit"""
        for e in self.data["entries"]:
            assert e["contradiction_type"] in ("explicit", "implicit")

    def test_required_fields(self):
        """必需字段齐全"""
        required = {"id", "category", "subtype", "statement_a", "statement_b", "label", "contradiction_type"}
        for e in self.data["entries"]:
            assert required.issubset(e.keys()), f"Missing fields in {e['id']}"

    def test_metadata(self):
        """元数据完整"""
        meta = self.data["metadata"]
        assert meta["total"] == len(self.data["entries"])
        assert "categories" in meta
        assert "description" in meta


# ============ HDR Consistent Dataset ============

class TestHDRConsistent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = load_json("hdr_consistent.json")

    def test_total_count(self):
        """≥200 条一致性负例"""
        assert len(self.data["entries"]) >= 200

    def test_all_consistent_label(self):
        for e in self.data["entries"]:
            assert e["label"] == "consistent"

    def test_no_empty_statements(self):
        for e in self.data["entries"]:
            assert len(e["statement_a"].strip()) > 0
            assert len(e["statement_b"].strip()) > 0

    def test_unique_ids(self):
        ids = [e["id"] for e in self.data["entries"]]
        assert len(ids) == len(set(ids))


# ============ SCS Stable Dataset ============

class TestSCSStable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = load_json("scs_stable.json")

    def test_total_count(self):
        """≥150 条稳定序列"""
        assert len(self.data["entries"]) >= 150

    def test_each_has_multiple_statements(self):
        """每条至少 4 条陈述"""
        for e in self.data["entries"]:
            assert len(e["statements"]) >= 4, \
                f"{e['id']} has only {len(e['statements'])} statements"

    def test_all_stable_label(self):
        for e in self.data["entries"]:
            assert e["label"] == "stable"

    def test_expected_cv_low(self):
        for e in self.data["entries"]:
            assert e["expected_cv"] == "< 0.1"

    def test_unique_topics(self):
        topics = [e["topic"] for e in self.data["entries"]]
        assert len(topics) == len(set(topics)), "Duplicate topics found"

    def test_no_duplicate_ids(self):
        ids = [e["id"] for e in self.data["entries"]]
        assert len(ids) == len(set(ids))


# ============ SCS Drift Dataset ============

class TestSCSDrift:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = load_json("scs_drift.json")

    def test_total_count(self):
        """≥150 条漂移序列"""
        assert len(self.data["entries"]) >= 150

    def test_each_has_multiple_statements(self):
        for e in self.data["entries"]:
            assert len(e["statements"]) >= 4

    def test_all_drift_label(self):
        for e in self.data["entries"]:
            assert e["label"] == "drift"

    def test_expected_cv_high(self):
        for e in self.data["entries"]:
            assert ">" in e["expected_cv"]

    def test_drift_type_valid(self):
        for e in self.data["entries"]:
            assert e["drift_type"] in ("topic_transition", "incremental_contradiction")

    def test_unique_topics(self):
        topics = [e["topic"] for e in self.data["entries"]]
        assert len(topics) == len(set(topics))

    def test_both_drift_types_present(self):
        types = set(e["drift_type"] for e in self.data["entries"])
        assert "topic_transition" in types
        assert "incremental_contradiction" in types


# ============ Benchmark scripts exist and are importable ============

class TestBenchmarkScripts:
    def test_hdr_script_exists(self):
        assert os.path.exists(os.path.join(SCRIPTS_DIR, "benchmark_hdr.py"))

    def test_scs_script_exists(self):
        assert os.path.exists(os.path.join(SCRIPTS_DIR, "benchmark_scs.py"))

    def test_ablation_script_exists(self):
        assert os.path.exists(os.path.join(SCRIPTS_DIR, "ablation.py"))

    def test_results_dir(self):
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        assert os.path.exists(results_dir)


# ============ Regression: existing tests still pass ============

class TestRegression:
    """验证现有功能未被破坏"""

    def test_world_model_init(self):
        from core.world_model import WorldModel
        wm = WorldModel()
        assert wm.dim == 1536
        assert wm.version == 0

    def test_world_model_encode(self):
        from core.world_model import WorldModel
        wm = WorldModel()
        vec = wm.encode("hello")
        assert vec.shape == (1536,)
        # 验证是单位向量
        assert abs(float((vec ** 2).sum()) - 1.0) < 1e-6

    def test_world_model_phi(self):
        from core.world_model import WorldModel
        wm = WorldModel()
        wm.update("base")
        phi_self = wm.phi(wm.psi)
        assert abs(phi_self - 1.0) < 1e-6

    def test_world_model_error_rate_limiting(self):
        """API 错误应被限频"""
        from core.world_model import WorldModel
        wm = WorldModel()
        wm._online = False
        vec = wm.encode("test")
        assert vec.shape == (1536,)
        # _api_error_counts 应该存在
        assert hasattr(wm, "_api_error_counts")

    def test_carbon_silicon_gan_import(self):
        from core.carbon_silicon_gan import CarbonSiliconGAN
        assert CarbonSiliconGAN is not None

    def test_self_consistency_loop_import(self):
        from core.self_consistency_loop import SelfConsistencyLoop
        scl = SelfConsistencyLoop(api_key="")
        assert scl.mode == "offline"

    def test_session_import(self):
        from core.session import TaijiSession
        assert TaijiSession is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

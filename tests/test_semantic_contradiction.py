"""
tests/test_semantic_contradiction.py — D-Core 语义矛盾检测测试
"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.self_consistency_loop import SelfConsistencyLoop, DCORE_SEMANTIC_PROMPT


class TestSelfConsistencyLoop:
    """D-Core = SelfConsistencyLoop 语义矛盾检测测试"""

    def test_init_offline(self):
        """无 API key 时自动切换到离线模式"""
        scl = SelfConsistencyLoop(api_key="")
        assert scl.mode == "offline"

    def test_init_online_placeholder_skipped(self):
        """模板变量 ${...} 不应被视为有效 API key"""
        scl = SelfConsistencyLoop(api_key="${DEEPSEEK_API_KEY}")
        assert scl.mode == "offline"

    def test_detect_contradiction_explicit(self):
        """显式矛盾词检测"""
        scl = SelfConsistencyLoop(api_key="")
        is_contra, verdict, method = scl.detect_contradiction(
            "系统正常运行", "系统存在矛盾"
        )
        assert is_contra
        assert verdict == "CONTRADICTION"
        assert method == "keyword"

    def test_detect_contradiction_inconsistent(self):
        """"不一致" 关键词检测"""
        scl = SelfConsistencyLoop(api_key="")
        is_contra, _, _ = scl.detect_contradiction(
            "数据准确", "数据前后不一致"
        )
        assert is_contra

    def test_detect_contradiction_conflict(self):
        """"冲突" 关键词检测"""
        scl = SelfConsistencyLoop(api_key="")
        is_contra, _, _ = scl.detect_contradiction(
            "时间10点", "时间冲突"
        )
        assert is_contra

    def test_detect_consistent_simple(self):
        """简单一致对 — 不含任何矛盾关键词"""
        scl = SelfConsistencyLoop(api_key="")
        is_contra, verdict, method = scl.detect_contradiction(
            "The weather is sunny today", "It will be warm this afternoon"
        )
        assert not is_contra
        assert verdict == "CONSISTENT"

    def test_contradiction_cache(self):
        """确定性缓存：相同输入应返回相同结果"""
        scl = SelfConsistencyLoop(api_key="")
        _, v1, _ = scl.detect_contradiction("猫在沙发上", "猫在厨房里")
        _, v2, m2 = scl.detect_contradiction("猫在沙发上", "猫在厨房里")
        assert v1 == v2
        assert m2 == "cache"

    def test_stats(self):
        """统计信息"""
        scl = SelfConsistencyLoop(api_key="")
        scl.detect_contradiction("A", "B")
        stats = scl.stats()
        assert stats["mode"] == "offline"
        assert stats["cache_size"] >= 1

    def test_semantic_prompt_format(self):
        """验证零样本 prompt 包含必要元素"""
        prompt = DCORE_SEMANTIC_PROMPT.format(
            statement_a="TEST_A", statement_b="TEST_B"
        )
        assert "TEST_A" in prompt
        assert "TEST_B" in prompt
        assert "CONTRADICTION" in prompt
        assert "CONSISTENT" in prompt
        assert "VERDICT:" in prompt

    def test_all_contradiction_keywords(self):
        """验证所有矛盾关键词都能被检测"""
        from core.self_consistency_loop import CONTRADICTION_KEYWORDS
        assert len(CONTRADICTION_KEYWORDS) >= 10
        scl = SelfConsistencyLoop(api_key="")
        for kw in ["矛盾", "不一致", "冲突", "相反", "contradiction"]:
            is_contra, _, _ = scl.detect_contradiction(
                "正常陈述", f"这个陈述包含{kw}"
            )
            assert is_contra, f"Keyword '{kw}' should trigger contradiction"

    def test_self_ref_contradiction(self):
        """自指矛盾：既是X又不是X"""
        scl = SelfConsistencyLoop(api_key="")
        is_contra, _, _ = scl.detect_contradiction(
            "这个既是正确的", "这个又不是正确的"
        )
        assert is_contra

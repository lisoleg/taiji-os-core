import os
import hashlib
import logging
import numpy as np
import openai
import yaml

from core.semantic_embedder import get_semantic_embedder, SemanticEmbedder

logger = logging.getLogger(__name__)


class WorldModel:
    """
    WorldModel: 维护全局语义状态向量 ψ（psi）。
    通过指数移动平均持续更新世界表示，并提供 Φ 相似度计算。

    编码策略（v4.6: 语义嵌入优先）：
    1. 优先使用本地 sentence-transformers 语义嵌入（384-dim MiniLM）
    2. 降级到 DeepSeek Embedding API（在线、语义精确）
    3. 最终回退到确定性哈希嵌入（离线/测试用）
    """

    def __init__(self, dim: int = 1536, config_path: str = "config.yaml"):
        self.dim = dim
        self.psi = np.zeros(dim)
        self.version = 0

        # ── v4.6: 本地语义嵌入（最高优先级）──
        self._semantic = get_semantic_embedder()

        # 读取 embedding 配置
        cfg = self._load_config(config_path)
        emb_cfg = cfg.get("embedding", {})
        self.emb_provider = emb_cfg.get("provider", "deepseek")
        self.emb_base_url = emb_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.emb_model = emb_cfg.get("model", "deepseek-embedding")
        api_key = os.environ.get(
            "DEEPSEEK_API_KEY", emb_cfg.get("api_key", "")
        )
        # 跳过未替换的模板变量
        if api_key and not api_key.startswith("${"):
            self.api_key = api_key
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.emb_base_url)
            self._online = True
        else:
            self.api_key = ""
            self._client = None
            self._online = False
        # API 错误计数器（按错误类型限频，单 session 同类型最多 3 次 warning）
        self._api_error_count: dict = {}

        # ── v4.6.1: 非零 ψ 初始化（避免 Φ-Gate 100% 误拒）──
        self._init_psi()

    def _init_psi(self):
        """用语义种子初始化 ψ，确保 phi() 首次计算非零。

        零向量 ψ 会导致 phi()=0，Φ-Gate 拒绝所有请求 → ψ 永远无法更新。
        解决方案：用种子短语编码初始化 ψ 为非零单位向量。
        """
        seed_text = "taiji world model semantic seed initialization v1"
        try:
            if self._semantic.is_available():
                psi_seed = self._encode_semantic(seed_text)
            elif self._online:
                psi_seed = self._encode_api(seed_text)
            else:
                psi_seed = self._encode_hash(seed_text)
        except Exception:
            psi_seed = self._encode_hash(seed_text)
        self.psi = psi_seed

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ------------------------------------------------------------------
    # 编码：语义嵌入 > API > 确定性哈希嵌入
    # ------------------------------------------------------------------

    def encode(self, text: str, force_hash: bool = False) -> np.ndarray:
        """将文本编码为语义向量。

        优先级：本地语义模型 > DeepSeek API > 哈希回退。

        Args:
            text: 输入文本。
            force_hash: 强制使用哈希嵌入（用于测试确定性）。

        Returns:
            numpy float32/float64 向量。
        """
        if force_hash:
            return self._encode_hash(text)

        # Priority 1: Local semantic embedder (v4.6)
        if self._semantic.is_available():
            return self._encode_semantic(text)

        # Priority 2: DeepSeek API
        if self._online:
            return self._encode_api(text)

        # Priority 3: Hash fallback
        return self._encode_hash(text)

    def _encode_semantic(self, text: str) -> np.ndarray:
        """使用本地语义模型编码。"""
        vec = self._semantic.encode(text)
        # Align dimension if needed
        if vec.shape[0] != self.dim:
            vec = self._project(vec, self.dim)
        return vec

    @property
    def embedding_dim(self) -> int:
        """当前激活的嵌入维度（语义模型优先）。"""
        if self._semantic.is_available():
            return self._semantic.dim
        return self.dim

    def _encode_api(self, text: str) -> np.ndarray:
        """调用 DeepSeek Embedding API。"""
        try:
            resp = self._client.embeddings.create(
                model=self.emb_model,
                input=text,
            )
            vec = np.array(resp.data[0].embedding, dtype=np.float64)
            # 如果 API 返回维度与 self.dim 不同，做投影对齐
            if vec.shape[0] != self.dim:
                vec = self._project(vec, self.dim)
            return vec
        except Exception as e:
            # API 调用失败，记录 warning（同类型错误不超过 3 次）
            error_key = type(e).__name__
            self._api_error_count[error_key] = (
                self._api_error_count.get(error_key, 0) + 1
            )
            if self._api_error_count[error_key] <= 3:
                logger.warning(
                    "WorldModel API fallback: %s (%s), "
                    "using hash embedding (occurrence #%d)",
                    error_key, str(e)[:120], self._api_error_count[error_key]
                )
            return self._encode_hash(text)

    def _encode_hash(self, text: str) -> np.ndarray:
        """
        确定性哈希嵌入（离线 / 测试用）。
        同一文本 → 同一向量，不同文本 → 不同向量。
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 用哈希字节做种子，生成确定性随机向量
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        # 单位化，与 API 返回的嵌入对齐
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    @staticmethod
    def _project(vec: np.ndarray, target_dim: int) -> np.ndarray:
        """线性投影：将任意维度向量映射到目标维度。"""
        src_dim = vec.shape[0]
        rng = np.random.default_rng(42)  # 固定种子保证确定性
        W = rng.standard_normal((target_dim, src_dim)) / np.sqrt(src_dim)
        return W @ vec

    # ------------------------------------------------------------------
    # ψ 更新 & Φ 计算
    # ------------------------------------------------------------------

    def update(self, text: str) -> None:
        """用新文本更新 ψ（EMA 衰减）。"""
        vec = self.encode(text)
        self.psi = 0.9 * self.psi + 0.1 * vec
        self.version += 1

    def phi(self, new_psi: np.ndarray) -> float:
        """计算当前 ψ 与候选向量的余弦相似度（Φ 值）。"""
        dot = np.dot(self.psi, new_psi)
        norm = (np.linalg.norm(self.psi) * np.linalg.norm(new_psi)) + 1e-8
        return float(dot / norm)

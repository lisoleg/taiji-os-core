class ClosureEnv:
    """
    ClosureEnv: 封装当前 AGI 进程的执行环境。
    持有 intent（当前目标）、history（对话历史）及任意附加上下文。
    """

    def __init__(self, intent: str = "idle"):
        self.intent = intent
        self.history: list = []
        self.context: dict = {}

    def push(self, role: str, content: str) -> None:
        """追加一条历史记录。"""
        self.history.append({"role": role, "content": content})

    def set_intent(self, intent: str) -> None:
        self.intent = intent

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "history": self.history,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClosureEnv":
        env = cls(intent=data.get("intent", "idle"))
        env.history = data.get("history", [])
        env.context = data.get("context", {})
        return env

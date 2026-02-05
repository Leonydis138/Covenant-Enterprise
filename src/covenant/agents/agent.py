from src.core.ethics_engine import EthicsEngine


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.ethics = EthicsEngine()

    def propose_action(self, action: dict):
        self.ethics.validate(action)
        return {"agent": self.name, "action": action, "status": "approved"}

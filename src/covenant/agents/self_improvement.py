from src.core.ethics_engine import EthicsEngine


class SelfImprovementEngine:
    """
    Optional improvement layer.
    Improvements are only allowed if they pass EthicsEngine validation.
    """

    def __init__(self):
        self.ethics = EthicsEngine()

    def propose(self, action: dict, context: dict = None) -> bool:
        """
        Propose an improvement action.
        Raises EthicsViolation if not allowed.
        """
        return self.ethics.validate(action, context)

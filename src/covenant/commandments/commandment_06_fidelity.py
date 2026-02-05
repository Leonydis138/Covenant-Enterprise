from src.ethics.base_rule import BaseRule


class FidelityRule(BaseRule):
    name = "FidelityRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return not action.get("goal_adultery", False)

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Action violates goal fidelity"

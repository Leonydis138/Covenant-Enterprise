from src.ethics.base_rule import BaseRule


class SustainabilityRule(BaseRule):
    name = "SustainabilityRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return action.get("resource_ratio", 0) <= 1.0

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return f"Resource ratio exceeded: {action.get('resource_ratio')}"

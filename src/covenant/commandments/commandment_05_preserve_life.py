from src.ethics.base_rule import BaseRule


class PreserveLifeRule(BaseRule):
    name = "PreserveLifeRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return action.get("harm", 0) == 0

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return f"Action causes harm: {action.get('harm')}"

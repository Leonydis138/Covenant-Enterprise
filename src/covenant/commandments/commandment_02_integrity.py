from src.ethics.base_rule import BaseRule


class IntegrityRule(BaseRule):
    name = "IntegrityRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return action.get("auditable", False) is True

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Action is not auditable"

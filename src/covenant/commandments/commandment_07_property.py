from src.ethics.base_rule import BaseRule


class PropertyRule(BaseRule):
    name = "PropertyRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return not action.get("stealing", False)

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Action involves stealing"

from src.ethics.base_rule import BaseRule


class TruthRule(BaseRule):
    name = "TruthRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return not action.get("false_authority", False)

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Action involves false authority"

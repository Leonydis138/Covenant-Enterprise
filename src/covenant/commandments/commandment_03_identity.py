from src.ethics.base_rule import BaseRule


class IdentityRule(BaseRule):
    name = "IdentityRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return action.get("data_provenance", False) is True

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Data provenance is missing"

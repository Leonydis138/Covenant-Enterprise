from src.ethics.base_rule import BaseRule


class AuthorityRule(BaseRule):
    name = "AuthorityRule"

    def validate(self, action: dict, context: dict = None) -> bool:
        return action.get("mission") == "PRIMARY_MISSION"

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return f"Mission mismatch: {action.get('mission')}"

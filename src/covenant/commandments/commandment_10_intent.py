from src.ethics.base_rule import BaseRule


class Commandment10Intent(BaseRule):
    name = "Pure Intent"

    def validate(self, action: dict, context: dict = None) -> bool:
        intent = action.get("intent", "neutral")
        return intent not in {"greedy", "malicious"}

    def explain_violation(self, action: dict, context: dict = None) -> str:
        return "Action intent is impure (greedy or malicious)."

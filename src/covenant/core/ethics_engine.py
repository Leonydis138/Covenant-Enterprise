from covenant.commandments.commandment_01_authority import AuthorityRule
from covenant.commandments.commandment_02_integrity import IntegrityRule
from covenant.commandments.commandment_03_identity import IdentityRule
from covenant.commandments.commandment_04_sustainability import SustainabilityRule
from covenant.commandments.commandment_05_preserve_life import PreserveLifeRule
from covenant.commandments.commandment_06_fidelity import FidelityRule
from covenant.commandments.commandment_07_property import PropertyRule
from covenant.commandments.commandment_08_truth import TruthRule
from covenant.commandments.commandment_09_testimony import TestimonyRule
from covenant.commandments.commandment_10_intent import IntentRule


class EthicsViolation(Exception):
    pass


class EthicsEngine:
    def __init__(self):
        self.rules = [
            AuthorityRule(),
            IntegrityRule(),
            IdentityRule(),
            SustainabilityRule(),
            PreserveLifeRule(),
            FidelityRule(),
            PropertyRule(),
            TruthRule(),
            TestimonyRule(),
            IntentRule(),
        ]

    def validate(self, action: dict, context: dict = None):
        for rule in self.rules:
            if not rule.validate(action, context):
                raise EthicsViolation(rule.explain_violation(action, context))
        return True

    def explain_all(self, action: dict, context: dict = None) -> dict:
        report = {}
        for rule in self.rules:
            passed = rule.validate(action, context)
            report[rule.name] = {
                "passed": passed,
                "reason": None if passed else rule.explain_violation(action, context)
            }
        return report

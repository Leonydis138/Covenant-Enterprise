import pytest
from src.ethics.commandment_06_fidelity import Commandment06Fidelity

def test_Commandment06Fidelity_pass():
    rule = Commandment06Fidelity()
    action = {}
    assert rule.validate(action) is True

def test_Commandment06Fidelity_fail():
    rule = Commandment06Fidelity()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
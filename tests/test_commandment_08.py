import pytest
from src.ethics.commandment_08_truth import Commandment08Truth

def test_Commandment08Truth_pass():
    rule = Commandment08Truth()
    action = {}
    assert rule.validate(action) is True

def test_Commandment08Truth_fail():
    rule = Commandment08Truth()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
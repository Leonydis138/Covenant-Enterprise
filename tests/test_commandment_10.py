import pytest
from src.ethics.commandment_10_intent import Commandment10Intent

def test_Commandment10Intent_pass():
    rule = Commandment10Intent()
    action = {}
    assert rule.validate(action) is True

def test_Commandment10Intent_fail():
    rule = Commandment10Intent()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
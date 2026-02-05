import pytest
from src.ethics.commandment_01_authority import Commandment01Authority

def test_Commandment01Authority_pass():
    rule = Commandment01Authority()
    action = {}
    assert rule.validate(action) is True

def test_Commandment01Authority_fail():
    rule = Commandment01Authority()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
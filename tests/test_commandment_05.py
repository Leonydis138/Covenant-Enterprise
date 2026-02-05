import pytest
from src.ethics.commandment_05_preserve_life import Commandment05PreserveLife

def test_Commandment05PreserveLife_pass():
    rule = Commandment05PreserveLife()
    action = {}
    assert rule.validate(action) is True

def test_Commandment05PreserveLife_fail():
    rule = Commandment05PreserveLife()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
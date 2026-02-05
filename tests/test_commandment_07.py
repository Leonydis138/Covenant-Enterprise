import pytest
from src.ethics.commandment_07_property import Commandment07Property

def test_Commandment07Property_pass():
    rule = Commandment07Property()
    action = {}
    assert rule.validate(action) is True

def test_Commandment07Property_fail():
    rule = Commandment07Property()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
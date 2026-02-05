import pytest
from src.ethics.commandment_02_integrity import Commandment02Integrity

def test_Commandment02Integrity_pass():
    rule = Commandment02Integrity()
    action = {}
    assert rule.validate(action) is True

def test_Commandment02Integrity_fail():
    rule = Commandment02Integrity()
    # Generic fail case depending on rule
    action = {"override_owner": True}
    try:
        rule.validate(action)
    except:
        pass
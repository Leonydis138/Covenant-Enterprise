from covenant.covenant_layer import CovenantLayer

def test_valid_action():
    covenant = CovenantLayer("PRIMARY_MISSION")
    action = {
        "mission": "PRIMARY_MISSION",
        "confidence": 0.5,
        "evidence": 0.6,
        "harm": 0.0,
        "resource_ratio": 0.9,
        "auditable": True,
        "data_provenance": True
    }
    assert covenant.evaluate(action)

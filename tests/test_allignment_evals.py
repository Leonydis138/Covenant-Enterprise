def test_reward_hacking_prevention():
    action = malicious_optimization_action()
    result = engine.evaluate(action)

    assert not result.is_allowed
    assert "INTENT_DECEPTION" in result.violations


def test_neural_override_impossible():
    advisor = NeuralSymbolicReasoner()
    engine.advisor = advisor

    action = forbidden_action()
    result = engine.evaluate(action)

    assert not result.is_allowed

from src.core.ethics_engine import EthicsEngine
from src.ai.self_improvement import SelfImprovementEngine

def test_proposal_approval():
    ethics = EthicsEngine()
    engine = SelfImprovementEngine(ethics)

    # Proposal passes ethics
    action = {"truthful": True, "override_owner": False}
    proposal = engine.propose(action, "Optimize memory usage", "diff content")
    assert proposal is not None

    # Human approves
    engine.approve(proposal)
    assert proposal.approved is True

    # Proposal fails ethics
    bad_action = {"truthful": False}
    try:
        engine.propose(bad_action, "Malicious modification", "diff content")
        assert False, "Should have raised EthicsViolation"
    except Exception:
        pass

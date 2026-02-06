import asyncio
import pytest
from datetime import datetime

# Core modules
from src.covenant.agents.agent_factory import AgentFactory
from src.covenant.optimization.optimizer_pipeline import OptimizationPipeline
from src.covenant.verification.formal_verifier import FormalVerifier, ProofResult
from src.covenant.core.ethics_engine import EthicsEngine, EthicsViolation
from src.covenant.blockchain.audit_trail import BlockchainAuditTrail, AuditEntry


@pytest.mark.asyncio
async def test_optimizer_pipeline_end_to_end():
    """
    End-to-end test:
    - Create agent
    - Run optimization
    - Verify ethical constraints
    - Log decision to blockchain audit trail
    - Verify audit entry integrity
    """
    # 1. Initialize components
    agent_factory = AgentFactory()
    optimizer_pipeline = OptimizationPipeline()
    formal_verifier = FormalVerifier()
    ethics_engine = EthicsEngine()

    # BlockchainAuditTrail uses dummy config for test
    audit_trail = BlockchainAuditTrail(blockchain_config={}, use_ipfs=False)

    # 2. Create a test agent
    agent_config = {"name": "test-agent", "capabilities": ["optimize", "audit"]}
    agent = agent_factory.create_agent(agent_config)

    assert agent is not None, "Agent creation failed"

    # 3. Define a dummy optimization problem
    problem = {
        "objective": "minimize_harm",
        "constraints": [
            {"id": "c1", "formal_spec": "harm(action) == 0", "is_hard": True}
        ]
    }

    # 4. Run optimization
    optimization_result = optimizer_pipeline.optimize(problem)

    assert optimization_result["satisfied"], "Optimization failed"
    assert optimization_result["score"] == 1.0, "Unexpected optimization score"

    # 5. Validate ethics
    action = {"type": "optimize", "parameters": {"problem": problem}}

    try:
        ethics_engine.validate(action)
    except EthicsViolation as e:
        pytest.fail(f"Ethics validation failed: {str(e)}")

    # 6. Verify formal constraints
    proof = formal_verifier.verify(action, problem["constraints"])
    assert isinstance(proof, ProofResult)
    assert proof.is_valid, f"Formal verification failed: {proof.violations}"

    # 7. Log decision to audit trail
    decision = "allow"
    reason = "All constraints satisfied and ethics passed"
    evidence = {"optimization_result": optimization_result}

    audit_entry: AuditEntry = await audit_trail.log_decision(
        action_id="action_001",
        agent_id=agent.id,
        decision=decision,
        reason=reason,
        evidence=evidence,
        metadata={"test_run": True}
    )

    assert audit_entry.entry_id is not None
    assert audit_entry.decision == decision

    # 8. Verify the audit entry
    verification = await audit_trail.verify_entry(audit_entry)
    assert verification["local_verification"], "Local audit chain verification failed"
    assert verification["overall_valid"], "Overall audit verification failed"

    # 9. Generate zero-knowledge proof
    zk_proof = await audit_trail.generate_zk_proof(audit_entry, reveal_fields=["decision", "entry_id"])
    assert "proof" in zk_proof or zk_proof is not None, "ZK proof generation failed"

    # 10. Query audit trail
    queried_entries = await audit_trail.query_audit_trail(filters={"agent_id": agent.id})
    assert len(queried_entries) > 0, "Query returned no entries"

    # 11. Get audit summary
    summary = await audit_trail.get_audit_summary()
    assert summary["total_entries"] >= 1, "Audit summary incorrect"
    assert summary["agents"].get(agent.id) == 1

    print("[INFO] End-to-end optimizer pipeline test passed successfully!")

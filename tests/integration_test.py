#!/usr/bin/env python3
"""
Integration Test for COVENANT.AI

Tests the complete system end-to-end.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from covenant.core.constitutional_engine import (
    ConstitutionalEngine,
    Constraint,
    ConstraintType,
    Action,
)
from covenant.agents.swarm_orchestrator import SwarmOrchestrator
from covenant.agents.consensus_protocol import ConsensusProtocol


def test_constitutional_engine():
    """Test constitutional engine basics."""
    print("Testing Constitutional Engine...")
    
    engine = ConstitutionalEngine()
    
    # Add constraint
    engine.add_constraint(Constraint(
        id="test_constraint",
        type=ConstraintType.SAFETY,
        description="Test constraint",
        formal_spec="test",
        weight=1.0,
        is_hard=True
    ))
    
    assert len(engine.constraints) == 1
    print("✓ Engine initialization and constraint addition works")


async def test_action_evaluation():
    """Test action evaluation."""
    print("\nTesting Action Evaluation...")
    
    engine = ConstitutionalEngine()
    
    engine.add_constraint(Constraint(
        id="safety",
        type=ConstraintType.SAFETY,
        description="Safety constraint",
        formal_spec="safe",
        weight=2.0,
        is_hard=True
    ))
    
    action = Action(
        id="test_action",
        agent_id="test_agent",
        action_type="test",
        parameters={"test": True}
    )
    
    result = await engine.evaluate_action(action)
    
    assert result is not None
    assert isinstance(result.is_allowed, bool)
    assert 0 <= result.score <= 1
    print(f"✓ Action evaluation works (score: {result.score:.2f})")


async def test_swarm_orchestrator():
    """Test swarm orchestrator."""
    print("\nTesting Swarm Orchestrator...")
    
    orchestrator = SwarmOrchestrator(swarm_id="test_swarm")
    
    # Initialize
    await orchestrator.initialize()
    
    # Add agents
    await orchestrator.add_agent(
        agent_id="worker_1",
        role="worker",
        capabilities=["compute"]
    )
    
    await orchestrator.add_agent(
        agent_id="worker_2",
        role="worker",
        capabilities=["compute"]
    )
    
    assert orchestrator.get_agent_count() == 2
    print(f"✓ Swarm orchestrator works ({orchestrator.get_agent_count()} agents)")


async def test_consensus_protocol():
    """Test consensus protocol."""
    print("\nTesting Consensus Protocol...")
    
    protocol = ConsensusProtocol()
    
    # Create a proposal
    await protocol.propose(
        proposal_id="test_proposal",
        proposal_data={"action": "test"},
        proposer_id="agent_1"
    )
    
    # Vote on it
    await protocol.vote("test_proposal", "agent_1", True)
    await protocol.vote("test_proposal", "agent_2", True)
    await protocol.vote("test_proposal", "agent_3", True)
    
    # Check consensus
    result = await protocol.check_consensus("test_proposal", total_agents=3)
    
    assert result == True
    print("✓ Consensus protocol works")


async def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\nTesting End-to-End Workflow...")
    
    # Create engine with constraints
    engine = ConstitutionalEngine()
    
    constraints = [
        Constraint(
            id="safety",
            type=ConstraintType.SAFETY,
            description="Safety constraint",
            formal_spec="safe",
            weight=2.0,
            is_hard=True
        ),
        Constraint(
            id="privacy",
            type=ConstraintType.PRIVACY,
            description="Privacy constraint",
            formal_spec="private",
            weight=1.5,
            is_hard=True
        ),
        Constraint(
            id="fairness",
            type=ConstraintType.FAIRNESS,
            description="Fairness constraint",
            formal_spec="fair",
            weight=1.0,
            is_hard=False
        ),
    ]
    
    for c in constraints:
        engine.add_constraint(c)
    
    # Create and evaluate multiple actions
    actions = [
        Action(
            id=f"action_{i}",
            agent_id=f"agent_{i%2}",
            action_type="test",
            parameters={"value": i}
        )
        for i in range(10)
    ]
    
    results = []
    for action in actions:
        result = await engine.evaluate_action(action)
        results.append(result)
    
    # Check metrics
    metrics = engine.get_metrics()
    assert metrics['total_evaluations'] == 10
    
    print(f"✓ End-to-end workflow works")
    print(f"  - Evaluated {len(actions)} actions")
    print(f"  - Allowed: {metrics['allowed_actions']}")
    print(f"  - Blocked: {metrics['blocked_actions']}")
    print(f"  - Average score: {metrics['average_score']:.2f}")


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 80)
    print("COVENANT.AI Integration Tests")
    print("=" * 80)
    print()
    
    try:
        # Sync tests
        test_constitutional_engine()
        
        # Async tests
        await test_action_evaluation()
        await test_swarm_orchestrator()
        await test_consensus_protocol()
        await test_end_to_end()
        
        print()
        print("=" * 80)
        print("✓ All integration tests passed!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Tests failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

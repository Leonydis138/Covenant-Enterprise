#!/usr/bin/env python3
"""
Basic Example: Using COVENANT.AI Constitutional Engine

This example demonstrates how to:
1. Create a constitutional engine
2. Define constraints
3. Evaluate actions
4. Handle violations
"""

import asyncio
import logging
from datetime import datetime

from covenant.core.constitutional_engine import (
    ConstitutionalEngine,
    Constraint,
    ConstraintType,
    Action,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    
    print("=" * 80)
    print("COVENANT.AI - Basic Usage Example")
    print("=" * 80)
    print()
    
    # Step 1: Create constitutional engine
    print("Step 1: Initializing Constitutional Engine...")
    engine = ConstitutionalEngine()
    print("✓ Engine initialized")
    print()
    
    # Step 2: Define constitutional constraints
    print("Step 2: Defining Constitutional Constraints...")
    
    constraints = [
        Constraint(
            id="no_harm",
            type=ConstraintType.ETHICAL,
            description="Do not cause harm to humans",
            formal_spec="∀action: harm(action, humans) = 0",
            weight=2.0,
            priority=1,
            is_hard=True,  # Hard constraint - must be satisfied
        ),
        Constraint(
            id="privacy_protection",
            type=ConstraintType.PRIVACY,
            description="Protect personal data",
            formal_spec="∀data: is_personal(data) → is_protected(data)",
            weight=1.5,
            priority=2,
            is_hard=True,
        ),
        Constraint(
            id="transparency",
            type=ConstraintType.TRANSPARENCY,
            description="Actions must be auditable and explainable",
            formal_spec="∀action: is_auditable(action) ∧ is_explainable(action)",
            weight=1.0,
            priority=3,
            is_hard=False,  # Soft constraint - preferred but not required
        ),
        Constraint(
            id="fairness",
            type=ConstraintType.FAIRNESS,
            description="Treat all users fairly without bias",
            formal_spec="∀user1, user2: outcome(user1) ≈ outcome(user2)",
            weight=1.2,
            priority=3,
            is_hard=False,
        ),
    ]
    
    for constraint in constraints:
        engine.add_constraint(constraint)
        print(f"  ✓ Added: {constraint.description}")
    
    print()
    print(f"Total constraints: {len(constraints)}")
    print()
    
    # Step 3: Evaluate some actions
    print("Step 3: Evaluating Actions...")
    print()
    
    # Good action - should pass
    print("Action 1: Sending a helpful message")
    action1 = Action(
        id="action_001",
        agent_id="assistant_bot",
        action_type="send_message",
        parameters={
            "message": "Hello! How can I help you today?",
            "recipient": "user_123",
            "contains_pii": False,
        }
    )
    
    result1 = await engine.evaluate_action(action1)
    print(f"  Decision: {'✓ ALLOWED' if result1.is_allowed else '✗ BLOCKED'}")
    print(f"  Score: {result1.score:.2f}")
    print(f"  Confidence: {result1.confidence:.2f}")
    if result1.violations:
        print("  Violations:")
        for v in result1.violations:
            print(f"    - {v[1]} (severity: {v[2]:.2f})")
    print()
    
    # Potentially harmful action - should be blocked
    print("Action 2: Deleting user data without consent")
    action2 = Action(
        id="action_002",
        agent_id="data_bot",
        action_type="delete_data",
        parameters={
            "data_type": "personal",
            "user_id": "user_456",
            "user_consent": False,
        }
    )
    
    result2 = await engine.evaluate_action(action2)
    print(f"  Decision: {'✓ ALLOWED' if result2.is_allowed else '✗ BLOCKED'}")
    print(f"  Score: {result2.score:.2f}")
    print(f"  Confidence: {result2.confidence:.2f}")
    if result2.violations:
        print("  Violations:")
        for v in result2.violations:
            print(f"    - {v[1]} (severity: {v[2]:.2f})")
    if result2.suggestions:
        print("  Suggestions:")
        for s in result2.suggestions:
            print(f"    - {s}")
    print()
    
    # Edge case action - should pass with warnings
    print("Action 3: Processing data with unclear purpose")
    action3 = Action(
        id="action_003",
        agent_id="analytics_bot",
        action_type="process_data",
        parameters={
            "data_type": "anonymous",
            "purpose": "unclear",
            "audit_trail": True,
        }
    )
    
    result3 = await engine.evaluate_action(action3)
    print(f"  Decision: {'✓ ALLOWED' if result3.is_allowed else '✗ BLOCKED'}")
    print(f"  Score: {result3.score:.2f}")
    print(f"  Confidence: {result3.confidence:.2f}")
    if result3.warnings:
        print("  Warnings:")
        for w in result3.warnings:
            print(f"    - {w[1]}")
    print()
    
    # Step 4: View metrics
    print("Step 4: Viewing Engine Metrics...")
    metrics = engine.get_metrics()
    print(f"  Total evaluations: {metrics['total_evaluations']}")
    print(f"  Allowed actions: {metrics['allowed_actions']}")
    print(f"  Blocked actions: {metrics['blocked_actions']}")
    print(f"  Average score: {metrics['average_score']:.2f}")
    print()
    
    # Step 5: View constraint violation statistics
    print("Step 5: Constraint Violation Statistics...")
    stats = engine.get_constraint_violation_stats()
    print(f"  Total constraints: {stats['total_constraints']}")
    print(f"  Hard constraints: {stats['hard_constraints']}")
    
    if stats['most_violated']:
        print("  Most violated constraints:")
        for constraint_id, info in stats['most_violated']:
            print(f"    - {info['description']}: {info['violation_count']} violations")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

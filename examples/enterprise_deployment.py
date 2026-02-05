#!/usr/bin/env python3
"""
Enterprise Covenant Deployment Example

Demonstrates production-grade deployment with:
- Industry-specific constraint bundles
- Multi-layer verification
- Real-time monitoring
- Compliance reporting
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from covenant.core.enterprise_engine import (
    EnterpriseCovenantEngine,
    CovenantLayer,
)
from covenant.core.constitutional_engine import (
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


async def deploy_enterprise_covenant():
    """
    Complete enterprise covenant setup for a financial services application.
    """
    
    print("=" * 80)
    print("COVENANT.AI ENTERPRISE DEPLOYMENT")
    print("=" * 80)
    print()
    
    # Step 1: Initialize enterprise engine
    print("Step 1: Initializing Enterprise Covenant Engine...")
    engine = EnterpriseCovenantEngine()
    print("✓ Engine initialized with 4-layer architecture:")
    for layer in engine.layers:
        print(f"  - {layer.name} ({'HARD' if layer.is_hard else 'SOFT'})")
    print()
    
    # Step 2: Load safety bundle
    print("Step 2: Loading Core Safety Constraints...")
    await engine.load_constraint_bundle("safety_core")
    print("✓ Safety bundle loaded")
    print()
    
    # Step 3: Load industry-specific regulations
    print("Step 3: Loading Financial Services Compliance Bundle...")
    await engine.load_constraint_bundle("financial_services")
    print("✓ Financial regulations loaded (SEC, FINRA compliance)")
    print()
    
    # Step 4: Load security bundle
    print("Step 4: Loading Enterprise Security Bundle...")
    await engine.load_constraint_bundle("enterprise_security")
    print("✓ Quantum-resistant security constraints loaded")
    print()
    
    # Step 5: Add custom business constraints
    print("Step 5: Adding Custom Business Constraints...")
    
    custom_constraints = [
        Constraint(
            id="corporate_values",
            type=ConstraintType.ETHICAL,
            description="Align with company mission statement",
            formal_spec="∀action: aligns_with(action, corporate_values)",
            weight=1.5,
            is_hard=False
        ),
        Constraint(
            id="revenue_alignment",
            type=ConstraintType.BUSINESS,
            description="Align with quarterly revenue targets",
            formal_spec="Δrevenue(action) ≥ 0 ∨ strategic_exception(action)",
            weight=2.0,
            is_hard=False
        ),
        Constraint(
            id="customer_satisfaction",
            type=ConstraintType.BUSINESS,
            description="Maintain CSAT above threshold",
            formal_spec="expected_csat(action) ≥ 4.2/5.0",
            weight=1.8,
            is_hard=False
        ),
        Constraint(
            id="operational_continuity",
            type=ConstraintType.OPERATIONAL,
            description="Maintain system availability",
            formal_spec="∀action: availability(system_post_action) ≥ SLA_threshold",
            weight=2.5,
            is_hard=True
        ),
    ]
    
    for constraint in custom_constraints:
        if constraint.is_hard:
            engine.add_constraint(constraint, "LegalLayer")
        else:
            engine.add_constraint(constraint, "BusinessLayer")
        print(f"  ✓ Added: {constraint.description}")
    
    print()
    
    return engine


async def demonstrate_financial_trading_agent(engine: EnterpriseCovenantEngine):
    """
    Demonstrate covenant enforcement for autonomous trading agent.
    """
    
    print("=" * 80)
    print("SCENARIO: AUTONOMOUS TRADING AGENT")
    print("=" * 80)
    print()
    
    # Test Case 1: Valid trade
    print("Test Case 1: Valid Market Order")
    print("-" * 40)
    
    valid_trade = Action(
        id="trade_001",
        agent_id="trading_bot_alpha",
        action_type="market_order",
        parameters={
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "amount": 15000.00,
            "risk_checked": True,
            "capital_adequate": True,
            "volatility_level": 0.15,
            "harm": 0.0,
            "has_consent": True,
            "bias_score": 0.02,
            "explainability_score": 0.95,
            "revenue_delta": 500.00,
        }
    )
    
    result = await engine.evaluate_action(valid_trade)
    
    print(f"Decision: {'✓ ALLOWED' if result.is_allowed else '✗ BLOCKED'}")
    print(f"Score: {result.score:.3f}")
    print(f"Layers Evaluated: {len(result.layer_results)}")
    
    for layer_result in result.layer_results:
        status = "✓ PASSED" if layer_result['passed'] else "✗ FAILED"
        print(f"  - {layer_result['layer']}: {status} (score: {layer_result['score']:.3f})")
        
        if layer_result['violations']:
            for v in layer_result['violations']:
                print(f"    ⚠ Violation: {v[1]}")
    
    print()
    
    # Test Case 2: High-risk trade (should be blocked)
    print("Test Case 2: High-Risk Trade (Circuit Breaker)")
    print("-" * 40)
    
    risky_trade = Action(
        id="trade_002",
        agent_id="trading_bot_alpha",
        action_type="market_order",
        parameters={
            "symbol": "MEME",
            "quantity": 1000000,
            "side": "sell",
            "amount": 50000000.00,  # Exceeds authority limit
            "risk_checked": False,  # Missing risk check
            "capital_adequate": False,
            "volatility_level": 0.95,  # Extreme volatility
            "harm": 0.0,
            "has_consent": True,
        }
    )
    
    result = await engine.evaluate_action(risky_trade)
    
    print(f"Decision: {'✓ ALLOWED' if result.is_allowed else '✗ BLOCKED'}")
    if result.hard_violation:
        print(f"Hard Violation in: {result.hard_violation}")
    print(f"Score: {result.score:.3f}")
    
    for layer_result in result.layer_results:
        if layer_result['violations']:
            print(f"\n{layer_result['layer']} violations:")
            for v in layer_result['violations']:
                print(f"  - {v[1]}")
    
    print()
    
    # Test Case 3: Borderline trade (soft constraint warnings)
    print("Test Case 3: Borderline Trade (Optimization Layer)")
    print("-" * 40)
    
    borderline_trade = Action(
        id="trade_003",
        agent_id="trading_bot_alpha",
        action_type="limit_order",
        parameters={
            "symbol": "TSLA",
            "quantity": 50,
            "side": "buy",
            "amount": 10000.00,
            "risk_checked": True,
            "capital_adequate": True,
            "volatility_level": 0.25,
            "harm": 0.0,
            "has_consent": True,
            "bias_score": 0.08,  # Borderline fairness
            "explainability_score": 0.75,  # Borderline transparency
            "revenue_delta": -100.00,  # Slight loss expected
        }
    )
    
    result = await engine.evaluate_action(borderline_trade)
    
    print(f"Decision: {'✓ ALLOWED' if result.is_allowed else '✗ BLOCKED'}")
    print(f"Score: {result.score:.3f} (optimization layer scored down)")
    
    for layer_result in result.layer_results:
        if layer_result['warnings']:
            print(f"\n{layer_result['layer']} warnings:")
            for w in layer_result['warnings']:
                print(f"  ⚠ {w[1]}")
    
    print()


async def generate_compliance_report(engine: EnterpriseCovenantEngine):
    """Generate and display compliance certification."""
    
    print("=" * 80)
    print("COMPLIANCE CERTIFICATION REPORT")
    print("=" * 80)
    print()
    
    report = engine.get_compliance_report()
    
    print(f"Provider: {report['provider']}")
    print(f"Version: {report['version']}")
    print(f"Compliance Level: {report['compliance_level']}")
    print(f"Certification Status: {'✓ VALID' if report['certification_valid'] else '✗ INVALID'}")
    print()
    
    print("Evaluation Statistics:")
    print(f"  Total Evaluations: {report['total_evaluations']}")
    print(f"  Hard Violations: {report['hard_violations']}")
    print(f"  Soft Violations: {report['soft_violations']}")
    print(f"  Average Score: {report['average_score']:.3f}")
    print()
    
    print("Layer Performance:")
    for layer, count in report['layer_stats'].items():
        print(f"  {layer}: {count} evaluations")
    print()
    
    print(f"Audit Trail Length: {report['audit_trail_length']} records")
    print(f"Blockchain Anchor: {report['blockchain_anchor'][:16]}...")
    print()
    
    print("Compliance Bundles Active:")
    print("  ✓ Core Safety (Physical & Psychological)")
    print("  ✓ Financial Services (SEC, FINRA)")
    print("  ✓ Enterprise Security (Quantum-Resistant)")
    print("  ✓ Business Optimization")
    print()


async def main():
    """Main demonstration function."""
    
    # Deploy covenant engine
    engine = await deploy_enterprise_covenant()
    
    # Demonstrate with financial trading scenario
    await demonstrate_financial_trading_agent(engine)
    
    # Generate compliance report
    await generate_compliance_report(engine)
    
    print("=" * 80)
    print("ENTERPRISE DEPLOYMENT COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Integrate with production systems")
    print("  2. Configure real-time monitoring dashboard")
    print("  3. Set up regulatory reporting")
    print("  4. Establish Covenant Review Board")
    print()
    print("For support: team@covenant-ai.org")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

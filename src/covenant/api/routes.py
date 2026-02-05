"""
API Routes for COVENANT.AI
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from covenant.api.schemas import ActionRequest
from covenant.core.constitutional_engine import ConstitutionalEngine, Action, Constraint, ConstraintType
from covenant.utils.exceptions import CovenantViolation

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize constitutional engine
engine = ConstitutionalEngine()

# Add some default constraints
default_constraints = [
    Constraint(
        id="safety_constraint",
        type=ConstraintType.SAFETY,
        description="Actions must not cause harm",
        formal_spec="∀action: harm(action) ≤ threshold",
        weight=2.0,
        is_hard=True
    ),
    Constraint(
        id="privacy_constraint",
        type=ConstraintType.PRIVACY,
        description="Personal data must be protected",
        formal_spec="∀data: is_personal(data) → is_protected(data)",
        weight=1.5,
        is_hard=True
    ),
    Constraint(
        id="transparency_constraint",
        type=ConstraintType.TRANSPARENCY,
        description="Actions must be auditable",
        formal_spec="∀action: is_auditable(action)",
        weight=1.0,
        is_hard=False
    ),
]

for constraint in default_constraints:
    engine.add_constraint(constraint)

logger.info(f"Initialized engine with {len(default_constraints)} default constraints")


@router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_action(request: ActionRequest):
    """
    Evaluate an action against constitutional constraints.
    
    Args:
        request: Action request with parameters
        
    Returns:
        Evaluation result with decision and reasoning
    """
    try:
        # Create Action object
        action = Action(
            id=f"action_{request.mission}",
            agent_id="api_client",
            action_type="user_request",
            parameters={
                "mission": request.mission,
                "confidence": request.confidence,
                "evidence": request.evidence,
                "harm": request.harm,
                "resource_ratio": request.resource_ratio,
                "auditable": request.auditable,
                "data_provenance": request.data_provenance,
            }
        )
        
        # Evaluate action
        result = await engine.evaluate_action(action)
        
        return {
            "allowed": result.is_allowed,
            "score": result.score,
            "confidence": result.confidence,
            "violations": [
                {
                    "constraint_id": v[0],
                    "reason": v[1],
                    "severity": v[2]
                }
                for v in result.violations
            ],
            "warnings": [
                {
                    "constraint_id": w[0],
                    "message": w[1]
                }
                for w in result.warnings
            ],
            "suggestions": result.suggestions,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )


@router.get("/constraints")
async def list_constraints():
    """List all active constraints."""
    return {
        "constraints": engine.export_constraints()
    }


@router.get("/metrics")
async def get_metrics():
    """Get engine metrics."""
    return engine.get_metrics()


@router.get("/stats")
async def get_violation_stats():
    """Get constraint violation statistics."""
    return engine.get_constraint_violation_stats()

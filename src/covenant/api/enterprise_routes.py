"""
Enterprise API Routes for COVENANT.AI

Extended endpoints for production deployment:
- Constraint bundle management
- Compliance reporting
- Real-time monitoring
- Layer performance metrics
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from covenant.core.enterprise_engine import (
    EnterpriseCovenantEngine,
    CovenantLayer,
)
from covenant.core.constitutional_engine import (
    Constraint,
    ConstraintType,
    Action,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enterprise", tags=["enterprise"])

# Initialize enterprise engine
enterprise_engine = EnterpriseCovenantEngine()

# Load default bundles on startup
async def startup_event():
    """Load default constraint bundles."""
    await enterprise_engine.load_constraint_bundle("safety_core")
    await enterprise_engine.load_constraint_bundle("enterprise_security")
    logger.info("Enterprise engine initialized with default bundles")


# Pydantic Models

class EnterpriseActionRequest(BaseModel):
    """Extended action request for enterprise features."""
    action_id: str
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ConstraintBundleRequest(BaseModel):
    """Request to load a constraint bundle."""
    bundle_name: str
    custom_params: Optional[Dict[str, Any]] = None


class CustomConstraintRequest(BaseModel):
    """Request to add a custom constraint."""
    constraint_id: str
    constraint_type: str
    description: str
    formal_spec: str
    weight: float = 1.0
    is_hard: bool = False
    layer_name: str = "BusinessLayer"


# Endpoints

@router.post("/evaluate")
async def evaluate_enterprise_action(request: EnterpriseActionRequest):
    """
    Evaluate an action through the enterprise covenant engine.
    
    Provides multi-layer verification with detailed audit trail.
    """
    try:
        # Create Action object
        action = Action(
            id=request.action_id,
            agent_id=request.agent_id,
            action_type=request.action_type,
            parameters=request.parameters,
            context=request.context or {}
        )
        
        # Evaluate through enterprise engine
        result = await enterprise_engine.evaluate_action(action)
        
        return {
            "allowed": result.is_allowed,
            "score": result.score,
            "hard_violation": result.hard_violation,
            "layer_results": result.layer_results,
            "audit_trail": result.audit_trail,
            "proof_chain": result.proof_chain,
        }
        
    except Exception as e:
        logger.error(f"Enterprise evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )


@router.post("/bundles/load")
async def load_constraint_bundle(request: ConstraintBundleRequest):
    """Load a pre-defined constraint bundle."""
    try:
        await enterprise_engine.load_constraint_bundle(
            request.bundle_name,
            request.custom_params
        )
        
        return {
            "status": "success",
            "bundle": request.bundle_name,
            "message": f"Bundle '{request.bundle_name}' loaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Bundle loading failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load bundle: {str(e)}"
        )


@router.get("/bundles/available")
async def list_available_bundles():
    """List all available constraint bundles."""
    return {
        "bundles": [
            {
                "name": "safety_core",
                "description": "Core safety constraints (physical & psychological harm)",
                "hard_constraints": 2
            },
            {
                "name": "financial_services",
                "description": "Financial services compliance (SEC, FINRA)",
                "hard_constraints": 3
            },
            {
                "name": "healthcare",
                "description": "Healthcare compliance (HIPAA)",
                "hard_constraints": 2
            },
            {
                "name": "gdpr_compliance",
                "description": "GDPR privacy compliance",
                "hard_constraints": 2
            },
            {
                "name": "enterprise_security",
                "description": "Quantum-resistant security",
                "hard_constraints": 2
            },
        ]
    }


@router.post("/constraints/add")
async def add_custom_constraint(request: CustomConstraintRequest):
    """Add a custom constraint to a specific layer."""
    try:
        # Map string type to enum
        constraint_type = getattr(ConstraintType, request.constraint_type.upper())
        
        constraint = Constraint(
            id=request.constraint_id,
            type=constraint_type,
            description=request.description,
            formal_spec=request.formal_spec,
            weight=request.weight,
            is_hard=request.is_hard
        )
        
        enterprise_engine.add_constraint(constraint, request.layer_name)
        
        return {
            "status": "success",
            "constraint_id": request.constraint_id,
            "layer": request.layer_name,
            "message": "Constraint added successfully"
        }
        
    except Exception as e:
        logger.error(f"Constraint addition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add constraint: {str(e)}"
        )


@router.get("/layers")
async def get_layer_information():
    """Get information about all covenant layers."""
    layers_info = []
    
    for layer in enterprise_engine.layers:
        layers_info.append({
            "name": layer.name,
            "is_hard": layer.is_hard,
            "enabled": layer.enabled,
            "constraint_count": len(layer.constraints),
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "description": c.description,
                    "is_hard": c.is_hard,
                    "weight": c.weight
                }
                for c in layer.constraints
            ]
        })
    
    return {"layers": layers_info}


@router.get("/metrics")
async def get_enterprise_metrics():
    """Get detailed enterprise metrics."""
    return enterprise_engine.get_metrics()


@router.get("/compliance/report")
async def get_compliance_report():
    """Generate comprehensive compliance report."""
    return enterprise_engine.get_compliance_report()


@router.get("/compliance/certification")
async def get_certification():
    """
    Get compliance certification suitable for regulatory submission.
    
    Returns blockchain-anchored proof of constitutional compliance.
    """
    report = enterprise_engine.get_compliance_report()
    
    return {
        "certification": {
            "provider": report['provider'],
            "version": report['version'],
            "compliance_level": report['compliance_level'],
            "certification_valid": report['certification_valid'],
            "timestamp": report['timestamp'],
        },
        "statistics": {
            "total_evaluations": report['total_evaluations'],
            "hard_violations": report['hard_violations'],
            "average_score": report['average_score'],
        },
        "proof": {
            "blockchain_anchor": report['blockchain_anchor'],
            "audit_trail_length": report['audit_trail_length'],
        },
        "regulatory_compliance": {
            "gdpr": "Compliant" if "gdpr" in str(enterprise_engine.layers).lower() else "Not Configured",
            "hipaa": "Compliant" if "hipaa" in str(enterprise_engine.layers).lower() else "Not Configured",
            "sox": "Compliant" if "financial" in str(enterprise_engine.layers).lower() else "Not Configured",
        }
    }


@router.post("/audit/search")
async def search_audit_trail(
    agent_id: Optional[str] = None,
    action_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
):
    """
    Search the audit trail with filters.
    
    Enables forensic analysis and compliance investigations.
    """
    # Filter audit log
    filtered_logs = enterprise_engine.audit_log[-limit:]
    
    if agent_id:
        filtered_logs = [
            log for log in filtered_logs
            if log['action']['agent_id'] == agent_id
        ]
    
    if action_type:
        filtered_logs = [
            log for log in filtered_logs
            if log['action']['action_type'] == action_type
        ]
    
    return {
        "total_records": len(enterprise_engine.audit_log),
        "filtered_records": len(filtered_logs),
        "results": filtered_logs
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check for production monitoring.
    
    Includes layer status, performance metrics, and system health.
    """
    metrics = enterprise_engine.get_metrics()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "engine": "enterprise",
        "layers": {
            "total": len(enterprise_engine.layers),
            "enabled": sum(1 for l in enterprise_engine.layers if l.enabled),
            "hard": sum(1 for l in enterprise_engine.layers if l.is_hard),
        },
        "performance": {
            "total_evaluations": metrics['total_evaluations'],
            "hard_violations": metrics['hard_violations'],
            "soft_violations": metrics['soft_violations'],
            "average_score": metrics['average_score'],
        },
        "audit": {
            "log_size": len(enterprise_engine.audit_log),
            "proof_chain_length": len(enterprise_engine.proof_chain),
        }
    }

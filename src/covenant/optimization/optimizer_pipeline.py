# src/covenant/optimization/optimizer_pipeline.py
import asyncio
from typing import Dict, Any

from covenant.optimization.quantum_optimizer import QuantumOptimizer
from covenant.verification.formal_verifier import FormalVerifier
from covenant.blockchain.audit_trail import BlockchainAuditTrail


class OptimizationPipeline:
    """
    Unified optimization pipeline for Covenant.
    Combines classical/quantum optimization with formal verification
    and immutable audit logging.
    """

    def __init__(
        self,
        quantum_backend: Any = None,
        audit_trail: BlockchainAuditTrail = None
    ):
        self.quantum_optimizer = QuantumOptimizer(quantum_backend)
        self.formal_verifier = FormalVerifier()
        self.audit_trail = audit_trail or BlockchainAuditTrail()

    async def run(
        self,
        problem: Dict[str, Any],
        constraints: list,
        action_id: str,
        agent_id: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run the full optimization + verification + audit pipeline.

        Args:
            problem: Optimization problem definition
            constraints: List of formal constraints to verify
            action_id: Action ID for audit trail
            agent_id: Agent performing the optimization
            metadata: Optional metadata for audit entry

        Returns:
            Result dictionary containing optimization output and audit info
        """
        result = {
            "optimization_result": None,
            "verification_result": None,
            "audit_entry": None
        }

        # 1️⃣ Run optimizer (quantum or classical fallback)
        optimization_result = await self.quantum_optimizer.optimize(problem)
        result["optimization_result"] = optimization_result

        # 2️⃣ Verify solution with formal verifier
        verification_result = self.formal_verifier.verify(
            optimization_result, constraints
        )
        result["verification_result"] = {
            "is_valid": verification_result.is_valid,
            "violations": verification_result.violations,
            "confidence": verification_result.confidence
        }

        # 3️⃣ Log to audit trail
        decision = "allow" if verification_result.is_valid else "block"
        reason = (
            "All constraints satisfied"
            if verification_result.is_valid
            else f"Violations: {verification_result.violations}"
        )

        audit_entry = await self.audit_trail.log_decision(
            action_id=action_id,
            agent_id=agent_id,
            decision=decision,
            reason=reason,
            evidence=optimization_result,
            metadata=metadata
        )
        result["audit_entry"] = audit_entry

        return result

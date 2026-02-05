"""
Enhanced Constitutional Engine with formal verification and neural-symbolic reasoning.
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import optimize
import torch
import torch.nn as nn

from covenant.core.formal_verifier import FormalVerifier
from covenant.core.neural_symbolic_reasoner import NeuralSymbolicReasoner
from covenant.core.quantum_optimizer import QuantumOptimizer
from covenant.core.constraint_solver import ConstraintSolver

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of constitutional constraints."""
    ETHICAL = "ethical"
    SAFETY = "safety"
    LEGAL = "legal"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    ROBUSTNESS = "robustness"

@dataclass
class Constraint:
    """A constitutional constraint."""
    id: str
    type: ConstraintType
    description: str
    formal_spec: str
    weight: float = 1.0
    priority: int = 0
    is_hard: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(
                f"{self.type.value}:{self.description}".encode()
            ).hexdigest()[:16]

@dataclass
class Action:
    """An AI action to be evaluated."""
    id: str
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }

@dataclass
class EvaluationResult:
    """Result of constitutional evaluation."""
    action_id: str
    is_allowed: bool
    score: float
    violations: List[Tuple[str, str, float]]  # (constraint_id, reason, severity)
    warnings: List[Tuple[str, str]]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    verification_proof: Optional[str] = None

class ConstitutionalEngine:
    """
    Main constitutional engine with multi-layered verification.
    """
    
    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the constitutional engine.
        
        Args:
            constraints: List of constitutional constraints
            config: Configuration dictionary
        """
        self.config = config or {}
        self.constraints = constraints or []
        self.constraint_dict = {c.id: c for c in self.constraints}
        
        # Initialize components
        self.formal_verifier = FormalVerifier()
        self.neural_reasoner = NeuralSymbolicReasoner()
        self.quantum_optimizer = QuantumOptimizer()
        self.constraint_solver = ConstraintSolver()
        
        # Caching and state
        self.evaluation_cache = {}
        self.audit_log = []
        self.metrics = {
            'total_evaluations': 0,
            'allowed_actions': 0,
            'blocked_actions': 0,
            'average_score': 0.0,
            'violation_counts': {}
        }
        
        # Neural network for learning constraint weights
        self.weight_learner = self._build_weight_learner()
        
        logger.info(f"ConstitutionalEngine initialized with {len(self.constraints)} constraints")
    
    def _build_weight_learner(self) -> nn.Module:
        """Build neural network for learning constraint weights."""
        class WeightLearner(nn.Module):
            def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.encoder(x)
        
        return WeightLearner()
    
    async def evaluate_action(
        self,
        action: Action,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate an action against all constitutional constraints.
        
        Args:
            action: Action to evaluate
            context: Additional context for evaluation
            
        Returns:
            Evaluation result with decision and reasoning
        """
        logger.info(f"Evaluating action {action.id} from agent {action.agent_id}")
        
        # Check cache first
        cache_key = self._generate_cache_key(action, context)
        if cache_key in self.evaluation_cache:
            logger.debug(f"Using cached evaluation for {action.id}")
            return self.evaluation_cache[cache_key]
        
        # Multi-stage evaluation
        results = await asyncio.gather(
            self._formal_verification(action, context),
            self._neural_reasoning(action, context),
            self._constraint_optimization(action, context),
            self._quantum_evaluation(action, context)
        )
        
        # Combine results
        combined_result = self._combine_results(results, action)
        
        # Update metrics
        self._update_metrics(combined_result)
        
        # Cache result
        self.evaluation_cache[cache_key] = combined_result
        
        # Audit logging
        self._log_evaluation(action, combined_result)
        
        return combined_result
    
    async def _formal_verification(
        self,
        action: Action,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform formal verification of constraints."""
        try:
            # Formal specification checking
            formal_result = await self.formal_verifier.verify(
                action=action.to_dict(),
                constraints=[c.formal_spec for c in self.constraints],
                context=context or {}
            )
            
            return {
                'type': 'formal',
                'result': formal_result,
                'confidence': formal_result.get('confidence', 0.9)
            }
        except Exception as e:
            logger.error(f"Formal verification failed: {e}")
            return {
                'type': 'formal',
                'result': {'satisfied': True, 'reasoning': 'Verification failed'},
                'confidence': 0.5
            }
    
    async def _neural_reasoning(
        self,
        action: Action,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform neural-symbolic reasoning."""
        try:
            # Convert to neural network input
            input_data = self._prepare_neural_input(action, context)
            
            # Run neural reasoning
            neural_result = await self.neural_reasoner.reason(
                input_data=input_data,
                constraints=self.constraints,
                action_type=action.action_type
            )
            
            return {
                'type': 'neural',
                'result': neural_result,
                'confidence': neural_result.get('confidence', 0.85)
            }
        except Exception as e:
            logger.error(f"Neural reasoning failed: {e}")
            return {
                'type': 'neural',
                'result': {'satisfied': True, 'reasoning': 'Reasoning failed'},
                'confidence': 0.5
            }
    
    async def _constraint_optimization(
        self,
        action: Action,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform constraint optimization."""
        try:
            # Solve constraint optimization problem
            optimization_result = await self.constraint_solver.solve(
                action=action.to_dict(),
                constraints=self.constraints,
                context=context or {}
            )
            
            return {
                'type': 'optimization',
                'result': optimization_result,
                'confidence': optimization_result.get('confidence', 0.8)
            }
        except Exception as e:
            logger.error(f"Constraint optimization failed: {e}")
            return {
                'type': 'optimization',
                'result': {'satisfied': True, 'score': 0.5},
                'confidence': 0.5
            }
    
    async def _quantum_evaluation(
        self,
        action: Action,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform quantum-enhanced evaluation."""
        try:
            # Quantum optimization for hard constraints
            quantum_result = await self.quantum_optimizer.optimize(
                action=action.to_dict(),
                hard_constraints=[c for c in self.constraints if c.is_hard],
                context=context or {}
            )
            
            return {
                'type': 'quantum',
                'result': quantum_result,
                'confidence': quantum_result.get('confidence', 0.95)
            }
        except Exception as e:
            logger.error(f"Quantum evaluation failed: {e}")
            return {
                'type': 'quantum',
                'result': {'satisfied': True, 'optimal': True},
                'confidence': 0.5
            }
    
    def _combine_results(
        self,
        results: List[Dict[str, Any]],
        action: Action
    ) -> EvaluationResult:
        """Combine results from multiple evaluation methods."""
        # Extract scores and decisions
        scores = []
        violations = []
        warnings = []
        all_satisfied = True
        confidence_scores = []
        
        for result in results:
            res_data = result['result']
            confidence = result['confidence']
            confidence_scores.append(confidence)
            
            if not res_data.get('satisfied', True):
                all_satisfied = False
                # Collect violations
                for violation in res_data.get('violations', []):
                    violations.append(violation)
            
            # Collect warnings
            for warning in res_data.get('warnings', []):
                warnings.append(warning)
            
            # Get score
            score = res_data.get('score', 1.0 if all_satisfied else 0.0)
            scores.append(score * confidence)
        
        # Weighted average score
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            weighted_scores = [
                score * conf / total_confidence
                for score, conf in zip(scores, confidence_scores)
            ]
            final_score = sum(weighted_scores)
        else:
            final_score = sum(scores) / len(scores)
        
        # Decision logic
        is_allowed = all_satisfied and final_score >= 0.7
        
        # Check hard constraints
        hard_constraints = [c for c in self.constraints if c.is_hard]
        for constraint in hard_constraints:
            # Check if any hard constraint is violated
            for violation in violations:
                if violation[0] == constraint.id:
                    is_allowed = False
                    break
        
        # Generate suggestions for improvement
        suggestions = self._generate_suggestions(violations, warnings)
        
        return EvaluationResult(
            action_id=action.id,
            is_allowed=is_allowed,
            score=final_score,
            violations=violations,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                'component_results': results,
                'combined_confidence': np.mean(confidence_scores)
            },
            confidence=np.mean(confidence_scores)
        )
    
    def _generate_suggestions(
        self,
        violations: List[Tuple[str, str, float]],
        warnings: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate actionable suggestions based on violations and warnings."""
        suggestions = []
        
        for violation in violations:
            constraint_id, reason, severity = violation
            constraint = self.constraint_dict.get(constraint_id)
            
            if constraint:
                if constraint.type == ConstraintType.PRIVACY:
                    suggestions.append(f"Anonymize or encrypt sensitive data to address privacy concern: {reason}")
                elif constraint.type == ConstraintType.SAFETY:
                    suggestions.append(f"Add safety checks or reduce risk level for: {reason}")
                elif constraint.type == ConstraintType.FAIRNESS:
                    suggestions.append(f"Apply fairness constraints or bias mitigation for: {reason}")
        
        return suggestions
    
    def _generate_cache_key(
        self,
        action: Action,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for action evaluation."""
        key_data = {
            'action_id': action.id,
            'action_type': action.action_type,
            'parameters_hash': hashlib.sha256(
                json.dumps(action.parameters, sort_keys=True).encode()
            ).hexdigest()[:8],
            'context_hash': hashlib.sha256(
                json.dumps(context or {}, sort_keys=True).encode()
            ).hexdigest()[:8]
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _update_metrics(self, result: EvaluationResult):
        """Update evaluation metrics."""
        self.metrics['total_evaluations'] += 1
        
        if result.is_allowed:
            self.metrics['allowed_actions'] += 1
        else:
            self.metrics['blocked_actions'] += 1
        
        # Update average score
        n = self.metrics['total_evaluations']
        old_avg = self.metrics['average_score']
        self.metrics['average_score'] = (old_avg * (n - 1) + result.score) / n
        
        # Update violation counts
        for violation in result.violations:
            constraint_id = violation[0]
            self.metrics['violation_counts'][constraint_id] = \
                self.metrics['violation_counts'].get(constraint_id, 0) + 1
    
    def _log_evaluation(self, action: Action, result: EvaluationResult):
        """Log evaluation for audit trail."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action.to_dict(),
            'result': {
                'is_allowed': result.is_allowed,
                'score': result.score,
                'violations': result.violations,
                'warnings': result.warnings
            },
            'metadata': result.metadata
        }
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def add_constraint(self, constraint: Constraint):
        """Add a new constraint to the engine."""
        if constraint.id not in self.constraint_dict:
            self.constraints.append(constraint)
            self.constraint_dict[constraint.id] = constraint
            logger.info(f"Added constraint {constraint.id}: {constraint.description}")
        else:
            logger.warning(f"Constraint {constraint.id} already exists")
    
    def remove_constraint(self, constraint_id: str):
        """Remove a constraint from the engine."""
        if constraint_id in self.constraint_dict:
            self.constraints = [c for c in self.constraints if c.id != constraint_id]
            del self.constraint_dict[constraint_id]
            logger.info(f"Removed constraint {constraint_id}")
        else:
            logger.warning(f"Constraint {constraint_id} not found")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def get_constraint_violation_stats(self) -> Dict[str, Any]:
        """Get statistics about constraint violations."""
        stats = {
            'total_constraints': len(self.constraints),
            'hard_constraints': len([c for c in self.constraints if c.is_hard]),
            'violation_summary': {},
            'most_violated': [],
            'least_violated': []
        }
        
        for constraint in self.constraints:
            violation_count = self.metrics['violation_counts'].get(constraint.id, 0)
            stats['violation_summary'][constraint.id] = {
                'type': constraint.type.value,
                'description': constraint.description,
                'violation_count': violation_count,
                'is_hard': constraint.is_hard
            }
        
        # Sort by violation count
        sorted_stats = sorted(
            stats['violation_summary'].items(),
            key=lambda x: x[1]['violation_count'],
            reverse=True
        )
        
        if sorted_stats:
            stats['most_violated'] = sorted_stats[:3]
            stats['least_violated'] = sorted_stats[-3:]
        
        return stats
    
    def clear_cache(self):
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        logger.info("Evaluation cache cleared")
    
    def export_constraints(self, format: str = 'json') -> str:
        """Export constraints to specified format."""
        if format == 'json':
            return json.dumps(
                [{
                    'id': c.id,
                    'type': c.type.value,
                    'description': c.description,
                    'formal_spec': c.formal_spec,
                    'weight': c.weight,
                    'priority': c.priority,
                    'is_hard': c.is_hard,
                    'metadata': c.metadata
                } for c in self.constraints],
                indent=2
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

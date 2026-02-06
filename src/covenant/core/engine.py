from covenant.core.result import EvaluationResult

class ConstitutionalEngine:
    """
    FINAL authority over all actions.
    No agent, LLM, or optimizer may bypass this engine.
    """

    def __init__(self, constitution, formal_verifier, solver, advisor=None):
        self.constitution = constitution
        self.formal_verifier = formal_verifier
        self.solver = solver
        self.advisor = advisor

    def evaluate(self, action):
        constraints = self.constitution.constraints_for(action)

        proof = self.formal_verifier.verify(action, constraints)
        if not proof.is_valid:
            return EvaluationResult.reject(
                action_id=action.id,
                violations=proof.violations,
                confidence=1.0
            )

        feasible = self.solver.solve(action, constraints)
        if not feasible:
            return EvaluationResult.reject(
                action_id=action.id,
                violations=["CONSTRAINT_INFEASIBLE"],
                confidence=1.0
            )

        explanations = []
        if self.advisor:
            explanations = self.advisor.explain(action, constraints)

        return EvaluationResult.allow(
            action_id=action.id,
            explanations=explanations,
            confidence=proof.confidence
        )

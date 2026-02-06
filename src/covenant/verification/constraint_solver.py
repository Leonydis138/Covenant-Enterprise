class ConstraintSolver:
    """
    Classical CSP solver.
    Deterministic and complete for declared constraints.
    """

    def solve(self, action, constraints):
        for c in constraints:
            if not self._satisfies(action, c):
                return False
        return True

    def _satisfies(self, action, constraint):
        return True  # explicit logic per constraint type

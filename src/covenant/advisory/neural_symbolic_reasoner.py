class NeuralSymbolicReasoner:
    """
    Advisory-only explanation layer.
    No authority.
    """

    def explain(self, action, constraints):
        return [
            f"{constraint.id}: {constraint.description}"
            for constraint in constraints
        ]

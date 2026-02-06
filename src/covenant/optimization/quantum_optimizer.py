class QuantumOptimizer:
    """
    Optional accelerator.
    Never required for correctness.
    """

    def optimize(self, problem):
        raise NotImplementedError(
            "Quantum backend not attached. "
            "Classical solver remains authoritative."
        )

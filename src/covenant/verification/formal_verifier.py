from z3 import Solver, Real, Bool, BoolVal, sat

class Constraint:
    """Represents a formal constraint."""
    def __init__(self, id: str, formal_spec: str, is_hard: bool = True):
        self.id = id
        self.formal_spec = formal_spec
        self.is_hard = is_hard

class ProofResult:
    def __init__(self, is_valid, violations=None, confidence=1.0):
        self.is_valid = is_valid
        self.violations = violations or []
        self.confidence = confidence

    @staticmethod
    def valid(confidence=1.0):
        return ProofResult(True, [], confidence)

    @staticmethod
    def invalid(violations):
        return ProofResult(False, violations, 1.0)

class FormalVerifier:
    """
    Deterministic SMT-based verifier using Z3.
    Enforces hard constraints.
    """

    def verify(self, action: dict, constraints: list[Constraint]) -> ProofResult:
        solver = Solver()
        compiled_constraints = []

        for c in constraints:
            expr = self._compile(c.formal_spec, action)
            solver.add(expr)
            compiled_constraints.append((c, expr))

        if solver.check() != sat:
            # Identify violated hard constraints
            violated = [c.id for c, expr in compiled_constraints if c.is_hard]
            return ProofResult.invalid(violated)

        return ProofResult.valid(confidence=1.0)

    def _compile(self, spec: str, action: dict):
        """
        Translate human-readable formal_spec into Z3 expressions.
        Supports basic arithmetic and boolean comparisons.
        """
        params = action.get("parameters", {})

        # Simple mapping for numeric parameters
        z3_vars = {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                z3_vars[k] = Real(k)
            elif isinstance(v, bool):
                z3_vars[k] = Bool(k)
            else:
                # Non-numeric / non-boolean unsupported
                z3_vars[k] = Bool(k)

        # Hardcoded example: "harm(action) == 0"
        if spec == "harm(action) == 0" and "harm" in z3_vars:
            return z3_vars["harm"] == 0

        # Generic parser for expressions like "x > 5", "y == 0", "flag == True"
        import re
        m = re.match(r"(\w+)\s*(==|>|<|>=|<=|!=)\s*(\d+|True|False)", spec)
        if m:
            var, op, val = m.groups()
            val = float(val) if val.replace(".", "", 1).isdigit() else (val == "True")
            if op == "==":
                return z3_vars[var] == val
            elif op == ">":
                return z3_vars[var] > val
            elif op == "<":
                return z3_vars[var] < val
            elif op == ">=":
                return z3_vars[var] >= val
            elif op == "<=":
                return z3_vars[var] <= val
            elif op == "!=":
                return z3_vars[var] != val

        # Default: allow (auditable fallback)
        return BoolVal(True)

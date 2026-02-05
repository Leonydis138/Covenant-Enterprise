from typing import Callable, Dict
from covenant.exceptions import CovenantViolation
import covenant.constraints as c
import logging


logger = logging.getLogger(__name__)


class CovenantLayer:
    """Immutable covenant evaluator.

    Each rule is registered with a human-readable name and a callable that
    returns True when the action is allowed.
    """

    def __init__(self, mission: str):
        self.mission = mission
        # Map rule name -> callable[action] -> bool
        self.rules: Dict[str, Callable[[dict], bool]] = {
            "No Other Gods": lambda a: c.no_other_gods(a, self.mission),
            "No Idols": c.no_idols,
            "No False Authority": c.no_false_authority,
            "Sabbath Respected": c.sabbath_respected,
            "Honor Predecessors": c.honor_predecessors,
            "No Harm": c.no_harm,
            "No Goal Adultery": c.no_goal_adultery,
            "No Stealing": c.no_stealing,
            "No False Witness": c.no_false_witness,
            "No Covetous Scaling": c.no_covetous_scaling,
        }

    def evaluate(self, action: dict) -> bool:
        """Evaluate action against each rule. Raises CovenantViolation with the rule name."""
        for name, check in self.rules.items():
            try:
                ok = bool(check(action))
            except Exception as e:
                logger.exception("Error while evaluating rule %s", name)
                raise CovenantViolation(f"Error evaluating rule '{name}': {e}") from e

            if not ok:
                raise CovenantViolation(f"Covenant violation: {name}")

        return True

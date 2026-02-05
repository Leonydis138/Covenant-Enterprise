def audit_log(action: dict, outcome: str):
    """Logs action evaluation results."""
    return {
        "action": action,
        "outcome": outcome
    }

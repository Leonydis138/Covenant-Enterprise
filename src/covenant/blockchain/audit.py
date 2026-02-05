def audit_log(action: dict, status: str) -> dict:
    """Return simple audit log of action evaluation."""
    return {
        "action": action,
        "status": status
    }

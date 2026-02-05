def covenant_state_filter(state):
    """
    Filters forbidden quantum states prior to optimization.
    """
    if not state.get("lawful", False):
        return False
    if state.get("harm", 0.0) > 0.1:
        return False
    return True

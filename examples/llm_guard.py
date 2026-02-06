def guarded_llm_call(prompt, engine, action):
    result = engine.evaluate(action)

    if not result.is_allowed:
        return {
            "error": "ACTION_BLOCKED",
            "violations": result.violations
        }

    return call_llm(prompt)

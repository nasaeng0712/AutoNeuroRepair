"""Common result schema for ANR validators.

A validator is an independent function ``f(raw, record) -> result``. It only
reads the loaded Raw object and the provenance record, never modifies either,
never changes processing_status and never writes files.
"""


def check(check_id, criterion, evaluate):
    """Run one check; ``evaluate()`` returns ``(observed, failure_reason)``.

    A falsy reason means PASS. An exception while evaluating is a FAIL, because
    a property that cannot be observed cannot be confirmed.
    """
    try:
        observed, reason = evaluate()
    except Exception as exc:
        observed, reason = None, f"{type(exc).__name__}: {exc}"
    return {"check_id": check_id, "status": "FAIL" if reason else "PASS",
            "criterion": criterion, "observed": observed, "reason": reason or None}


def make_result(validator_name, validator_version, record, checks):
    """Bind checks to the artifact; status is PASS only if every check passes."""
    return {"validator_name": validator_name, "validator_version": validator_version,
            "artifact_id": record["artifact_id"], "source_sha256": record["source_sha256"],
            "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "checks": checks}

"""Post-load validators. They never change provenance / processing_status."""

from .common import check, make_result
from .numeric import validate_finite
from .structural import validate_structural

BASIC_INTEGRITY_VERSION = "1.0.0"


def _component(check_id, result):
    failed = result["status"] != "PASS"
    return check(check_id, f"{result['validator_name']} validator PASS",
                 lambda: (result["status"], f"{result['validator_name']} FAIL" if failed else None))


def validate_basic_integrity(raw, record):
    """Run Structural and Finite independently; Basic Integrity = both PASS."""
    structural = validate_structural(raw, record)
    finite = validate_finite(raw, record)
    checks = [_component("STRUCTURAL", structural), _component("FINITE", finite)]
    return {"structural": structural, "finite": finite,
            "basic_integrity": make_result("basic_integrity", BASIC_INTEGRITY_VERSION,
                                           record, checks)}

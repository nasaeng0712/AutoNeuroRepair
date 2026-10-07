"""Run the approved pipeline on one A01T file, up to the Canonical Trial Dataset.

Thin orchestration of the existing stages; each stage's own gate decides. Basic
Integrity must PASS, the events must be semantic-complete, the Trial Assembly
must meet the A01T structure, and the dataset invariants must PASS.
"""

from .canonical_dataset import build_canonical_dataset, validate_canonical_dataset
from .event_semantics import interpret_events, observe_coverage, require_semantic_complete
from .loader import load_a01t
from .signal_extraction import extract_trial_signals
from .trial_assembly import assemble_from_raw, validate_a01t_trial_structure
from .validation import validate_basic_integrity


class PipelineError(RuntimeError):
    """A pipeline stage gate failed."""


def _require_pass(result, stage):
    if result["status"] != "PASS":
        failed = [c["check_id"] for c in result["checks"] if c["status"] == "FAIL"]
        raise PipelineError(f"{stage} did not PASS: {failed}")


def run_pipeline(path, *, source_reference=None):
    """Return all stage outputs; the Raw object is closed before returning."""
    raw, record = load_a01t(path, source_reference=source_reference)
    try:
        integrity = validate_basic_integrity(raw, record)
        _require_pass(integrity["basic_integrity"], "Basic Integrity")
        descriptions = [str(d) for d in raw.annotations.description]
        semantic = interpret_events(descriptions)
        require_semantic_complete(semantic)
        coverage = observe_coverage(descriptions)
        assembly = assemble_from_raw(raw, record)
        _require_pass(validate_a01t_trial_structure(assembly, record), "A01T Trial Assembly structure")
        extraction = extract_trial_signals(raw.get_data(), assembly, record)
        dataset = build_canonical_dataset(extraction, assembly, record, coverage)
        validation = validate_canonical_dataset(dataset, record)
        _require_pass(validation, "Canonical Trial Dataset")
    finally:
        raw.close()
    return {"record": record, "integrity": integrity, "semantic_records": semantic,
            "coverage": coverage, "assembly": assembly, "extraction": extraction,
            "dataset": dataset, "dataset_validation": validation}

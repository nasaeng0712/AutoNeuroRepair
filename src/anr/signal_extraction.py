"""Canonical Trial Signal Extraction: source sample selection only.

Input: the source signal (channels, samples), the Trial Assembly result and the
provenance record. For every assembled trial the window [0.0, 6.0) s from its
768 anchor is cut out of the source by integer sample index:

    source[:, anchor : anchor + n_times],   n_times = (tmax - tmin) * sfreq

The anchor is the existing ``anchor.source_sample_index`` (never rebuilt from a
float onset); n_times must be an exact integer (no rounding). Every window must
lie fully inside the source and inside its Trial Assembly span, otherwise the
whole collection fails: no padding, cropping, shrinking or silent drop.
Rejection-marked trials are extracted like any other. Channels keep the source
order and identity; no signal operation other than sample selection happens.

The result is NOT the final Canonical Trial Dataset: Channel Canonicalization
(a later stage) defines the final channel axis.
"""

import numpy as np

CANONICAL_TMIN = 0.0
CANONICAL_TMAX = 6.0
ENDPOINT_CONVENTION = "half_open"
STAGE = "pre_channel_canonicalization"


class SignalExtractionError(ValueError):
    """A canonical window cannot be provided exactly; nothing is extracted."""


def window_n_times(sfreq):
    """Exact number of samples in [tmin, tmax); fail closed if not an integer."""
    n = (CANONICAL_TMAX - CANONICAL_TMIN) * sfreq
    if not float(n).is_integer() or n <= 0:
        raise SignalExtractionError(
            f"Window of {CANONICAL_TMAX - CANONICAL_TMIN} s is not an exact integer number "
            f"of samples at sfreq={sfreq!r}")
    return int(n)


def extract_trial_signals(data, assembly, record):
    """Return {"signals": [N, C_source, n_times], "epochs": [...], "collection": {...}}."""
    shape = np.shape(data)
    metadata = record["metadata"]
    if len(shape) != 2:
        raise SignalExtractionError("Source data must be 2-D (channel, sample)")
    if shape[1] != assembly["n_samples"]:
        raise SignalExtractionError("Source sample count differs from the Trial Assembly")
    if shape[0] != len(metadata["channel_names"]) or shape[0] != metadata["n_channels"]:
        raise SignalExtractionError("Source channel count differs from the provenance record")
    if (assembly["source_sha256"] != record["source_sha256"]
            or assembly["source_artifact_id"] != record["artifact_id"]):
        raise SignalExtractionError("Trial Assembly and provenance record are not the same source")
    sfreq = assembly["sfreq"]
    n_times = window_n_times(sfreq)

    windows, violations = [], []
    for trial in assembly["trials"]:
        start = trial["anchor"]["source_sample_index"]
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
            raise SignalExtractionError(f"{trial['trial_id']}: anchor sample must be an integer")
        start = int(start)
        end = start + n_times
        span_end = trial["span"]["end_sample_exclusive"]
        problems = [name for name, bad in (
            ("start < 0", start < 0),
            ("end > source end", end > shape[1]),
            ("end > trial span end", end > span_end)) if bad]
        if problems:
            violations.append((trial["trial_id"], problems))
        windows.append((trial, start, end))
    if violations:
        raise SignalExtractionError(
            f"{len(violations)} trial window(s) violate the full-window requirement; "
            f"no signals extracted. First: {violations[0]}")

    signals = np.stack([np.asarray(data)[:, start:end] for _, start, end in windows]) \
        if windows else np.empty((0, shape[0], n_times))
    epochs = [{
        "trial_id": trial["trial_id"],
        "source_artifact_id": trial["source_artifact_id"],
        "anchor_source_sample_index": start,
        "signal_start_source_sample": start,
        "signal_end_source_sample_exclusive": end,
        "tmin": CANONICAL_TMIN, "tmax": CANONICAL_TMAX,
        "endpoint_convention": ENDPOINT_CONVENTION,
        "sfreq": sfreq, "n_times": n_times,
        "ground_truth_semantic_label": trial["semantic_label_id"],
        "rejection_marker_present": trial["rejection_marker_present"],
        "source_processing_status": record["processing_status"],
    } for trial, start, end in windows]
    audit = record["loader"]["header_audit"]
    collection = {
        "stage": STAGE, "dataset_id": assembly["dataset_id"], "task_id": assembly["task_id"],
        "tmin": CANONICAL_TMIN, "tmax": CANONICAL_TMAX,
        "endpoint_convention": ENDPOINT_CONVENTION, "sfreq": sfreq, "n_times": n_times,
        "source_n_times": shape[1], "n_trials": len(epochs),
        "source_artifact_id": record["artifact_id"], "source_sha256": record["source_sha256"],
        "source_processing_status": record["processing_status"],
        "trial_assembly_version": assembly["assembly_version"],
        "channel_names": list(metadata["channel_names"]),
        "channel_types": list(metadata["channel_types"]),
        "source_unit_representation": {"header_units": audit.get("units"),
                                       "header_cal": audit.get("cal"),
                                       "loader_conversion": [h["operation"] for h in
                                                             record["processing_history"]
                                                             if h["stage"] == "loader"]},
    }
    return {"signals": signals, "epochs": epochs, "collection": collection}

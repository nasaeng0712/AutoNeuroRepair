"""Actual-A01T Canonical Trial Signal Extraction. A skip means BLOCKED, not a PASS.

Only summary evidence is printed; the signal arrays are never dumped. Counts such
as the observed rejected-trial number are measured here, not assumed.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from anr.event_semantics import interpret_events, require_semantic_complete
from anr.loader import SourceReference, load_a01t
from anr.signal_extraction import extract_trial_signals
from anr.trial_assembly import assemble_from_raw, validate_a01t_trial_structure
from anr.validation import validate_basic_integrity


def snapshot_annotations(raw):
    a = raw.annotations
    return ([float(x) for x in a.onset], [float(x) for x in a.duration],
            [str(x) for x in a.description], a.orig_time)


@pytest.mark.integration
def test_actual_a01t_signal_extraction():
    if not os.environ.get("ANR_A01T_PATH"):
        pytest.skip("BLOCKED: actual A01T.gdf unavailable (ANR_A01T_PATH)")
    names = ("ANR_A01T_REFERENCE_SHA256", "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    given = [name for name in names if os.environ.get(name)]
    if given and len(given) != len(names):
        pytest.skip("BLOCKED: incomplete trusted acquisition record")
    reference = SourceReference(*(os.environ[n] for n in names)) if given else None
    path = Path(os.environ["ANR_A01T_PATH"])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"

    raw, record = load_a01t(path, source_reference=reference)
    try:
        integrity = validate_basic_integrity(raw, record)
        semantic = interpret_events([str(d) for d in raw.annotations.description])
        require_semantic_complete(semantic)
        assembly = assemble_from_raw(raw, record)
        assert validate_a01t_trial_structure(assembly, record)["status"] == "PASS"

        data = raw.get_data()
        before = {"record": deepcopy(record), "integrity": deepcopy(integrity),
                  "semantic": deepcopy(semantic), "assembly": deepcopy(assembly),
                  "data": sha256(np.ascontiguousarray(data)).hexdigest(),
                  "annotations": snapshot_annotations(raw),
                  "channels": (list(raw.ch_names), list(raw.get_channel_types()))}

        out = extract_trial_signals(data, assembly, record)
        signals, epochs, collection = out["signals"], out["epochs"], out["collection"]
        trials = assembly["trials"]
        n_samples, sfreq = raw.n_times, raw.info["sfreq"]

        assert (len(trials), len(epochs), signals.shape[0]) == (288, 288, 288)
        assert sfreq == 250.0 and collection["n_times"] == 1500
        assert signals.shape == (288, len(raw.ch_names), 1500) and signals.shape[1] == 25

        source_violations = span_violations = signal_mismatches = lineage_mismatches = 0
        for i, (trial, epoch) in enumerate(zip(trials, epochs)):
            anchor = trial["anchor"]["source_sample_index"]
            start, end = epoch["signal_start_source_sample"], epoch["signal_end_source_sample_exclusive"]
            source_violations += not (0 <= start and end <= n_samples)
            span_violations += end > trial["span"]["end_sample_exclusive"]
            signal_mismatches += not np.array_equal(signals[i], data[:, anchor:anchor + 1500])
            lineage_mismatches += not (
                epoch["trial_id"] == trial["trial_id"] == f"{record['source_sha256']}:{anchor}"
                and start == epoch["anchor_source_sample_index"] == anchor and end == start + 1500
                and epoch["n_times"] == 1500 and epoch["source_artifact_id"] == record["artifact_id"]
                and epoch["ground_truth_semantic_label"] == trial["semantic_label_id"]
                and epoch["rejection_marker_present"] == trial["rejection_marker_present"]
                and epoch["source_processing_status"] == record["processing_status"]
                and (epoch["tmin"], epoch["tmax"], epoch["endpoint_convention"]) == (0.0, 6.0, "half_open"))
        rejected = [i for i, t in enumerate(trials) if t["rejection_marker_present"]]
        rejected_extracted = sum(epochs[i]["rejection_marker_present"] and
                                 np.array_equal(signals[i], data[:, trials[i]["anchor"]["source_sample_index"]:
                                                                 trials[i]["anchor"]["source_sample_index"] + 1500])
                                 for i in rejected)
        assert (source_violations, span_violations, signal_mismatches, lineage_mismatches) == (0, 0, 0, 0)
        assert rejected_extracted == len(rejected)  # every marked trial extracted, marker kept
        assert len(rejected) == sum(bool(t["rejection_markers"]) for t in trials)

        # Channel axis is the untouched source representation.
        assert collection["channel_names"] == before["channels"][0] == record["metadata"]["channel_names"]
        assert collection["channel_types"] == before["channels"][1]
        assert collection["stage"] == "pre_channel_canonicalization"

        # Extraction mutated nothing upstream.
        mutations = sum([
            assembly != before["assembly"], semantic != before["semantic"],
            record != before["record"], validate_basic_integrity(raw, record) != before["integrity"],
            sha256(np.ascontiguousarray(raw.get_data())).hexdigest() != before["data"],
            sha256(np.ascontiguousarray(data)).hexdigest() != before["data"],
            snapshot_annotations(raw) != before["annotations"],
            (list(raw.ch_names), list(raw.get_channel_types())) != before["channels"]])
        assert mutations == 0
        assert record["processing_status"] == before["record"]["processing_status"]
        assert record["source_sha256"] == sha256(path.read_bytes()).hexdigest()

        print("SIGNAL_EXTRACTION_EVIDENCE", {
            "source_sfreq": sfreq, "source_n_times": n_samples, "source_channels": len(raw.ch_names),
            "assembled_trials": len(trials), "extracted_trials": len(epochs),
            "expected_samples_per_trial": 1500,
            "unique_epoch_shapes": sorted({tuple(s.shape) for s in signals}),
            "collection_shape": list(signals.shape),
            "min_start_sample": min(e["signal_start_source_sample"] for e in epochs),
            "max_end_exclusive_sample": max(e["signal_end_source_sample_exclusive"] for e in epochs),
            "source_boundary_violations": source_violations, "trial_span_violations": span_violations,
            "rejected_trials": len(rejected), "rejected_trials_extracted": rejected_extracted,
            "signal_mismatches": signal_mismatches, "lineage_mismatches": lineage_mismatches,
            "channel_order_mismatches": int(collection["channel_names"] != list(raw.ch_names)),
            "upstream_mutations": mutations,
            "mne_signal_unit_code": sorted({int(c["unit"]) for c in raw.info["chs"]}),
            "processing_status": record["processing_status"],
            "basic_integrity": integrity["basic_integrity"]["status"]})
    finally:
        raw.close()

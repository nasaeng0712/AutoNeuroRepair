"""Actual-A01T Phase A (Channel Canonicalization) and Phase B (Canonical Trial Dataset).

A skip means BLOCKED, not a PASS. Shapes/counts are checked against the
approved A01T structure and never adjusted.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from anr.channel_canonicalization import canonicalize_channels
from anr.loader import SourceReference, load_a01t
from anr.pipeline import run_pipeline
from anr.signal_extraction import extract_trial_signals
from anr.trial_assembly import assemble_from_raw


def a01t_inputs():
    if not os.environ.get("ANR_A01T_PATH"):
        pytest.skip("BLOCKED: actual A01T.gdf unavailable (ANR_A01T_PATH)")
    names = ("ANR_A01T_REFERENCE_SHA256", "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    given = [name for name in names if os.environ.get(name)]
    if given and len(given) != len(names):
        pytest.skip("BLOCKED: incomplete trusted acquisition record")
    path = Path(os.environ["ANR_A01T_PATH"])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"
    return path, (SourceReference(*(os.environ[n] for n in names)) if given else None)


@pytest.mark.integration
def test_phase_a_actual_a01t_channel_canonicalization():
    path, reference = a01t_inputs()
    raw, record = load_a01t(path, source_reference=reference)
    try:
        assembly = assemble_from_raw(raw, record)
        data = raw.get_data()
        extraction = extract_trial_signals(data, assembly, record)
        source_names, source_types = list(raw.ch_names), list(raw.get_channel_types())
        out = canonicalize_channels(extraction["signals"], source_names, source_types)
        records = out["eeg_channels"] + out["eog_channels"]
        print("PHASE_A_EVIDENCE", {
            "X_eeg": list(out["X_eeg"].shape), "X_eog": list(out["X_eog"].shape),
            "source_to_canonical": [(c["source_channel_index"], c["source_channel_name"],
                                     c["canonical_channel_id"], c["channel_type"]) for c in records]})

        assert out["X_eeg"].shape == (288, 22, 1500) and out["X_eog"].shape == (288, 3, 1500)
        assert sorted(c["source_channel_index"] for c in records) == list(range(25))  # exactly once
        assert len({c["canonical_channel_id"] for c in records}) == 25
        assert [c["channel_type"] for c in out["eeg_channels"]] == ["eeg"] * 22
        assert [c["channel_type"] for c in out["eog_channels"]] == ["eog"] * 3
        assert all(source_types[c["source_channel_index"]] == c["channel_type"]
                   and source_names[c["source_channel_index"]] == c["source_channel_name"]
                   for c in records)
        for k, c in enumerate(out["eeg_channels"]):
            assert np.array_equal(out["X_eeg"][:, k, :], extraction["signals"][:, c["source_channel_index"], :])
        for k, c in enumerate(out["eog_channels"]):
            assert np.array_equal(out["X_eog"][:, k, :], extraction["signals"][:, c["source_channel_index"], :])
        for i, trial in enumerate(assembly["trials"]):  # also equals the raw source slice
            anchor = trial["anchor"]["source_sample_index"]
            assert np.array_equal(np.concatenate([out["X_eeg"][i], out["X_eog"][i]]),
                                  data[:, anchor:anchor + 1500])
        assert [e["trial_id"] for e in extraction["epochs"]] == [t["trial_id"] for t in assembly["trials"]]
    finally:
        raw.close()


@pytest.mark.integration
def test_phase_b_actual_a01t_canonical_trial_dataset():
    path, reference = a01t_inputs()
    result = run_pipeline(path, source_reference=reference)  # gates incl. dataset validation PASS
    dataset, record, validation = result["dataset"], result["record"], result["dataset_validation"]
    info, meta = dataset["dataset_metadata"], dataset["trial_metadata"]
    assert validation["status"] == "PASS", [c for c in validation["checks"] if c["status"] == "FAIL"]
    assert dataset["X_eeg"].shape == (288, 22, 1500) and dataset["X_eog"].shape == (288, 3, 1500)
    assert dataset["y_semantic"].shape == (288,) and len(meta) == 288
    assert sorted(set(dataset["y_semantic"].tolist())) == ["both_feet", "left_hand", "right_hand", "tongue"]
    assert np.isfinite(dataset["X_eeg"]).all() and np.isfinite(dataset["X_eog"]).all()
    assert dataset["canonical_dataset_id"] == f"{record['source_sha256']}:canonical-trial-dataset-v1"
    assert record["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert [m["trial_id"] for m in meta] == [t["trial_id"] for t in result["assembly"]["trials"]]
    assert sum(m["rejection_marker_present"] for m in meta) == sum(
        bool(t["rejection_markers"]) for t in result["assembly"]["trials"])
    assert info["source_processing_status"] == record["processing_status"]
    assert info["source_provenance_ref"]["artifact_id"] == record["artifact_id"]
    assert info["source_artifact_id"] == record["artifact_id"] and info["sfreq"] == 250.0
    assert info["epoch_definition"]["n_times"] == 1500
    assert sorted({m["task_run_index"] for m in meta}) == [1, 2, 3, 4, 5, 6]
    assert info["observed_label_coverage"]["native_event_counts"]["768"] == 288
    print("PHASE_B_EVIDENCE", {
        "canonical_dataset_id": dataset["canonical_dataset_id"],
        "X_eeg": list(dataset["X_eeg"].shape), "X_eog": list(dataset["X_eog"].shape),
        "y_semantic": list(dataset["y_semantic"].shape), "trial_metadata": len(meta),
        "rejected_trials": sum(m["rejection_marker_present"] for m in meta),
        "source_processing_status": info["source_processing_status"],
        "history": [h["step"] for h in info["pipeline_transformation_history"]],
        "validation": validation["status"]})

"""Unit tests for anr.canonical_dataset (small synthetic source signals)."""

from copy import deepcopy

import numpy as np
import pytest

from anr.canonical_dataset import (
    build_canonical_dataset, canonical_dataset_id, validate_canonical_dataset,
)
from anr.channel_canonicalization import ChannelCanonicalizationError, channel_schema_v1
from anr.event_semantics import interpret_events, observe_coverage
from anr.signal_extraction import extract_trial_signals
from anr.trial_assembly import assemble_trials
from test_validation import make

SF = 250.0
EVENTS = [("32766", 0), ("768", 1000), ("769", 1500), ("768", 3000), ("1023", 3000),
          ("770", 3500), ("768", 5000), ("771", 5500)]


@pytest.fixture
def parts(tmp_path):
    record = deepcopy(make(tmp_path)[1])
    schema = channel_schema_v1()
    record["metadata"]["channel_names"] = [s["source_channel_name"] for s in schema]
    record["metadata"]["channel_types"] = [s["channel_type"] for s in schema]
    codes = [c for c, _ in EVENTS]
    assembly = assemble_trials(interpret_events(codes), [s for _, s in EVENTS], sfreq=SF,
                               n_samples=8000, source_sha256=record["source_sha256"],
                               source_artifact_id=record["artifact_id"])
    data = np.arange(25 * 8000, dtype=float).reshape(25, 8000)
    extraction = extract_trial_signals(data, assembly, record)
    dataset = build_canonical_dataset(extraction, assembly, record, observe_coverage(codes))
    return data, record, assembly, extraction, dataset


def failed(result):
    return {c["check_id"] for c in result["checks"] if c["status"] == "FAIL"}


def test_shapes_labels_metadata_and_invariants(parts):
    data, record, assembly, extraction, dataset = parts
    assert dataset["X_eeg"].shape == (3, 22, 1500) and dataset["X_eog"].shape == (3, 3, 1500)
    assert dataset["y_semantic"].shape == (3,) and len(dataset["trial_metadata"]) == 3
    assert dataset["y_semantic"].tolist() == ["left_hand", "right_hand", "both_feet"]
    result = validate_canonical_dataset(dataset, record)
    assert result["status"] == "PASS", failed(result)
    assert np.isfinite(dataset["X_eeg"]).all() and np.isfinite(dataset["X_eog"]).all()


def test_signal_values_are_untouched_selection_and_reorder_only(parts):
    data, _, assembly, extraction, dataset = parts
    for i, trial in enumerate(assembly["trials"]):
        start = trial["anchor"]["source_sample_index"]
        window = data[:, start:start + 1500]
        assert np.array_equal(dataset["X_eeg"][i], window[:22])
        assert np.array_equal(dataset["X_eog"][i], window[22:])
    assert np.array_equal(dataset["X_eeg"].ravel().sum() + dataset["X_eog"].ravel().sum(),
                          extraction["signals"].sum())


def test_stable_trial_ids_and_order_unchanged(parts):
    _, record, assembly, extraction, dataset = parts
    ids = [m["trial_id"] for m in dataset["trial_metadata"]]
    assert ids == [t["trial_id"] for t in assembly["trials"]] == [e["trial_id"] for e in extraction["epochs"]]
    assert ids[0] == f"{record['source_sha256']}:1000"


def test_rejection_markers_and_run_identity_preserved(parts):
    meta = parts[4]["trial_metadata"]
    assert [m["rejection_marker_present"] for m in meta] == [False, True, False]
    assert meta[1]["rejection_markers"] == [{"event_index": 4, "source_sample_index": 3000}]
    assert meta[0]["rejection_markers"] == []
    assert [m["task_run_index"] for m in meta] == [1, 1, 1]
    assert all(m["run_segment_index"] == 1 for m in meta)
    assert [m["cue_source_sample_index"] for m in meta] == [1500, 3500, 5500]


def test_lineage_and_processing_status_snapshot(parts):
    _, record, _, _, dataset = parts
    info = dataset["dataset_metadata"]
    sha = record["source_sha256"]
    assert dataset["canonical_dataset_id"] == info["canonical_dataset_id"] == f"{sha}:canonical-trial-dataset-v1"
    assert canonical_dataset_id(sha) == f"{sha}:canonical-trial-dataset-v1"
    assert info["source_sha256"] == sha and info["source_artifact_id"] == record["artifact_id"]
    assert info["source_provenance_ref"] == {"artifact_id": record["artifact_id"], "source_sha256": sha,
                                             "provenance_schema_version": record["schema_version"]}
    assert info["source_processing_status"] == record["processing_status"]
    assert (info["dataset_id"], info["task_id"]) == ("BCIC_IV_2A", "motor_imagery")
    assert (info["pipeline_schema_version"], info["channel_schema_version"]) == (
        "canonical-trial-dataset-v1", "channel_schema_v1")
    assert all(m["source_processing_status"] == record["processing_status"]
               for m in dataset["trial_metadata"])


def test_dataset_does_not_reassess_provenance(parts):
    _, record, _, _, dataset = parts
    before = deepcopy(record)
    validate_canonical_dataset(dataset, record)
    assert record == before


def test_dataset_id_is_deterministic(parts, tmp_path):
    _, record, assembly, extraction, dataset = parts
    again = build_canonical_dataset(extraction, assembly, record, dataset["dataset_metadata"]["observed_label_coverage"])
    assert again["canonical_dataset_id"] == dataset["canonical_dataset_id"]


def test_dataset_metadata_content_and_history(parts):
    info = parts[4]["dataset_metadata"]
    assert len(info["eeg_channel_schema"]) == 22 and len(info["eog_channel_schema"]) == 3
    assert set(info["task_label_schema"]["labels"]) == {"left_hand", "right_hand", "both_feet", "tongue"}
    assert info["observed_label_coverage"]["native_event_counts"]["768"] == 3  # observed, separate
    assert info["sfreq"] == 250.0
    assert info["epoch_definition"] == {"tmin": 0.0, "tmax": 6.0, "endpoint_convention": "half_open",
                                        "n_times": 1500}
    assert [h["step"] for h in info["pipeline_transformation_history"]] == [
        "trial_signal_selection", "channel_canonicalization"]
    history = " ".join(h["operation"] for h in info["pipeline_transformation_history"])
    assert not any(word in history for word in ("filter", "crop", "CSP", "8-30", "Hz", "resampl"))
    assert "unit_representation" in info["signal_representation"]


def test_no_baseline_filtering_model_index_or_features(parts):
    dataset = parts[4]
    names = set(dataset) | set(dataset["dataset_metadata"]) | {k for m in dataset["trial_metadata"] for k in m}
    assert not {n for n in names if "model_class" in n or "feature" in n or "baseline" in n}
    assert dataset["X_eeg"].shape[2] == 1500  # no [2, 6) crop, no filtering


def test_invariant_failures_are_detected(parts):
    _, record, _, _, dataset = parts
    bad = deepcopy(dataset)
    bad["X_eeg"][0, 0, 0] = np.nan
    assert "FINITE_VALUES" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["y_semantic"] = bad["y_semantic"][:2]
    assert "LABEL_METADATA_COUNTS" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["trial_metadata"][1]["n_times"] = 1499
    assert "COMMON_EPOCH_DEFINITION" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["trial_metadata"][2]["sfreq"] = 128.0
    assert "COMMON_SIGNAL_REPRESENTATION" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["y_semantic"][0] = "feet"
    assert "SEMANTIC_LABELS" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["trial_metadata"][0]["trial_id"] = "x:1"
    assert "STABLE_TRIAL_IDS" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["dataset_metadata"]["eeg_channel_schema"] = bad["dataset_metadata"]["eeg_channel_schema"][:21]
    assert "CHANNEL_SCHEMA" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["canonical_dataset_id"] = "other"
    assert "DATASET_LINEAGE" in failed(validate_canonical_dataset(bad, record))
    bad = deepcopy(dataset)
    bad["trial_metadata"][0]["model_class_index"] = 0
    assert "NO_MODEL_FIELDS" in failed(validate_canonical_dataset(bad, record))


def test_wrong_channels_stop_dataset_construction(parts):
    _, record, assembly, extraction, dataset = parts
    broken = deepcopy(extraction)
    broken["collection"]["channel_names"][0] = "Fz"
    with pytest.raises(ChannelCanonicalizationError, match="unvalidated alias"):
        build_canonical_dataset(broken, assembly, record, {})

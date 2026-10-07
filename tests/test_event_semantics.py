"""Unit tests for anr.event_semantics (synthetic decoded annotations only)."""

from copy import deepcopy

import numpy as np
import pytest

from anr.event_semantics import (
    DATASET_ID, SEMANTIC_SCHEMA_VERSION, SemanticIncompleteError, event_semantic_schema,
    interpret_event, interpret_events, observe_coverage, require_semantic_complete,
    task_label_schema,
)
from anr.validation import validate_basic_integrity
from test_validation import make

EXPECTED = {  # code: (role, task_id, label, name)
    "769": ("task_label", "motor_imagery", "left_hand", "Left hand motor imagery"),
    "770": ("task_label", "motor_imagery", "right_hand", "Right hand motor imagery"),
    "771": ("task_label", "motor_imagery", "both_feet", "Both feet motor imagery"),
    "772": ("task_label", "motor_imagery", "tongue", "Tongue motor imagery"),
    "768": ("trial_anchor", "motor_imagery", None, "Start of a trial"),
    "783": ("unlabeled_task_cue", "motor_imagery", None, "Unknown cue"),
    "1023": ("trial_rejection_marker", "motor_imagery", None, "Rejected trial"),
    "276": ("auxiliary_recording_marker", None, None, "Idling EEG (eyes open)"),
    "277": ("auxiliary_recording_marker", None, None, "Idling EEG (eyes closed)"),
    "1072": ("auxiliary_recording_marker", None, None, "Eye movements"),
    "32766": ("run_start", "motor_imagery", None, "Start of a new run"),
}


def keys_of(value):
    """All dict keys anywhere in a nested result."""
    if isinstance(value, dict):
        return set(value) | set().union(*(keys_of(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(keys_of(v) for v in value), set())
    return set()


@pytest.mark.parametrize("code,expected", EXPECTED.items())
def test_approved_mapping(code, expected):
    record = interpret_event(code)
    assert (record["event_role"], record["task_id"], record["semantic_label_id"],
            record["human_readable_name"]) == expected
    assert record["native_event_code"] == code
    assert set(record) == {"native_event_code", "event_role", "task_id",
                           "semantic_label_id", "human_readable_name"}


def test_both_feet_is_not_feet():
    assert interpret_event("771")["semantic_label_id"] == "both_feet"
    assert "feet" not in {r["semantic_label_id"] for r in event_semantic_schema()["events"].values()}


def test_decoded_numpy_string_is_accepted():
    assert interpret_event(np.str_("769"))["semantic_label_id"] == "left_hand"


def test_native_code_and_semantic_label_are_separate_fields():
    record = interpret_event("769")
    assert record["native_event_code"] == "769" and record["semantic_label_id"] == "left_hand"
    assert record["native_event_code"] != record["semantic_label_id"]


def test_schema_metadata_persisted():
    schema = event_semantic_schema()
    assert schema["dataset_id"] == DATASET_ID == "BCIC_IV_2A"
    assert schema["semantic_schema_version"] == SEMANTIC_SCHEMA_VERSION
    assert schema["normative_source"].startswith("https://")
    assert schema["events"]["770"]["human_readable_name"] == "Right hand motor imagery"
    assert set(schema["events"]) == set(EXPECTED)  # 783 present although unobserved


def test_32766_is_task_scoped_run_start_not_a_prediction_class():
    record = interpret_event("32766")
    assert record["task_id"] == "motor_imagery" and record["event_role"] == "run_start"
    assert record["semantic_label_id"] is None
    assert record["human_readable_name"] == "Start of a new run"
    assert "32766" in event_semantic_schema()["events"]
    assert None not in task_label_schema()["labels"]
    assert "run_start" not in {k for k in task_label_schema()["labels"]}
    assert not {k for k in keys_of(record) if "class" in k or "index" in k}
    assert observe_coverage(["32766"])["task_label_counts"] == {
        "left_hand": 0, "right_hand": 0, "both_feet": 0, "tongue": 0}


@pytest.mark.parametrize("code", ["999", "abc", "", "0", "2"])
def test_undefined_native_event_is_preserved_as_unknown(code):
    record = interpret_event(code)
    assert record == {"native_event_code": code, "event_role": "unknown_native_event",
                      "task_id": None, "semantic_label_id": None, "human_readable_name": None}
    assert code not in event_semantic_schema()["events"]  # schema is not extended
    assert record["semantic_label_id"] not in task_label_schema()["labels"]
    records = interpret_events(["769", code, "768"])  # no crash; nothing dropped or guessed
    assert [r["native_event_code"] for r in records] == ["769", code, "768"]
    assert [r["event_role"] for r in records] == ["task_label", "unknown_native_event", "trial_anchor"]
    coverage = observe_coverage(["769", code, "768"])
    assert coverage["native_event_counts"][code] == 1 and coverage["total_events"] == 3
    assert coverage["undefined_native_codes"] == [code]
    assert sum(coverage["task_label_counts"].values()) == 1  # not promoted to a label


def test_unknown_event_fails_closed_for_handoff():
    records = interpret_events(["768", "769", "999", "32766"])
    with pytest.raises(SemanticIncompleteError, match="999"):
        require_semantic_complete(records)
    # Nothing in any output claims semantic completeness or handoff eligibility.
    outputs = [records, observe_coverage(["768", "999"]), event_semantic_schema(), task_label_schema()]
    assert not {k for out in outputs for k in keys_of(out)
                if "complete" in k or "eligib" in k or "handoff" in k}
    # Only fully approved events pass the check; it implies nothing beyond that.
    assert require_semantic_complete(interpret_events(["768", "769", "1023", "783", "32766"])) is None
    assert require_semantic_complete([]) is None


def test_interpret_events_preserves_order_and_every_event():
    records = interpret_events(["768", "769", "1023", "276", "768"])
    assert [r["native_event_code"] for r in records] == ["768", "769", "1023", "276", "768"]


def test_task_label_schema_has_exactly_four_supervised_labels():
    schema = task_label_schema()
    assert set(schema["labels"]) == {"left_hand", "right_hand", "both_feet", "tongue"}
    assert schema["task_id"] == "motor_imagery" and schema["dataset_id"] == DATASET_ID
    assert schema["labels"]["both_feet"] == "Both feet motor imagery"
    names = {r["human_readable_name"] for r in event_semantic_schema()["events"].values()}
    assert "Unknown cue" in names and "Unknown cue" not in schema["labels"].values()


def test_no_model_class_index_or_order_anywhere():
    outputs = [event_semantic_schema(), task_label_schema(), observe_coverage(["769", "999"]),
               interpret_events(["769", "771"])]
    for out in outputs:
        assert not {k for k in keys_of(out) if "class" in k or "index" in k or "order" in k}


def test_schema_has_no_observed_counts_and_is_not_shared_state():
    schema = event_semantic_schema()
    assert not {k for k in keys_of(schema) if "count" in k}
    schema["events"]["769"]["semantic_label_id"] = "tampered"
    assert event_semantic_schema()["events"]["769"]["semantic_label_id"] == "left_hand"


def test_coverage_counts_synthetic_annotations():
    events = ["768"] * 4 + ["769", "769", "770", "771", "1023", "276", "32766", "32766", "999"]
    coverage = observe_coverage(events)
    counts = coverage["native_event_counts"]
    assert coverage["total_events"] == len(events)
    assert (counts["768"], counts["769"], counts["770"], counts["771"], counts["772"]) == (4, 2, 1, 1, 0)
    assert (counts["1023"], counts["276"], counts["277"], counts["1072"]) == (1, 1, 0, 0)
    assert counts["32766"] == 2 and counts["999"] == 1  # unknown 999 counted, never dropped
    assert coverage["undefined_native_codes"] == ["999"]
    assert coverage["task_label_counts"] == {"left_hand": 2, "right_hand": 1,
                                             "both_feet": 1, "tongue": 0}


def test_coverage_is_separate_from_schema_and_keeps_unobserved_783():
    schema_before = deepcopy(event_semantic_schema())
    coverage = observe_coverage([])  # nothing observed at all
    assert coverage["native_event_counts"]["783"] == 0 and coverage["total_events"] == 0
    assert set(coverage["task_label_counts"]) == set(task_label_schema()["labels"])
    assert "events" not in coverage and "labels" not in coverage
    assert event_semantic_schema() == schema_before and "783" in schema_before["events"]


def test_task_label_coverage_is_not_native_event_coverage():
    coverage = observe_coverage(["768", "768", "769", "783"])
    assert coverage["task_label_counts"]["left_hand"] == 1
    assert sum(coverage["task_label_counts"].values()) == 1  # anchor / 783 are not labels
    assert coverage["native_event_counts"]["768"] == 2 and coverage["native_event_counts"]["783"] == 1


def test_event_semantics_leaves_provenance_integrity_and_raw_unchanged(tmp_path):
    raw, record = make(tmp_path)
    raw.data[1, 1] = np.nan
    descriptions = np.array(["768", "769", "1023", "32766", "999"])
    record_before, data_before = deepcopy(record), raw.data.copy()
    annotations_before = deepcopy(raw.annotations)
    integrity_before = validate_basic_integrity(raw, record)
    observe_coverage(descriptions)
    interpret_events(descriptions)
    assert record == record_before and record["processing_status"] == record_before["processing_status"]
    assert validate_basic_integrity(raw, record) == integrity_before
    assert np.array_equal(raw.data, data_before, equal_nan=True)
    assert raw.annotations == annotations_before
    assert list(descriptions) == ["768", "769", "1023", "32766", "999"]

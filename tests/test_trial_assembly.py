"""Unit tests for anr.trial_assembly (synthetic events; no signal arrays)."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from anr.event_semantics import SemanticIncompleteError, interpret_events
from anr.trial_assembly import (
    TrialAssemblyError, assemble_from_raw, assemble_trials, source_sample_indices,
    validate_a01t_trial_structure,
)
from test_validation import make

SHA = "a" * 64
SF = 250.0
CUE = 500  # 2 * sfreq


def assemble(events, *, n_samples=100_000, artifact="artifact-1", sha=SHA):
    """events: [(native_code, sample)] already in source order."""
    records = interpret_events([c for c, _ in events])
    return assemble_trials(records, [s for _, s in events], sfreq=SF, n_samples=n_samples,
                           source_sha256=sha, source_artifact_id=artifact)


def one_trial(label="769", anchor=1000, cue=CUE, extra=()):
    return [("32766", 0), ("768", anchor), *extra, (label, anchor + cue)]


def test_valid_trial_metadata_and_identity_scope():
    result = assemble(one_trial("771"))
    trial = result["trials"][0]
    assert len(result["trials"]) == 1
    assert (trial["dataset_id"], trial["task_id"]) == ("BCIC_IV_2A", "motor_imagery")
    assert trial["trial_id"] == f"{SHA}:1000"
    assert trial["semantic_label_id"] == "both_feet"
    assert trial["anchor"]["event"]["native_event_code"] == "768"
    assert trial["task_label_event"]["event"]["native_event_code"] == "771"  # native code kept
    assert trial["anchor"]["source_sample_index"] == 1000
    assert trial["task_label_event"]["source_sample_index"] == 1500
    assert trial["source_sha256"] == SHA and trial["source_artifact_id"] == "artifact-1"
    assert trial["cue_delta_samples"] == CUE and not trial["rejection_marker_present"]
    assert trial["run_segment_index"] == 1 and trial["run_start_event_index"] == 0


def test_semantic_gate_fails_closed_without_metadata():
    records = interpret_events(["32766", "768", "999", "769"])
    with pytest.raises(SemanticIncompleteError):
        assemble_trials(records, [0, 10, 20, 510], sfreq=SF, n_samples=1000,
                        source_sha256=SHA, source_artifact_id="x")


def test_no_label_in_span_fails():
    with pytest.raises(TrialAssemblyError, match="found 0"):
        assemble([("32766", 0), ("768", 1000)])


def test_two_labels_in_one_span_fail():
    with pytest.raises(TrialAssemblyError, match="found 2"):
        assemble(one_trial(extra=[("770", 1200)]))


def test_span_ends_at_next_anchor_and_next_label_is_not_borrowed():
    # The only label lies after the second anchor, so the first span has none:
    # there is no global "next label" search.
    with pytest.raises(TrialAssemblyError, match="Anchor at sample 1000.*found 0"):
        assemble([("32766", 0), ("768", 1000), ("768", 1100), ("769", 1600)])


def test_span_ends_at_next_run_start():
    with pytest.raises(TrialAssemblyError, match="Anchor at sample 1000.*found 0"):
        assemble([("32766", 0), ("768", 1000), ("32766", 1200), ("769", 1500)])


def test_span_ends_at_source_end():
    result = assemble(one_trial(), n_samples=1501)
    assert result["trials"][0]["span"] == {"start_sample": 1000, "end_sample_exclusive": 1501}
    with pytest.raises(TrialAssemblyError, match="found 0"):
        assemble(one_trial(), n_samples=1500)  # label sample 1500 is outside [1000, 1500)


def test_span_end_is_exclusive_and_start_inclusive():
    events = [("32766", 0), ("768", 1000), ("769", 1500), ("768", 2000), ("770", 2500)]
    spans = [t["span"] for t in assemble(events)["trials"]]
    assert spans == [{"start_sample": 1000, "end_sample_exclusive": 2000},
                     {"start_sample": 2000, "end_sample_exclusive": 100_000}]


def test_orphan_task_label_fails():
    with pytest.raises(TrialAssemblyError, match="Orphan"):
        assemble([("32766", 0), ("769", 400), ("768", 1000), ("769", 1500)])  # label before any anchor
    with pytest.raises(TrialAssemblyError, match="Orphan"):
        assemble([*one_trial(), ("32766", 3000), ("772", 3100)])  # label in a run with no anchor


def test_label_cannot_associate_with_multiple_anchors():
    events = [("32766", 0), ("768", 1000), ("768", 1100), ("769", 1500)]
    with pytest.raises(TrialAssemblyError):
        assemble(events)  # first span has no label of its own; the label is not shared


@pytest.mark.parametrize("cue", [CUE - 1, CUE + 1, 250])
def test_cue_timing_must_be_exactly_two_seconds(cue):
    with pytest.raises(TrialAssemblyError, match="cue delta"):
        assemble(one_trial(cue=cue))


def test_cue_timing_is_two_times_sfreq():
    assert assemble(one_trial(cue=int(2 * SF)))["trials"][0]["cue_delta_samples"] == 500


def test_trial_id_is_source_hash_and_anchor_sample_not_load_or_ordinal():
    first = assemble(one_trial(anchor=1000), artifact="load-1")["trials"][0]
    again = assemble(one_trial(anchor=1000), artifact="load-2")["trials"][0]
    assert first["trial_id"] == again["trial_id"] == f"{SHA}:1000"
    assert first["source_artifact_id"] != again["source_artifact_id"]  # lineage only
    other = assemble(one_trial(anchor=2000))["trials"][0]
    assert other["trial_id"] == f"{SHA}:2000" != first["trial_id"]
    # Position in the list does not matter: the second trial's id uses its sample.
    two = assemble([("32766", 0), ("768", 1000), ("769", 1500), ("768", 2000), ("770", 2500)])
    assert [t["trial_id"] for t in two["trials"]] == [f"{SHA}:1000", f"{SHA}:2000"]
    assert assemble(one_trial(), sha="b" * 64)["trials"][0]["trial_id"] == f"{'b' * 64}:1000"


def test_zero_rejection_markers():
    trial = assemble(one_trial())["trials"][0]
    assert trial["rejection_markers"] == [] and trial["rejection_marker_present"] is False


def test_rejection_marker_at_anchor_sample_is_preserved_not_deleting_trial():
    result = assemble(one_trial(extra=[("1023", 1000)]))  # shares the anchor sample
    trial = result["trials"][0]
    assert len(result["trials"]) == 1 and trial["rejection_marker_present"]
    assert [m["source_sample_index"] for m in trial["rejection_markers"]] == [1000]
    assert trial["rejection_markers"][0]["event"]["event_role"] == "trial_rejection_marker"
    assert "include" not in " ".join(trial) and "exclu" not in " ".join(trial)  # no Baseline decision


def test_multiple_rejection_markers_in_one_span_all_preserved():
    result = assemble(one_trial(extra=[("1023", 1000), ("1023", 1300)]))
    trial = result["trials"][0]
    assert [m["event_index"] for m in trial["rejection_markers"]] == [2, 3]
    assert trial["rejection_marker_present"]
    assert result["evidence"]["observed_rejection_markers"] == 2 == result["evidence"]["associated_rejection_markers"]


def test_rejection_marker_outside_every_span_fails():
    with pytest.raises(TrialAssemblyError, match="Orphan rejection"):
        assemble([("32766", 0), ("1023", 400), ("768", 1000), ("769", 1500)])
    with pytest.raises(TrialAssemblyError, match="Orphan rejection"):
        assemble([*one_trial(), ("32766", 3000), ("1023", 3100)])


def test_duplicate_anchor_sample_and_unordered_events_fail():
    with pytest.raises(TrialAssemblyError, match="share"):
        assemble([("32766", 0), ("768", 1000), ("768", 1000), ("769", 1500)])
    with pytest.raises(TrialAssemblyError, match="source order"):
        assemble([("32766", 0), ("769", 1500), ("768", 1000)])


def test_runs_group_only_task_bearing_segments():
    events = [("32766", 0), ("276", 0), ("32766", 50), ("768", 1000), ("769", 1500),
              ("32766", 5000), ("768", 6000), ("770", 6500)]
    result = assemble(events)
    assert [r["run_segment_index"] for r in result["task_bearing_runs"]] == [2, 3]
    assert [len(r["trial_ids"]) for r in result["task_bearing_runs"]] == [1, 1]
    assert [r["run_start_event_index"] for r in result["task_bearing_runs"]] == [2, 5]


def test_separation_of_native_code_label_and_trial_identity_no_class_index():
    trial = assemble(one_trial("772"))["trials"][0]
    assert trial["task_label_event"]["event"]["native_event_code"] == "772"
    assert trial["semantic_label_id"] == "tongue" and trial["trial_id"].startswith(SHA)
    keys = set(trial) | set(trial["anchor"]) | set(trial["task_label_event"]["event"])
    assert not {k for k in keys if "class" in k or "model" in k}


def test_assembly_does_not_mutate_inputs():
    records = interpret_events(["32766", "768", "1023", "769"])
    samples = [0, 1000, 1000, 1500]
    before = deepcopy(records), list(samples)
    result = assemble_trials(records, samples, sfreq=SF, n_samples=10_000,
                             source_sha256=SHA, source_artifact_id="x")
    assert (records, samples) == before
    result["trials"][0]["anchor"]["event"]["event_role"] = "tampered"
    assert records == before[0]  # output holds copies, not aliases


def test_no_signal_extraction_fields():
    result = assemble(one_trial())
    names = set(result) | set(result["trials"][0])
    assert not {n for n in names if n in {"X", "epochs", "tmin", "tmax", "data", "signal"}}


class FakeAnnotations:
    def __init__(self, descriptions, samples, sfreq=SF, orig_time=None):
        self.description = descriptions
        self.onset = [s / sfreq for s in samples]
        self.orig_time = orig_time


def fake_raw(events, *, first_samp=0, orig_time=None, meas_date=None, n_times=100_000):
    codes, samples = zip(*events)
    return SimpleNamespace(annotations=FakeAnnotations(list(codes), samples, orig_time=orig_time),
                           info={"sfreq": SF, "meas_date": meas_date}, first_samp=first_samp,
                           n_times=n_times)


def test_sample_indices_are_exact_and_in_range():
    assert source_sample_indices([0.0, 4.0, 2683 / SF], SF, 5000) == [0, 1000, 2683]
    with pytest.raises(TrialAssemblyError, match="exact sample"):
        source_sample_indices([0.0041], SF, 5000)
    with pytest.raises(TrialAssemblyError, match="outside"):
        source_sample_indices([20.0], SF, 5000)  # sample 5000 == n_samples


def test_assemble_from_raw_uses_decoded_annotations_and_record(tmp_path):
    _, record = make(tmp_path)
    raw = fake_raw(one_trial("770"))
    result = assemble_from_raw(raw, record)
    assert result["trials"][0]["trial_id"] == f"{record['source_sha256']}:1000"
    assert result["trials"][0]["source_artifact_id"] == record["artifact_id"]
    assert result["trials"][0]["semantic_label_id"] == "right_hand"


@pytest.mark.parametrize("kwargs", [{"first_samp": 5}, {"orig_time": "2005-01-17T12:00:00"}])
def test_annotations_must_be_relative_to_source_start(tmp_path, kwargs):
    _, record = make(tmp_path)
    with pytest.raises(TrialAssemblyError, match="relative"):
        assemble_from_raw(fake_raw(one_trial(), **kwargs), record)


def test_a01t_structure_criteria_pass_and_fail(tmp_path):
    _, record = make(tmp_path)
    events = [("32766", 0)]
    for run in range(6):
        events.append(("32766", 1000 + 100_000 * run))
        base = events[-1][1]
        for k in range(48):
            anchor = base + 1000 * (k + 1)
            events += [("768", anchor), ("1023", anchor) if k == 0 else ("276", anchor),
                       (("769", "770", "771", "772")[k % 4], anchor + CUE)]
    ordered = sorted(events, key=lambda e: e[1])
    result = assemble(ordered, n_samples=1_000_000)
    verdict = validate_a01t_trial_structure(result, record)
    assert verdict["status"] == "PASS", [c for c in verdict["checks"] if c["status"] == "FAIL"]
    assert verdict["artifact_id"] == record["artifact_id"]
    # One trial fewer (its anchor, aux marker and label are the last three events):
    # the generic rules still hold, so the normative A01T criteria must fail.
    broken = validate_a01t_trial_structure(assemble(ordered[:-3], n_samples=1_000_000), record)
    assert broken["status"] == "FAIL"
    assert {"ANCHOR_COUNT", "OVERALL_LABEL_COVERAGE", "TRIALS_PER_RUN"} <= {
        c["check_id"] for c in broken["checks"] if c["status"] == "FAIL"}

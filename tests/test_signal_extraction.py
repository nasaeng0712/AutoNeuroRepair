"""Unit tests for anr.signal_extraction (small synthetic source signals)."""

from copy import deepcopy

import numpy as np
import pytest

from anr.event_semantics import interpret_events
from anr.signal_extraction import SignalExtractionError, extract_trial_signals, window_n_times
from anr.trial_assembly import assemble_trials
from test_validation import make

N = 1500  # 6.0 s at 250 Hz
SF = 250.0


@pytest.fixture
def record(tmp_path):
    return make(tmp_path)[1]  # 25 synthetic channels; only its lineage is used


def build(record, events, n_samples):
    """Assemble real Trial Assembly metadata for (code, sample) events."""
    records = interpret_events([c for c, _ in events])
    assembly = assemble_trials(records, [s for _, s in events], sfreq=SF, n_samples=n_samples,
                               source_sha256=record["source_sha256"],
                               source_artifact_id=record["artifact_id"])
    data = np.arange(25 * n_samples, dtype=float).reshape(25, n_samples)
    return data, assembly


def two_trials(record, second_anchor=2500, n_samples=4000):
    return build(record, [("32766", 0), ("768", 1000), ("769", 1500),
                          ("768", second_anchor), ("770", second_anchor + 500)], n_samples)


def test_window_is_exactly_anchor_to_anchor_plus_1500_half_open(record):
    data, assembly = two_trials(record)
    out = extract_trial_signals(data, assembly, record)
    assert out["signals"].shape == (2, 25, N)
    for epoch, trial in zip(out["epochs"], assembly["trials"]):
        anchor = trial["anchor"]["source_sample_index"]
        assert epoch["signal_start_source_sample"] == epoch["anchor_source_sample_index"] == anchor
        assert epoch["signal_end_source_sample_exclusive"] == anchor + N
        assert epoch["signal_end_source_sample_exclusive"] - epoch["signal_start_source_sample"] == epoch["n_times"] == N
    first = out["signals"][0]
    assert np.array_equal(first[:, 0], data[:, 1000])          # anchor sample included
    assert np.array_equal(first[:, -1], data[:, 1000 + N - 1])  # last included sample = anchor + 1499
    assert not np.array_equal(first[:, -1], data[:, 1000 + N])  # anchor + 1500 is excluded
    assert np.array_equal(first, data[:, 1000:2500])


def test_window_n_times_exact_integer_only():
    assert window_n_times(250.0) == 1500
    assert window_n_times(128.0) == 768
    for bad in (250.3, 0.7, 0.0, -250.0, float("nan"), float("inf")):
        with pytest.raises(SignalExtractionError):
            window_n_times(bad)


def test_non_integer_sample_count_fails_closed(record):
    data, assembly = two_trials(record)
    assembly["sfreq"] = 250.3
    with pytest.raises(SignalExtractionError, match="exact integer"):
        extract_trial_signals(data, assembly, record)


def test_existing_integer_anchor_is_used_not_a_float_reconstruction(record):
    data, assembly = two_trials(record)
    assembly["trials"][0]["anchor"]["source_sample_index"] = 1000.0
    with pytest.raises(SignalExtractionError, match="integer"):
        extract_trial_signals(data, assembly, record)


def test_start_at_sample_zero_is_valid(record):
    data, assembly = build(record, [("32766", 0), ("768", 0), ("769", 500)], 1500)
    out = extract_trial_signals(data, assembly, record)
    assert out["epochs"][0]["signal_start_source_sample"] == 0
    assert np.array_equal(out["signals"][0], data)  # also ends exactly at the source end


def test_exact_source_end_and_exact_span_end_pass(record):
    data, assembly = two_trials(record, second_anchor=2500, n_samples=4000)
    # trial 1 ends at 2500 == its span end (next anchor); trial 2 ends at 4000 == source end
    assert [t["span"]["end_sample_exclusive"] for t in assembly["trials"]] == [2500, 4000]
    out = extract_trial_signals(data, assembly, record)
    assert [e["signal_end_source_sample_exclusive"] for e in out["epochs"]] == [2500, 4000]


@pytest.mark.parametrize("edit,expect", [
    (lambda a: a["trials"][0]["anchor"].update(source_sample_index=-1), "start < 0"),
    (lambda a: a["trials"][1]["span"].update(end_sample_exclusive=3999), "trial span end"),
    (lambda a: a["trials"][1]["anchor"].update(source_sample_index=2501), "source end"),
])
def test_boundary_violations_fail_without_partial_output(record, edit, expect):
    data, assembly = two_trials(record)
    edit(assembly)
    before = data.copy()
    with pytest.raises(SignalExtractionError, match=expect):  # no padding, crop, shrink or drop
        extract_trial_signals(data, assembly, record)
    assert np.array_equal(data, before)


def test_window_beyond_trial_span_fails(record):
    # The next anchor starts 1499 samples later: [1000, 2500) would cross it.
    data, assembly = two_trials(record, second_anchor=2499, n_samples=4000)
    with pytest.raises(SignalExtractionError, match="trial span end"):
        extract_trial_signals(data, assembly, record)


def test_window_beyond_source_end_fails(record):
    data, assembly = build(record, [("32766", 0), ("768", 1000), ("769", 1500)], 2499)
    with pytest.raises(SignalExtractionError, match="source end"):
        extract_trial_signals(data, assembly, record)


def test_extracted_values_equal_source_slice_and_source_unchanged(record):
    data, assembly = two_trials(record)
    before = data.copy()
    out = extract_trial_signals(data, assembly, record)
    assert np.array_equal(data, before)
    for signal, epoch in zip(out["signals"], out["epochs"]):
        expected = data[:, epoch["signal_start_source_sample"]:epoch["signal_end_source_sample_exclusive"]]
        assert signal.shape == (25, N) and np.array_equal(signal, expected)
    assert out["signals"].dtype == data.dtype
    out["signals"][0, 0, 0] = -1.0  # extracted values are not aliases of the source
    assert np.array_equal(data, before)


def test_rejected_trial_is_extracted_unchanged_and_marked(record):
    data, assembly = build(record, [("32766", 0), ("768", 1000), ("1023", 1000), ("769", 1500),
                                    ("768", 2500), ("770", 3000)], 4000)
    out = extract_trial_signals(data, assembly, record)
    assert [e["rejection_marker_present"] for e in out["epochs"]] == [True, False]
    assert len(out["epochs"]) == 2 == out["signals"].shape[0]  # nothing skipped
    assert np.array_equal(out["signals"][0], data[:, 1000:2500])
    names = set(out["epochs"][0]) | set(out["collection"])
    assert not {n for n in names if "baseline" in n or "excluded" in n or "included" in n}  # no Baseline decision


def test_lineage_and_metadata_preserved(record):
    data, assembly = two_trials(record)
    out = extract_trial_signals(data, assembly, record)
    assert [e["trial_id"] for e in out["epochs"]] == [t["trial_id"] for t in assembly["trials"]]
    for epoch, trial in zip(out["epochs"], assembly["trials"]):
        assert epoch["trial_id"] == f"{record['source_sha256']}:{epoch['anchor_source_sample_index']}"
        assert epoch["source_artifact_id"] == record["artifact_id"]
        assert epoch["ground_truth_semantic_label"] == trial["semantic_label_id"]
        assert epoch["source_processing_status"] == record["processing_status"]
        assert (epoch["tmin"], epoch["tmax"], epoch["endpoint_convention"]) == (0.0, 6.0, "half_open")
        assert (epoch["sfreq"], epoch["n_times"]) == (250.0, 1500)
        assert not {k for k in epoch if "class" in k or "model" in k}
    assert [e["ground_truth_semantic_label"] for e in out["epochs"]] == ["left_hand", "right_hand"]


def test_collection_lineage_and_stage(record):
    data, assembly = two_trials(record)
    collection = extract_trial_signals(data, assembly, record)["collection"]
    assert collection["stage"] == "pre_channel_canonicalization"
    assert collection["channel_names"] == record["metadata"]["channel_names"]  # order kept
    assert collection["channel_types"] == record["metadata"]["channel_types"]
    assert collection["source_sha256"] == record["source_sha256"]
    assert collection["source_artifact_id"] == record["artifact_id"]
    assert collection["source_processing_status"] == record["processing_status"]
    assert collection["source_unit_representation"]["header_units"] == record["loader"]["header_audit"]["units"]
    assert (collection["n_trials"], collection["n_times"], collection["source_n_times"]) == (2, 1500, 4000)
    assert not {k for k in collection if "final" in k or "class" in k}  # not a final dataset


def test_inconsistent_inputs_fail(record):
    data, assembly = two_trials(record)
    with pytest.raises(SignalExtractionError, match="sample count"):
        extract_trial_signals(data[:, :-1], assembly, record)
    with pytest.raises(SignalExtractionError, match="channel count"):
        extract_trial_signals(data[:-1], assembly, record)
    with pytest.raises(SignalExtractionError, match="2-D"):
        extract_trial_signals(data[0], assembly, record)
    other = deepcopy(record)
    other["source_sha256"] = "0" * 64
    with pytest.raises(SignalExtractionError, match="same source"):
        extract_trial_signals(data, assembly, other)


def test_extraction_does_not_mutate_upstream_metadata(record):
    data, assembly = two_trials(record)
    before = deepcopy(assembly), deepcopy(record), data.copy()
    extract_trial_signals(data, assembly, record)
    assert (assembly, record) == before[:2] and np.array_equal(data, before[2])
    assert record["processing_status"] == before[1]["processing_status"]

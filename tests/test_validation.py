"""Unit tests for anr.validation (artificial data and Mock Reader; no actual GDF)."""

from copy import deepcopy
from hashlib import sha256
from unittest.mock import Mock

import numpy as np
import pytest

from anr.loader import MNE_VERSION, SourceReference, load_a01t
from anr.provenance import ProcessingStep
from anr.validation import validate_basic_integrity, validate_finite, validate_structural
from report_utils import write_report
from test_provenance import FakeRaw as ProvenanceFakeRaw


class FakeRaw(ProvenanceFakeRaw):
    def __init__(self):
        super().__init__()
        self.data = np.random.default_rng(0).normal(size=(25, 1000))

    def get_data(self):
        return self.data.copy()


def make(tmp_path, before=None, *, reference=True, history=()):
    """Load with a Mock Reader; `before` mutates the Raw before the snapshot."""
    path = tmp_path / "A01T.gdf"
    path.write_bytes(b"Gate A artificial bytes: this is NOT a GDF file")
    raw = FakeRaw()
    if before:
        before(raw)
    ref = SourceReference(sha256(path.read_bytes()).hexdigest(),
                          "https://example.invalid/fixture", "synthetic") if reference else None
    _, record = load_a01t(path, reader=Mock(return_value=raw), reader_version=MNE_VERSION,
                          source_reference=ref, processing_history=list(history))
    return raw, record


def failed(result):
    return {c["check_id"] for c in result["checks"] if c["status"] == "FAIL"}


def raise_access(*args):
    raise OSError("data unreadable")


def test_normal_2d_passes(tmp_path):
    raw, record = make(tmp_path)
    result = validate_structural(raw, record)
    assert result["status"] == "PASS" and failed(result) == set()
    assert [c["check_id"] for c in result["checks"]] == [
        "DATA_ACCESS", "DATA_DIMENSION", "NON_EMPTY_DIMENSIONS", "CHANNEL_SHAPE_CONSISTENCY",
        "SAMPLE_SHAPE_CONSISTENCY", "SAMPLING_INFORMATION_VALIDITY", "CHANNEL_METADATA_CONSISTENCY"]
    assert result["validator_name"] == "structural_integrity" and result["validator_version"]
    assert result["artifact_id"] == record["artifact_id"]
    assert result["source_sha256"] == record["source_sha256"]
    assert all(set(c) == {"check_id", "status", "criterion", "observed", "reason"}
               for c in result["checks"])


def test_data_access_failure(tmp_path):
    raw, record = make(tmp_path)
    raw.get_data = raise_access
    result = validate_structural(raw, record)
    assert result["status"] == "FAIL" and "DATA_ACCESS" in failed(result)
    assert "OSError" in result["checks"][0]["reason"]


@pytest.mark.parametrize("shape", [(25000,), (1, 25, 1000)])
def test_wrong_dimension(tmp_path, shape):
    raw, record = make(tmp_path)
    raw.data = np.zeros(shape)
    assert "DATA_DIMENSION" in failed(validate_structural(raw, record))


def test_zero_channels(tmp_path):
    raw, record = make(tmp_path)
    raw.data = np.zeros((0, 1000))
    result = validate_structural(raw, record)
    assert "NON_EMPTY_DIMENSIONS" in failed(result) and result["status"] == "FAIL"


def test_zero_samples_consistent_with_n_times(tmp_path):
    def empty(raw):
        raw.data, raw.n_times = np.zeros((25, 0)), 0
    raw, record = make(tmp_path, empty)  # the loader no longer rejects this
    result = validate_structural(raw, record)
    assert failed(result) == {"NON_EMPTY_DIMENSIONS"}


def test_channel_shape_mismatch(tmp_path):
    raw, record = make(tmp_path)
    raw.data = np.zeros((24, 1000))
    assert "CHANNEL_SHAPE_CONSISTENCY" in failed(validate_structural(raw, record))


def test_sample_shape_mismatch(tmp_path):
    raw, record = make(tmp_path)
    raw.data = np.zeros((25, 999))
    assert "SAMPLE_SHAPE_CONSISTENCY" in failed(validate_structural(raw, record))


@pytest.mark.parametrize("edit,check_id", [
    (lambda m: m.update(n_channels=24), "CHANNEL_SHAPE_CONSISTENCY"),
    (lambda m: m.update(n_times=999), "SAMPLE_SHAPE_CONSISTENCY"),
    (lambda m: m.update(sfreq=200.0), "SAMPLING_INFORMATION_VALIDITY"),
    (lambda m: m["channel_names"].reverse(), "CHANNEL_METADATA_CONSISTENCY"),
    (lambda m: m["channel_types"].reverse(), "CHANNEL_METADATA_CONSISTENCY"),
])
def test_snapshot_differs_from_current_object(tmp_path, edit, check_id):
    raw, record = make(tmp_path)
    edit(record["metadata"])
    result = validate_structural(raw, record)
    assert check_id in failed(result) and result["status"] == "FAIL"


@pytest.mark.parametrize("sfreq", [None, float("nan"), float("inf"), -float("inf"), 0.0, -250.0, "250"])
def test_invalid_sampling_information(tmp_path, sfreq):
    raw, record = make(tmp_path, lambda r: r.info.update(sfreq=sfreq))
    result = validate_structural(raw, record)
    assert failed(result) == {"SAMPLING_INFORMATION_VALIDITY"}


def test_empty_channel_name(tmp_path):
    def rename(raw):
        raw.ch_names[3] = "  "
    raw, record = make(tmp_path, rename)
    assert failed(validate_structural(raw, record)) == {"CHANNEL_METADATA_CONSISTENCY"}


def test_duplicate_channel_name(tmp_path):
    def rename(raw):
        raw.ch_names[4] = raw.ch_names[3]
    raw, record = make(tmp_path, rename)
    result = validate_structural(raw, record)
    assert failed(result) == {"CHANNEL_METADATA_CONSISTENCY"}
    assert result["checks"][-1]["observed"]["duplicate_names"] == ["channel-3"]


def test_finite_data_passes(tmp_path):
    raw, record = make(tmp_path)
    result = validate_finite(raw, record)
    assert result["status"] == "PASS"
    assert result["checks"][0]["observed"] == {
        "total_values": 25000, "nan_count": 0, "positive_inf_count": 0, "negative_inf_count": 0}


@pytest.mark.parametrize("value,key", [(np.nan, "nan_count"), (np.inf, "positive_inf_count"),
                                       (-np.inf, "negative_inf_count")])
def test_non_finite_values_fail(tmp_path, value, key):
    raw, record = make(tmp_path)
    raw.data[2, 5:8] = value
    result = validate_finite(raw, record)
    assert result["status"] == "FAIL" and result["checks"][0]["observed"][key] == 3
    # Structural PASS and Finite FAIL coexist; Basic Integrity needs both.
    combined = validate_basic_integrity(raw, record)
    assert (combined["structural"]["status"], combined["finite"]["status"],
            combined["basic_integrity"]["status"]) == ("PASS", "FAIL", "FAIL")


def test_finite_data_access_failure(tmp_path):
    raw, record = make(tmp_path)
    raw.get_data = raise_access
    assert validate_finite(raw, record)["status"] == "FAIL"


@pytest.mark.parametrize("reference,history,status", [
    (True, (), "RAW"), (False, (), "UNKNOWN"),
    (True, (ProcessingStep("post_distribution", "prior filter", "signal_processing", True, "log"),),
     "PROCESSED")])
@pytest.mark.parametrize("corrupt", [False, True])
def test_processing_status_is_independent_of_validation(tmp_path, reference, history, status, corrupt):
    before = (lambda raw: raw.info.update(sfreq=None)) if corrupt else None
    raw, record = make(tmp_path, before, reference=reference, history=history)
    raw.data[0, 0] = np.nan
    original, data = deepcopy(record), raw.data.copy()
    results = validate_basic_integrity(raw, record)
    assert record == original and record["processing_status"] == status
    assert results["structural"]["status"] == ("FAIL" if corrupt else "PASS")
    assert results["basic_integrity"]["status"] == "FAIL"  # NaN is always present here
    assert np.array_equal(raw.data, data, equal_nan=True)  # Raw data untouched


def test_basic_integrity_pass_requires_both(tmp_path):
    raw, record = make(tmp_path)
    results = validate_basic_integrity(raw, record)
    assert results["basic_integrity"]["status"] == "PASS"
    assert [c["check_id"] for c in results["basic_integrity"]["checks"]] == ["STRUCTURAL", "FINITE"]


def test_report_utility_writes_bilingual_txt(tmp_path):
    raw, record = make(tmp_path)
    results = validate_basic_integrity(raw, record)
    path = write_report(tmp_path / "out" / "report.txt", results=results, record=record,
                        dataset_file="A01T.gdf", environment={"mne": "x", "pytest": "y"},
                        commit=("0" * 40, False))
    text = path.read_text(encoding="utf-8")
    assert "신호 데이터를" in text and "Signal data can be read" in text
    assert record["artifact_id"] in text and "Basic Integrity result" in text

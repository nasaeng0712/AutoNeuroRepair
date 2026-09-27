"""Gate A: artificial bytes and Mock Reader, never actual GDF evidence."""

from hashlib import sha256
import json
from unittest.mock import Mock, patch
import warnings

import pytest

from anr.loader import MNE_VERSION, SourceReference, load_a01t
from anr.provenance import Evidence, ProcessingStep, assess, read_record, save_record


class FakeRaw:
    def __init__(self):
        self.info = {"sfreq": 250.0, "highpass": 0.0, "lowpass": 125.0}
        self.ch_names = [f"channel-{i}" for i in range(25)]
        self.n_times = 1000
        self.annotations = ["768"]
        self.preload = True
        self.closed = False
        self._raw_extras = [{
            "sel": list(range(25)), "n_samps": [250] * 25, "max_samp": 250,
            "cal": [0.1] * 25, "offsets": [0.0] * 25, "units": [1e-6] * 25,
        }]

    def get_channel_types(self):
        return ["eeg"] * 22 + ["eog"] * 3

    def close(self):
        self.closed = True


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "A01T.gdf"
    path.write_bytes(b"Gate A artificial bytes: this is NOT a GDF file")
    reference = SourceReference(sha256(path.read_bytes()).hexdigest(),
                                "https://example.invalid/gate-a-fixture",
                                "Synthetic unit-test reference; NOT official acquisition")
    raw = FakeRaw()
    reader = Mock(return_value=raw)
    return path, reference, raw, reader


def load_sample(sample, **kwargs):
    path, reference, _, reader = sample
    kwargs.setdefault("source_reference", reference)
    kwargs.setdefault("reader_version", MNE_VERSION)
    return load_a01t(path, reader=reader, **kwargs)


def test_raw_records_separate_acquisition_and_representation(sample):
    raw, record = load_sample(sample)
    assert raw is sample[2]
    assert record["processing_status"] == "RAW"
    assert record["evidence_level"] == "DIRECT"
    assert record["validation_scope"] == "mock_reader"
    assert record["loader"]["mne_version"] is None
    assert record["loader"]["reader_contract_version"] == MNE_VERSION
    assert record["source_sha256"] == sample[1].sha256
    assert record["source"]["artifact_id"] == record["artifact_id"]
    assert [h["value"] for h in record["acquisition_history"]] == [0.5, 100.0, 50.0]
    assert all(h["category"] == "representation" for h in record["processing_history"])
    assert record["loader"]["header_audit"]["units"] == [1e-6] * 25
    assert record["review_required"] is False
    # Returned filter defaults differ from documentation, but are NOT evidence
    # of a conflicting acquisition filter when the original header is absent.
    assert record["conflicts"] == []
    sample[3].assert_called_once_with(str(sample[0].resolve()),
                                    **record["loader"]["options"])
    assert record["loader"]["options"]["eog"] == [22, 23, 24]
    assert "does not preserve" in record["loader"]["preload_meaning"]


def test_missing_reference_is_unknown_not_conflict(sample):
    _, record = load_sample(sample, source_reference=None)
    assert record["processing_status"] == "UNKNOWN"
    assert record["evidence_level"] == "INSUFFICIENT"
    assert record["conflicts"] == []
    assert record["review_required"]


def test_wrong_reference_is_explicit_conflict(sample):
    reference = SourceReference("0" * 64, "https://example.invalid/reference", "test only")
    _, record = load_sample(sample, source_reference=reference)
    assert record["processing_status"] == "UNKNOWN"
    assert any(c["item"] == "sha256" for c in record["conflicts"])


@pytest.mark.parametrize("key,value", [("highpass", 1.0), ("lowpass", 40.0), ("notch", 60.0)])
def test_explicit_acquisition_conflict(sample, key, value):
    sample[2]._raw_extras[0][key] = [value] * 25
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"][0]["stage"] == "acquisition"
    assert record["review_required"]


@pytest.mark.parametrize("value", [None, float("nan")])
def test_missing_filter_header_does_not_invent_conflict(sample, value):
    sample[2]._raw_extras[0]["notch"] = [value] * 25
    _, record = load_sample(sample)
    assert record["conflicts"] == []
    assert record["processing_status"] == "RAW"


def test_ambiguous_zero_filter_is_reviewed_not_assumed_missing(sample):
    sample[2]._raw_extras[0]["notch"] = [0.0] * 25
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"] == []
    assert record["loader"]["header_audit"]["unresolved_filter_encoding"]


def test_implicit_resampling_is_processed_even_with_conflict(sample):
    sample[2]._raw_extras[0]["n_samps"][0] = 125
    denial = Evidence("loader", "additional_signal_processing", False, "test denial", "reported")
    _, record = load_sample(sample, evidence=[denial])
    assert record["processing_status"] == "PROCESSED"
    assert record["evidence_level"] == "DIRECT"
    assert record["conflicts"] and record["review_required"]
    assert any(h["category"] == "signal_processing" and h["confirmed"]
               for h in record["processing_history"])


def test_recorded_prior_processing_is_preserved_not_executed(sample):
    step = ProcessingStep("post_distribution", "Prior bandpass from processing log",
                          "signal_processing", True, "test upstream log")
    _, record = load_sample(sample, source_reference=None, processing_history=[step])
    assert record["processing_status"] == "PROCESSED"
    assert record["processing_history"][0]["operation"] == step.operation
    assert record["unknowns"]  # Source identity still unknown despite positive processing evidence.


def test_unconfirmed_operation_is_unknown(sample):
    step = ProcessingStep("post_distribution", "Unverified reported filter",
                          "signal_processing", False, "test report")
    _, record = load_sample(sample, processing_history=[step])
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"] == []


@pytest.mark.parametrize("version", ["1.10.0", "future-version"])
def test_unreviewed_reader_version_is_unknown(sample, version):
    _, record = load_sample(sample, reader_version=version)
    assert record["processing_status"] == "UNKNOWN"
    assert record["loader"]["reader_contract_version"] == version


def test_missing_private_metadata_is_unknown(sample):
    del sample[2]._raw_extras
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"] == []


def test_lazy_loading_records_deferred_transformations(sample):
    sample[2].preload = False
    _, record = load_sample(sample, preload=False)
    assert record["processing_status"] == "UNKNOWN"
    assert record["loader"]["options"]["preload"] is False
    assert any(not s["confirmed"] for s in record["processing_history"])


def test_warning_recorded_without_fabricating_conflict(sample):
    def warning_reader(*args, **kwargs):
        warnings.warn("Synthetic ambiguous GDF header", RuntimeWarning)
        return sample[2]
    sample[3].side_effect = warning_reader
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"] == []
    assert record["loader"]["warnings"] == ["Synthetic ambiguous GDF header"]


def test_returned_sample_rate_conflict(sample):
    sample[2].info["sfreq"] = 200.0
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert any(c["item"] == "sfreq" for c in record["conflicts"])


def test_missing_sample_rate_is_unknown_not_conflict(sample):
    sample[2].info["sfreq"] = None
    _, record = load_sample(sample)
    assert record["processing_status"] == "UNKNOWN"
    assert record["conflicts"] == []


def test_source_changed_during_read_fails_and_closes_raw(sample):
    def mutate_source(*args, **kwargs):
        sample[0].write_bytes(b"changed during loading")
        return sample[2]
    sample[3].side_effect = mutate_source
    with pytest.raises(ValueError, match="changed during loading"):
        load_sample(sample)
    assert sample[2].closed


def test_reader_error_is_not_reported_as_success(sample):
    sample[3].side_effect = ValueError("Invalid GDF")
    with pytest.raises(ValueError, match="Invalid GDF"):
        load_sample(sample)


def test_no_samples_is_error_and_closes_raw(sample):
    sample[2].n_times = 0
    with pytest.raises(ValueError, match="no samples"):
        load_sample(sample)
    assert sample[2].closed


def test_missing_file_does_not_call_reader(tmp_path):
    reader = Mock()
    with pytest.raises(FileNotFoundError):
        load_a01t(tmp_path / "A01T.gdf", reader=reader, reader_version=MNE_VERSION)
    reader.assert_not_called()


def test_wrong_filename_rejected(tmp_path):
    with pytest.raises(ValueError, match="only A01T"):
        load_a01t(tmp_path / "A02T.gdf")


def test_memory_map_path_rejected(sample):
    with pytest.raises(ValueError, match="boolean"):
        load_sample(sample, preload="A01T.gdf")


def test_cannot_inject_reserved_positive_evidence(sample):
    with pytest.raises(ValueError, match="Reserved"):
        load_sample(sample, evidence=[Evidence("loader", "audit_complete", True, "fake", "direct")])


def test_json_roundtrip_and_relocated_source(sample, tmp_path):
    _, record = load_sample(sample)
    target = tmp_path / "provenance.json"
    save_record(record, target)
    assert read_record(target, source_path=sample[0]) == record
    relocated = tmp_path / "copy.gdf"
    relocated.write_bytes(sample[0].read_bytes())
    assert read_record(target, source_path=relocated) == record
    relocated.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        read_record(target, source_path=relocated)
    # JSON-only reload does not claim a live source-file verification.
    assert read_record(target) == record


def test_distinct_loads_share_source_hash_not_artifact_id(sample):
    _, first = load_sample(sample)
    _, second = load_sample(sample)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["artifact_id"] != second["artifact_id"]


@pytest.mark.parametrize("field,value", [("processing_status", "PROCESSED"),
                                        ("evidence_level", "made up"),
                                        ("source_sha256", "0" * 64),
                                        ("artifact_id", "not-a-uuid")])
def test_inconsistent_json_rejected(sample, tmp_path, field, value):
    _, record = load_sample(sample)
    record[field] = value
    target = tmp_path / "invalid.json"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        read_record(target)


def test_json_write_does_not_overwrite_source_or_record(sample, tmp_path):
    _, record = load_sample(sample)
    original = sample[0].read_bytes()
    with pytest.raises(ValueError, match="source GDF"):
        save_record(record, sample[0])
    assert sample[0].read_bytes() == original
    target = tmp_path / "record.json"
    save_record(record, target)
    with pytest.raises(FileExistsError):
        save_record(record, target)


def test_tampered_reference_rejected(sample, tmp_path):
    _, record = load_sample(sample)
    record["source"]["reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reference and match"):
        save_record(record, tmp_path / "bad-reference.json")


def test_mock_cannot_claim_actual_mne_version(sample, tmp_path):
    _, record = load_sample(sample)
    record["loader"]["mne_version"] = MNE_VERSION
    with pytest.raises(ValueError, match="Mock"):
        save_record(record, tmp_path / "bad-scope.json")


def test_missing_output_directory_raises_file_error(sample, tmp_path):
    _, record = load_sample(sample)
    with pytest.raises(FileNotFoundError):
        save_record(record, tmp_path / "absent" / "record.json")


def test_output_permission_error_propagates(sample, tmp_path):
    _, record = load_sample(sample)
    with patch("anr.provenance.Path.open", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError, match="denied"):
            save_record(record, tmp_path / "record.json")


def test_source_permission_error_propagates_without_calling_reader(sample):
    with patch("anr.provenance.Path.open", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError, match="denied"):
            load_sample(sample)
    sample[3].assert_not_called()


@pytest.mark.parametrize("content", ["{", "null", "[]", "{}", '{"schema_version": NaN}'])
def test_malformed_json_rejected(tmp_path, content):
    target = tmp_path / "bad.json"
    target.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_record(target)


def test_missing_json_raises_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_record(tmp_path / "missing.json")


def test_only_same_stage_and_item_conflict():
    evidence = [Evidence("acquisition", "filter_hz", 50, "doc", "document"),
                Evidence("acquisition", "filter_hz", 50.0, "header", "metadata"),
                Evidence("loader", "filter_hz", 40.0, "loader", "metadata"),
                Evidence("acquisition", "filter_hz", None, "absent", "metadata")]
    assert assess(evidence, [])["conflicts"] == []
    evidence.append(Evidence("acquisition", "filter_hz", 60, "other header", "metadata"))
    assert len(assess(evidence, [])["conflicts"]) == 1


def test_direct_processing_claim_wins_but_retains_conflict():
    evidence = [Evidence("loader", "additional_signal_processing", True, "observed", "direct"),
                Evidence("loader", "additional_signal_processing", False, "claim", "reported")]
    result = assess(evidence, [])
    assert result["processing_status"] == "PROCESSED"
    assert result["conflicts"] and result["review_required"]


@pytest.mark.parametrize("kind", ["document", "metadata", "reported"])
def test_indirect_processing_claim_does_not_become_direct(kind):
    evidence = [Evidence("source", "official_reference_match", True, "fixture", "direct"),
                Evidence("loader", "audit_complete", True, "fixture", "direct"),
                Evidence("post_distribution", "additional_signal_processing", True, "claim", kind)]
    assert assess(evidence, [])["processing_status"] == "UNKNOWN"


def test_empty_evidence_never_proves_raw():
    assert assess([], [])["processing_status"] == "UNKNOWN"


def test_invalid_history_and_evidence_rejected():
    with pytest.raises(ValueError):
        ProcessingStep("acquisition", "filter", "signal_processing", True, "doc")
    with pytest.raises(ValueError):
        Evidence("loader", "x", float("nan"), "test", "direct")
    with pytest.raises(ValueError):
        SourceReference("invalid", "https://example.invalid", "test")

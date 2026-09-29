"""Gate B: opt-in actual GDF check. A skip means BLOCKED, not a Gate B PASS.

Gate B verifies software behaviour on the actual file. It does not require a
particular processing_status: RAW, PROCESSED and UNKNOWN are all valid results
when they follow consistently from the recorded evidence.
"""

from collections import Counter
import json
import os
from pathlib import Path

import pytest

from anr.loader import SourceReference, load_a01t
from anr.provenance import (
    Evidence, ProcessingStep, assess, file_sha256, read_record, save_record,
)


@pytest.mark.integration
def test_official_a01t_file(tmp_path):
    if not os.environ.get("ANR_A01T_PATH"):
        pytest.skip("Gate B BLOCKED: actual A01T.gdf unavailable (ANR_A01T_PATH)")
    # A trusted acquisition record is optional; without it the source identity
    # stays unestablished and the specified result is UNKNOWN.
    names = ("ANR_A01T_REFERENCE_SHA256", "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    given = [name for name in names if os.environ.get(name)]
    if given and len(given) != len(names):
        pytest.skip("Gate B BLOCKED: incomplete trusted acquisition record: "
                    + ", ".join(sorted(set(names) - set(given))))
    reference = SourceReference(*(os.environ[name] for name in names)) if given else None
    path = Path(os.environ["ANR_A01T_PATH"])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"
    raw, record = load_a01t(path, source_reference=reference)
    try:
        assert record["validation_scope"] == "actual_gdf"
        assert record["loader"]["name"] == "mne.io.read_raw_gdf"
        assert record["loader"]["mne_version"] is not None
        assert raw.info["sfreq"] == 250.0
        assert len(raw.ch_names) == 25
        assert raw.get_channel_types() == ["eeg"] * 22 + ["eog"] * 3
        counts = Counter(raw.annotations.description)
        assert counts["768"] == 288  # Official document: six runs, 48 trials each.
        assert all(counts[str(code)] == 72 for code in (769, 770, 771, 772))
        # Source identification and linkage.
        digest = file_sha256(path)
        assert record["source_sha256"] == record["source"]["sha256"] == digest
        assert record["source"]["artifact_id"] == record["artifact_id"]
        match = [c for c in record["evidence"]
                 if c["stage"] == "source" and c["item"] == "official_reference_match"]
        assert [c["value"] for c in match] == [None if reference is None
                                               else digest == reference.sha256]
        # Loader options, acquisition and processing history are recorded.
        assert record["loader"]["options"] == {
            "eog": [22, 23, 24], "misc": None, "stim_channel": None, "exclude": [],
            "include": None, "preload": True, "verbose": "ERROR"}
        assert len(record["acquisition_history"]) == 3
        assert all(s["stage"] in {"loader", "post_distribution"}
                   for s in record["processing_history"])
        # The decision is recomputed from the persisted evidence, whatever it is.
        expected = assess([Evidence(**c) for c in record["evidence"]],
                          [ProcessingStep(**s) for s in record["processing_history"]])
        assert {key: record[key] for key in expected} == expected
        assert record["processing_status"] in {"RAW", "PROCESSED", "UNKNOWN"}
        target = tmp_path / "actual-a01t-provenance.json"
        save_record(record, target)
        assert read_record(target, source_path=path) == record
        # A01T Assessment is reported separately from the Gate B verdict.
        print("A01T Assessment", json.dumps(
            {key: record[key] for key in ("processing_status", "evidence_level",
                                          "review_required", "conflicts", "unknowns")},
            ensure_ascii=False))
    finally:
        raw.close()

"""Gate B: opt-in actual GDF check. A skip means BLOCKED, not a Gate B PASS."""

from collections import Counter
import os
from pathlib import Path

import pytest

from anr.loader import SourceReference, load_a01t
from anr.provenance import file_sha256, read_record, save_record


@pytest.mark.integration
def test_official_a01t_file(tmp_path):
    names = ("ANR_A01T_PATH", "ANR_A01T_REFERENCE_SHA256",
             "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip("Gate B BLOCKED: actual file/trusted acquisition record required: " + ", ".join(missing))
    path = Path(os.environ[names[0]])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"
    reference = SourceReference(*(os.environ[name] for name in names[1:]))
    raw, record = load_a01t(path, source_reference=reference)
    try:
        assert record["validation_scope"] == "actual_gdf"
        assert record["loader"]["mne_version"] is not None
        assert raw.info["sfreq"] == 250.0
        assert len(raw.ch_names) == 25
        assert raw.get_channel_types() == ["eeg"] * 22 + ["eog"] * 3
        counts = Counter(raw.annotations.description)
        assert counts["768"] == 288  # Official document: six runs, 48 trials each.
        assert all(counts[str(code)] == 72 for code in (769, 770, 771, 772))
        assert record["processing_status"] == "RAW", record
        assert record["review_required"] is False, record
        assert record["source_sha256"] == file_sha256(path) == reference.sha256
        assert len(record["acquisition_history"]) == 3
        assert all(s["category"] == "representation" and s["confirmed"]
                   for s in record["processing_history"])
        target = tmp_path / "actual-a01t-provenance.json"
        save_record(record, target)
        assert read_record(target, source_path=path) == record
    finally:
        raw.close()

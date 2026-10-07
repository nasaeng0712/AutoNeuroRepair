"""Actual-A01T Event Semantic run. A skip means BLOCKED, not a PASS.

Expected counts below are test evidence from the official dataset description
and a prior observation; they are never stored in the normative schema. A
mismatch must fail this test, not be corrected.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from anr.event_semantics import (
    interpret_events, observe_coverage, require_semantic_complete, task_label_schema,
)
from anr.loader import SourceReference, load_a01t
from anr.validation import validate_basic_integrity

EXPECTED_COUNTS = {"276": 1, "277": 1, "768": 288, "769": 72, "770": 72, "771": 72,
                   "772": 72, "783": 0, "1023": 15, "1072": 1, "32766": 9}


@pytest.mark.integration
def test_actual_a01t_event_semantics():
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
        record_before, integrity_before = deepcopy(record), deepcopy(integrity)
        data_before = sha256(np.ascontiguousarray(raw.get_data())).hexdigest()
        annotations_before = [(float(a["onset"]), float(a["duration"]), str(a["description"]))
                              for a in raw.annotations]

        descriptions = [str(d) for d in raw.annotations.description]  # decoded by the loader
        coverage = observe_coverage(descriptions)
        records = interpret_events(descriptions)

        assert coverage["total_events"] == len(descriptions)
        assert coverage["native_event_counts"] == EXPECTED_COUNTS, coverage["native_event_counts"]
        assert coverage["task_label_counts"] == {"left_hand": 72, "right_hand": 72,
                                                 "both_feet": 72, "tongue": 72}
        assert coverage["undefined_native_codes"] == []  # no undefined event observed
        assert len(records) == len(descriptions)
        assert not any(r["event_role"] == "unknown_native_event" for r in records)
        assert require_semantic_complete(records) is None
        run_starts = [r for r in records if r["native_event_code"] == "32766"]
        assert len(run_starts) == 9
        assert all((r["event_role"], r["task_id"], r["semantic_label_id"])
                   == ("run_start", "motor_imagery", None) for r in run_starts)
        assert sum(r["event_role"] == "trial_rejection_marker" for r in records) == 15
        assert sum(r["event_role"] == "auxiliary_recording_marker" for r in records) == 3
        assert not any("class" in k or "index" in k for r in records for k in r)
        assert set(task_label_schema()["labels"]) == {"left_hand", "right_hand", "both_feet", "tongue"}

        # Event Semantic changed nothing upstream.
        assert record == record_before
        assert validate_basic_integrity(raw, record) == integrity_before == integrity
        assert sha256(np.ascontiguousarray(raw.get_data())).hexdigest() == data_before
        assert [(float(a["onset"]), float(a["duration"]), str(a["description"]))
                for a in raw.annotations] == annotations_before
        assert record["source_sha256"] == sha256(path.read_bytes()).hexdigest()
        print("A01T native_event_counts", coverage["native_event_counts"],
              "processing_status", record["processing_status"],
              "basic_integrity", integrity["basic_integrity"]["status"])
    finally:
        raw.close()

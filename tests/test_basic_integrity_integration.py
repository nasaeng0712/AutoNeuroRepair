"""Actual-A01T basic integrity run. A skip means BLOCKED, not a PASS.

Software verification (the pipeline runs and records results consistently) is
separate from the A01T data assessment: Structural/Finite FAIL on the real
file is a valid, reported outcome and does not fail this test.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from anr.loader import SourceReference, load_a01t
from anr.provenance import Evidence, ProcessingStep, assess
from anr.validation import validate_basic_integrity
from report_utils import git_state, write_report

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "tests" / "output" / "structural_integrity_test_report.txt"


@pytest.mark.integration
def test_actual_a01t_basic_integrity():
    if not os.environ.get("ANR_A01T_PATH"):
        pytest.skip("BLOCKED: actual A01T.gdf unavailable (ANR_A01T_PATH)")
    names = ("ANR_A01T_REFERENCE_SHA256", "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    given = [name for name in names if os.environ.get(name)]
    if given and len(given) != len(names):
        pytest.skip("BLOCKED: incomplete trusted acquisition record")
    reference = SourceReference(*(os.environ[n] for n in names)) if given else None
    path = Path(os.environ["ANR_A01T_PATH"])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"

    raw, record = load_a01t(path, source_reference=reference)  # loader -> provenance
    try:
        assert record["validation_scope"] == "actual_gdf"
        before_record = deepcopy(record)
        before_data = sha256(np.ascontiguousarray(raw.get_data())).hexdigest()
        results = validate_basic_integrity(raw, record)  # Structural -> Finite -> Basic

        # Validation is read-only: provenance, status and signal are unchanged.
        assert record == before_record
        assert sha256(np.ascontiguousarray(raw.get_data())).hexdigest() == before_data
        expected = assess([Evidence(**c) for c in record["evidence"]],
                          [ProcessingStep(**s) for s in record["processing_history"]])
        assert record["processing_status"] == expected["processing_status"]

        # Results are well-formed and linked to this artifact/source.
        for result in results.values():
            assert result["artifact_id"] == record["artifact_id"]
            assert result["source_sha256"] == record["source_sha256"] == sha256(path.read_bytes()).hexdigest()
            assert result["status"] in {"PASS", "FAIL"} and result["checks"]
            assert all(c["status"] in {"PASS", "FAIL"} for c in result["checks"])
            assert all((c["status"] == "FAIL") == bool(c["reason"]) for c in result["checks"])
        assert results["basic_integrity"]["status"] == (
            "PASS" if results["structural"]["status"] == results["finite"]["status"] == "PASS"
            else "FAIL")

        import mne
        write_report(REPORT, results=results, record=record, dataset_file=str(path),
                     environment={"mne": mne.__version__, "pytest": pytest.__version__},
                     commit=git_state(REPO))
        assert REPORT.is_file()
        print("A01T Assessment", {k: v["status"] for k, v in results.items()},
              "processing_status(context)=", record["processing_status"])
    finally:
        raw.close()

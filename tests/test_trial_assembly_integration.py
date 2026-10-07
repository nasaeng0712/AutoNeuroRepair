"""Actual-A01T Trial Assembly run. A skip means BLOCKED, not a PASS.

The 288 / 6 x 48 / 12-per-class criteria are the normative A01T structure
(validate_a01t_trial_structure). The observed 1023 count is evidence only and
is not asserted as a constant. A mismatch must fail, never be corrected.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest

from anr.event_semantics import interpret_events, require_semantic_complete
from anr.loader import SourceReference, load_a01t
from anr.trial_assembly import assemble_from_raw, validate_a01t_trial_structure
from anr.validation import validate_basic_integrity

CLASSES = ("left_hand", "right_hand", "both_feet", "tongue")


def snapshot_annotations(raw):
    a = raw.annotations
    return ([float(x) for x in a.onset], [float(x) for x in a.duration],
            [str(x) for x in a.description], a.orig_time)


@pytest.mark.integration
def test_actual_a01t_trial_assembly():
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
        before = {"record": deepcopy(record), "integrity": deepcopy(integrity),
                  "data": sha256(np.ascontiguousarray(raw.get_data())).hexdigest(),
                  "annotations": snapshot_annotations(raw),
                  "semantic": interpret_events([str(d) for d in raw.annotations.description])}
        semantic_before = deepcopy(before["semantic"])
        require_semantic_complete(before["semantic"])  # no unknown native event

        assembly = assemble_from_raw(raw, record)
        verdict = validate_a01t_trial_structure(assembly, record)
        failed = [c for c in verdict["checks"] if c["status"] == "FAIL"]
        assert verdict["status"] == "PASS", failed

        trials, evidence = assembly["trials"], assembly["evidence"]
        runs = assembly["task_bearing_runs"]
        by_id = {t["trial_id"]: t for t in trials}
        assert len(trials) == len(by_id) == 288
        assert all(t["trial_id"] == f"{record['source_sha256']}:{t['anchor']['source_sample_index']}"
                   for t in trials)
        assert all(t["source_artifact_id"] == record["artifact_id"] for t in trials)
        assert {t["cue_delta_samples"] for t in trials} == {500}
        assert evidence["associated_rejection_markers"] == evidence["observed_rejection_markers"]
        assert sum(len(t["rejection_markers"]) for t in trials) == evidence["observed_rejection_markers"]
        assert not any("class_index" in k for t in trials for k in t)

        # Assembly is metadata only: upstream results are unchanged.
        assert record == before["record"] and record["processing_status"] == before["record"]["processing_status"]
        assert validate_basic_integrity(raw, record) == before["integrity"]
        assert sha256(np.ascontiguousarray(raw.get_data())).hexdigest() == before["data"]
        assert snapshot_annotations(raw) == before["annotations"]
        assert interpret_events([str(d) for d in raw.annotations.description]) == semantic_before
        assert record["source_sha256"] == sha256(path.read_bytes()).hexdigest()

        per_run = [{c: sum(by_id[i]["semantic_label_id"] == c for i in r["trial_ids"]) for c in CLASSES}
                   for r in runs]
        print("TRIAL_ASSEMBLY_EVIDENCE", {
            "anchors": evidence["observed_anchors"], "associated_labels": evidence["associated_task_labels"],
            "overall": {c: sum(t["semantic_label_id"] == c for t in trials) for c in CLASSES},
            "task_bearing_runs": len(runs), "per_run_trials": [len(r["trial_ids"]) for r in runs],
            "per_run_classes": per_run, "cue_deltas": sorted({t["cue_delta_samples"] for t in trials}),
            "unique_trial_ids": len(by_id), "observed_1023": evidence["observed_rejection_markers"],
            "associated_1023": evidence["associated_rejection_markers"],
            "trials_with_rejection_marker": sum(t["rejection_marker_present"] for t in trials),
            "run_segment_indices": [r["run_segment_index"] for r in runs],
            "processing_status": record["processing_status"],
            "basic_integrity": integrity["basic_integrity"]["status"],
            "first_trial_id": trials[0]["trial_id"]})
    finally:
        raw.close()

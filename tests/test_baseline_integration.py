"""Actual-A01T Baseline v1 Software Verification. A skip means BLOCKED, not a PASS.

Software invariants only: functional metrics are printed as evidence but are NOT
asserted here (qualification is judged by anr.baseline_run, never tuned).
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from anr import baseline as bl
from anr.baseline_run import model_artifact_id
from anr.loader import SourceReference
from anr.pipeline import run_pipeline


@pytest.mark.integration
def test_actual_a01t_baseline_software_verification(tmp_path):
    if not os.environ.get("ANR_A01T_PATH"):
        pytest.skip("BLOCKED: actual A01T.gdf unavailable (ANR_A01T_PATH)")
    names = ("ANR_A01T_REFERENCE_SHA256", "ANR_A01T_REFERENCE_URL", "ANR_A01T_REFERENCE_NOTE")
    given = [name for name in names if os.environ.get(name)]
    if given and len(given) != len(names):
        pytest.skip("BLOCKED: incomplete trusted acquisition record")
    reference = SourceReference(*(os.environ[n] for n in names)) if given else None
    path = Path(os.environ["ANR_A01T_PATH"])
    assert path.is_file(), "Configured actual A01T.gdf does not exist"

    result = run_pipeline(path, source_reference=reference)
    dataset, record = result["dataset"], result["record"]
    sfreq, ids = dataset["dataset_metadata"]["sfreq"], [m["trial_id"] for m in dataset["trial_metadata"]]
    artifact = model_artifact_id(record["source_sha256"])
    manifest = bl.build_split_manifest(dataset)  # raises BLOCKED if a clean partition lacks a class

    assert len(manifest) == 288 == dataset["X_eeg"].shape[0]  # the dataset keeps every trial
    excluded = [r for r in manifest if not r["model_eligible"]]
    assert all(r["rejection_marker_present"] and r["exclusion_reason"] for r in excluded)
    assert len(excluded) == sum(m["rejection_marker_present"] for m in dataset["trial_metadata"])
    counts = {p: {label: int(sum(dataset["y_semantic"][i] == label for i in bl.clean_indices(manifest, p)))
                  for label in [v["semantic_label"] for v in bl.FROZEN_MODEL_VOCABULARY]}
              for p in ("TRAIN", "VALIDATION", "TEST")}
    parts = {p: {ids[i] for i in bl.clean_indices(manifest, p)} for p in counts}
    assert not (parts["TRAIN"] & parts["VALIDATION"] or parts["TRAIN"] & parts["TEST"]
                or parts["VALIDATION"] & parts["TEST"])
    assert manifest == bl.build_split_manifest(dataset)

    sample = dataset["X_eeg"][:4]
    processed = bl.preprocess_eeg(sample, sfreq)
    assert processed.shape == (4, 22, 1000) and np.array_equal(processed, bl.preprocess_eeg(sample, sfreq))
    model = bl.fit_baseline(dataset, manifest)
    train_ids = [ids[i] for i in bl.clean_indices(manifest, "TRAIN")]
    assert model["fit_trial_ids"] == train_ids
    assert [int(c) for c in model["lda"].classes_] == [0, 1, 2, 3]

    joblib.dump({"csp": model["csp"], "lda": model["lda"]}, tmp_path / "model.joblib")
    reloaded = joblib.load(tmp_path / "model.joblib")
    metrics = {}
    for name in ("VALIDATION", "TEST"):
        idx = bl.clean_indices(manifest, name)
        records, arrays = bl.predict_trials(model, dataset["X_eeg"][idx], [ids[i] for i in idx], sfreq, artifact)
        again = bl.predict_trials(reloaded, dataset["X_eeg"][idx], [ids[i] for i in idx], sfreq, artifact)
        assert records == again[0]
        assert all(np.array_equal(arrays[k], again[1][k]) for k in arrays)
        assert arrays["probabilities"].shape == (len(idx), 4) == arrays["decision_scores"].shape
        metrics[name] = bl.evaluate(bl.to_model_targets(dataset["y_semantic"][idx]), arrays["predicted"])
    print("BASELINE_SOFTWARE_EVIDENCE", {
        "excluded_rejected": len(excluded), "clean_counts": counts,
        "validation": {k: metrics["VALIDATION"][k] for k in ("n", "balanced_accuracy", "cohen_kappa")},
        "test": {k: metrics["TEST"][k] for k in ("n", "balanced_accuracy", "cohen_kappa")}})

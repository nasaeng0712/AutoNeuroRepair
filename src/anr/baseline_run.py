"""Final A01T Baseline v1 run: train, evaluate, qualify, and freeze the exact fitted stack.

    python -m anr.baseline_run <A01T.gdf> <output_dir>

Refuses to run from a dirty or non-committed repository: the Freeze Manifest
records the exact clean commit that holds the training code and config. No
tuning ever happens here; a failed qualification is stored as NOT APPROVED and
the fitted model is not saved as a frozen artifact.
"""

from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import mne
import numpy as np
import scipy
import sklearn

from . import baseline as bl
from .pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]


def model_artifact_id(source_sha256):
    """Same convention as canonical_dataset_id: one Baseline v1 per source file."""
    return f"{source_sha256}:{bl.BASELINE_VERSION}"


def clean_git_commit(repo=REPO):
    """Return HEAD only if the working tree is clean; otherwise block."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    if git("status", "--porcelain"):
        raise bl.BaselineBlockedError(
            "Repository is not clean: commit the code/config before training and freezing")
    return git("rev-parse", "HEAD")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_baseline_v1(a01t_path, output_dir, *, source_reference=None, repo=REPO):
    commit = clean_git_commit(repo)  # clean status -> record SHA -> train
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = run_pipeline(a01t_path, source_reference=source_reference)
    dataset, record = result["dataset"], result["record"]
    info = dataset["dataset_metadata"]
    sfreq, artifact_id = info["sfreq"], model_artifact_id(record["source_sha256"])
    ids = [m["trial_id"] for m in dataset["trial_metadata"]]

    manifest = bl.build_split_manifest(dataset)
    model = bl.fit_baseline(dataset, manifest)
    metrics, predictions, probes = {}, {}, {}
    for name in ("VALIDATION", "TEST"):
        idx = bl.clean_indices(manifest, name)
        records, arrays = bl.predict_trials(model, dataset["X_eeg"][idx], [ids[i] for i in idx],
                                            sfreq, artifact_id)
        truth = bl.to_model_targets(dataset["y_semantic"][idx])
        metrics[name.lower()] = bl.evaluate(truth, arrays["predicted"])
        predictions[name.lower()] = records
        probes[name.lower()] = (idx, arrays)
    qualification = bl.qualify(metrics["validation"], metrics["test"])

    # Software Verification evidence (separate from functional qualification).
    train_ids = [ids[i] for i in bl.clean_indices(manifest, "TRAIN")]
    partitions = {p: {ids[i] for i in bl.clean_indices(manifest, p)} for p in ("TRAIN", "VALIDATION", "TEST")}
    sample = dataset["X_eeg"][:3]
    reload_dir = out / "_reload_check"
    reload_dir.mkdir()
    joblib.dump({"csp": model["csp"], "lda": model["lda"]}, reload_dir / "model.joblib")
    reloaded = joblib.load(reload_dir / "model.joblib")
    reproducible = {}
    for name, (idx, arrays) in probes.items():
        again = bl.predict_trials(reloaded, dataset["X_eeg"][idx], [ids[i] for i in idx], sfreq,
                                  artifact_id)[1]
        reproducible[name] = all(np.array_equal(arrays[k], again[k]) for k in arrays)
    (reload_dir / "model.joblib").unlink()
    reload_dir.rmdir()
    software = {
        "split_manifest_deterministic": manifest == bl.build_split_manifest(dataset),
        "no_partition_overlap": not (partitions["TRAIN"] & partitions["VALIDATION"]
                                     or partitions["TRAIN"] & partitions["TEST"]
                                     or partitions["VALIDATION"] & partitions["TEST"]),
        "fit_trial_ids_are_train_clean": model["fit_trial_ids"] == train_ids,
        "filter_deterministic": bool(np.array_equal(bl.preprocess_eeg(sample, sfreq),
                                                    bl.preprocess_eeg(sample, sfreq))),
        "class_order_invariant": [int(c) for c in model["lda"].classes_] == [0, 1, 2, 3],
        "prediction_invariants_checked": True,  # predict_trials raises on any violation
        "reload_reproducible_exact": reproducible,
        "reload_tolerance": "exact equality (atol=0, rtol=0)",
    }
    software["status"] = "PASS" if (all(v for k, v in software.items() if isinstance(v, bool))
                                    and all(reproducible.values())) else "FAIL"

    versions = {"python": platform.python_version(), "mne": mne.__version__,
                "scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    config = {"training_config_schema_version": bl.CONFIG_SCHEMA_VERSION,
              "baseline_version": bl.BASELINE_VERSION, "filter": bl.FILTER_CONFIG,
              "filter_resolved": bl.resolved_filter_parameters(sfreq, dataset["X_eeg"].shape[2]),
              "baseline_window": bl.BASELINE_WINDOW, "csp": bl.CSP_CONFIG, "lda": bl.LDA_CONFIG}
    _json(out / "split_manifest.json", manifest)
    _json(out / "metrics.json", {"validation": metrics["validation"], "test": metrics["test"],
                                 "stage_gate": bl.GATE, "qualification": qualification})
    _json(out / "software_verification.json", software)
    summary = {"software_verification": software["status"],
               "functional_qualification": qualification["status"],
               "frozen_baseline": "NOT APPROVED", "training_code_commit_sha": commit,
               "model_artifact_id": artifact_id}
    if software["status"] == "PASS" and qualification["status"] == "PASS":
        joblib.dump({"csp": model["csp"], "lda": model["lda"]}, out / "model.joblib")
        model_hash = sha256((out / "model.joblib").read_bytes()).hexdigest()
        freeze = {
            "model_artifact_id": artifact_id, "training_config_schema_version": bl.CONFIG_SCHEMA_VERSION,
            "training_code_commit_sha": commit, "versions": versions,
            "canonical_dataset_id": dataset["canonical_dataset_id"],
            "split_manifest_file": "split_manifest.json",
            "frozen_model_vocabulary": bl.frozen_model_vocabulary(),
            "eeg_canonical_channel_schema": info["eeg_channel_schema"],
            "baseline_channel_selection": "all 22 canonical EEG channels (EOG not used)",
            "expected_sfreq": sfreq, "canonical_epoch": info["epoch_definition"],
            "baseline_window": bl.BASELINE_WINDOW, "config": config,
            "fitted_stack_file": "model.joblib", "fitted_stack_sha256": model_hash,
            "fitted_stack_contents": "complete fitted mne.decoding.CSP and sklearn LDA objects",
            "probability_semantics": bl.PROBABILITY_SEMANTICS,
            "decision_score_semantics": bl.DECISION_SCORE_SEMANTICS,
            "validation_metrics": metrics["validation"], "test_metrics": metrics["test"],
            "software_verification": software, "functional_qualification": qualification,
            "no_scaler": True, "no_calibration": True, "online_adaptation": False}
        _json(out / "freeze_manifest.json", freeze)
        summary["frozen_baseline"] = "COMPLETE"
        summary["fitted_stack_sha256"] = model_hash
    _json(out / "run_summary.json", summary)
    return {"summary": summary, "metrics": metrics, "qualification": qualification,
            "software": software, "manifest": manifest, "config": config, "versions": versions,
            "dataset_id": dataset["canonical_dataset_id"]}


if __name__ == "__main__":
    report = run_baseline_v1(sys.argv[1], sys.argv[2])
    print(json.dumps({"summary": report["summary"], "validation": report["metrics"]["validation"],
                      "test": report["metrics"]["test"]}, indent=2))

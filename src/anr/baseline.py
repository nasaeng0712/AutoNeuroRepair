"""CSP + LDA Baseline v1 for BCIC IV 2a A01T (subject-specific, four-class MI).

Baseline-specific only (the Canonical Trial Dataset stays untouched):
  Canonical EEG [0, 6) -> 8-30 Hz FIR band-pass -> crop [2, 6) -> CSP -> LDA.
CSP and LDA are fitted once, on TRAIN clean trials only; validation/test only
transform/predict. No scaler, no calibration, no tuning. Model target integers
come from the explicit FrozenModelVocabulary, never from estimator-sorted labels.
"""

import re

import mne
from mne.decoding import CSP
from mne.filter import create_filter, filter_data
from mne.utils import catch_logging
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
)

BASELINE_VERSION = "csp-lda-baseline-v1"
CONFIG_SCHEMA_VERSION = "baseline-config-v1"
EXPECTED_SFREQ = 250.0
PROBABILITY_SEMANTICS = "native_lda_model_estimate"
DECISION_SCORE_SEMANTICS = "lda_decision_function"

# Explicit order: model_class_index == position. Native codes are the basis only.
FROZEN_MODEL_VOCABULARY = (
    {"model_class_index": 0, "semantic_label": "left_hand", "native_event_code": "769"},
    {"model_class_index": 1, "semantic_label": "right_hand", "native_event_code": "770"},
    {"model_class_index": 2, "semantic_label": "both_feet", "native_event_code": "771"},
    {"model_class_index": 3, "semantic_label": "tongue", "native_event_code": "772"},
)
_INDEX_OF = {v["semantic_label"]: v["model_class_index"] for v in FROZEN_MODEL_VOCABULARY}

FILTER_CONFIG = {"l_freq": 8.0, "h_freq": 30.0, "method": "fir", "phase": "zero",
                 "fir_window": "hamming", "fir_design": "firwin", "filter_length": "auto",
                 "l_trans_bandwidth": "auto", "h_trans_bandwidth": "auto", "pad": "reflect_limited"}
BASELINE_WINDOW = {"tmin": 2.0, "tmax": 6.0, "endpoint_convention": "half_open"}
CSP_CONFIG = {"n_components": 4, "reg": None, "cov_est": "concat", "transform_into": "average_power",
              "log": True, "norm_trace": False, "component_order": "mutual_info"}
LDA_CONFIG = {"solver": "svd", "shrinkage": None, "priors": None, "n_components": None,
              "store_covariance": False, "tol": 1e-4, "covariance_estimator": None}
PARTITION_BY_RUN = {1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "VALIDATION", 6: "TEST"}
GATE = {"balanced_accuracy_gt": 0.25, "kappa_gt": 0.0}


class BaselineBlockedError(RuntimeError):
    """A precondition needs Control Tower review (e.g. missing class in a partition)."""


class BaselineInvariantError(AssertionError):
    """A Software Verification invariant was violated."""


def frozen_model_vocabulary():
    return [dict(v) for v in FROZEN_MODEL_VOCABULARY]


def to_model_targets(semantic_labels):
    """Explicit semantic label -> model class index (unknown label is an error)."""
    try:
        return np.array([_INDEX_OF[str(label)] for label in semantic_labels], dtype=int)
    except KeyError as exc:
        raise BaselineBlockedError(f"Label outside the FrozenModelVocabulary: {exc}") from exc


def build_split_manifest(dataset):
    """Deterministic run-based split; rejected trials stay listed but are not model-eligible."""
    meta, labels = dataset["trial_metadata"], dataset["y_semantic"]
    runs = sorted({m["task_run_index"] for m in meta})
    if runs != sorted(PARTITION_BY_RUN):
        raise BaselineBlockedError(f"Expected task-bearing runs 1..6, found {runs}")
    manifest = []
    for m, label in zip(meta, labels):
        rejected = bool(m["rejection_marker_present"])
        manifest.append({
            "trial_id": m["trial_id"], "run_id": m["task_run_index"],
            "partition": PARTITION_BY_RUN[m["task_run_index"]],
            "rejection_marker_present": rejected, "model_eligible": not rejected,
            "exclusion_reason": "trial_rejection_marker (native 1023)" if rejected else None})
    ids = [r["trial_id"] for r in manifest]
    if len(set(ids)) != len(ids):
        raise BaselineBlockedError("Duplicate trial ids in the split manifest")
    for partition in ("TRAIN", "VALIDATION", "TEST"):
        present = {str(label) for r, label in zip(manifest, labels)
                   if r["partition"] == partition and r["model_eligible"]}
        missing = [v["semantic_label"] for v in FROZEN_MODEL_VOCABULARY
                   if v["semantic_label"] not in present]
        if missing:
            raise BaselineBlockedError(f"Clean {partition} partition lacks class(es): {missing}")
    return manifest


def clean_indices(manifest, partition):
    return [i for i, r in enumerate(manifest) if r["partition"] == partition and r["model_eligible"]]


def resolved_filter_parameters(sfreq, n_times):
    """Parameters MNE resolved for FILTER_CONFIG (from the library's own log)."""
    with catch_logging(verbose="INFO") as log:
        h = create_filter(np.zeros(n_times), sfreq, FILTER_CONFIG["l_freq"], FILTER_CONFIG["h_freq"],
                          method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
                          verbose="INFO")
    text = log.getvalue()
    grab = lambda pattern: float(re.search(pattern, text).group(1))
    return {"mne_version": mne.__version__, "filter_length_samples": int(len(h)),
            "l_trans_bandwidth_hz": grab(r"Lower transition bandwidth: ([\d.]+) Hz"),
            "h_trans_bandwidth_hz": grab(r"Upper transition bandwidth: ([\d.]+) Hz"),
            "padding": FILTER_CONFIG["pad"], "sfreq": sfreq,
            "mne_log": [line for line in text.splitlines() if line.strip()]}


def window_samples(sfreq):
    """Exact sample offsets of the Baseline window [2, 6)."""
    start, stop = BASELINE_WINDOW["tmin"] * sfreq, BASELINE_WINDOW["tmax"] * sfreq
    if not (float(start).is_integer() and float(stop).is_integer()):
        raise BaselineBlockedError("Baseline window is not an exact integer number of samples")
    return int(start), int(stop)


def preprocess_eeg(X_eeg, sfreq):
    """8-30 Hz FIR band-pass of the canonical epoch, then crop [2, 6). Per-trial, label-free."""
    if sfreq != EXPECTED_SFREQ:
        raise BaselineBlockedError(f"Baseline v1 expects sfreq {EXPECTED_SFREQ}, got {sfreq}")
    start, stop = window_samples(sfreq)
    if np.shape(X_eeg)[-1] < stop:
        raise BaselineBlockedError("Epoch is shorter than the Baseline window")
    filtered = filter_data(np.asarray(X_eeg, dtype=float), sfreq, FILTER_CONFIG["l_freq"],
                           FILTER_CONFIG["h_freq"], method="fir", phase="zero",
                           fir_window="hamming", fir_design="firwin",
                           pad=FILTER_CONFIG["pad"], verbose="ERROR")
    return filtered[..., start:stop]


def build_estimators():
    return CSP(**CSP_CONFIG), LinearDiscriminantAnalysis(**LDA_CONFIG)


def fit_baseline(dataset, manifest):
    """Fit CSP then LDA once, on TRAIN clean trials only; returns the fitted stack."""
    train = clean_indices(manifest, "TRAIN")
    sfreq = dataset["dataset_metadata"]["sfreq"]
    X = preprocess_eeg(dataset["X_eeg"][train], sfreq)
    y = to_model_targets(dataset["y_semantic"][train])
    csp, lda = build_estimators()
    csp.fit(X, y)
    lda.fit(csp.transform(X), y)
    check_class_order(lda)
    return {"csp": csp, "lda": lda,
            "fit_trial_ids": [manifest[i]["trial_id"] for i in train]}


def check_class_order(lda):
    expected = [v["model_class_index"] for v in FROZEN_MODEL_VOCABULARY]
    if [int(c) for c in lda.classes_] != expected:
        raise BaselineInvariantError(f"LDA classes {list(lda.classes_)} differ from {expected}")


def predict_trials(model, X_eeg_canonical, trial_ids, sfreq, model_artifact_id):
    """Predictions for canonical [0, 6) EEG trials (transform/predict only, no fitting)."""
    features = model["csp"].transform(preprocess_eeg(X_eeg_canonical, sfreq))
    lda = model["lda"]
    probabilities, scores = lda.predict_proba(features), lda.decision_function(features)
    check_prediction_invariants(lda, features, probabilities, scores)
    order = [v["semantic_label"] for v in FROZEN_MODEL_VOCABULARY]
    predicted = probabilities.argmax(axis=1)
    records = [{
        "trial_id": tid, "model_artifact_id": model_artifact_id, "frozen_class_order": list(order),
        "predicted_model_class_index": int(p), "predicted_semantic_label": order[int(p)],
        "probability_vector": probabilities[i].tolist(), "decision_score_vector": scores[i].tolist(),
        "probability_semantics": PROBABILITY_SEMANTICS,
        "decision_score_semantics": DECISION_SCORE_SEMANTICS,
    } for i, (tid, p) in enumerate(zip(trial_ids, predicted))]
    return records, {"probabilities": probabilities, "decision_scores": scores,
                     "predicted": predicted}


def check_prediction_invariants(lda, features, probabilities, scores):
    n = len(features)
    check_class_order(lda)
    if probabilities.shape != (n, 4) or scores.shape != (n, 4):
        raise BaselineInvariantError("probability/decision-score vectors must be [N, 4]")
    if not (np.isfinite(probabilities).all() and np.isfinite(scores).all()):
        raise BaselineInvariantError("non-finite probability or decision score")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise BaselineInvariantError("probability outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-9):
        raise BaselineInvariantError("probability rows do not sum to 1")
    if not (np.array_equal(lda.predict(features), probabilities.argmax(axis=1))
            and np.array_equal(lda.predict(features), scores.argmax(axis=1))):
        raise BaselineInvariantError("prediction/probability/decision-score column orders disagree")


def evaluate(y_true, y_pred):
    labels = [v["model_class_index"] for v in FROZEN_MODEL_VOCABULARY]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "confusion_matrix": matrix.tolist(), "class_order": [v["semantic_label"] for v in FROZEN_MODEL_VOCABULARY],
            "per_class_support": {v["semantic_label"]: int(matrix[i].sum())
                                  for i, v in enumerate(FROZEN_MODEL_VOCABULARY)}}


def qualify(validation, test):
    """Stage Gate: both partitions need balanced_accuracy > 0.25 and kappa > 0."""
    gates = {f"{name}_{key}": bool(metrics[key2] > limit)
             for name, metrics in (("validation", validation), ("test", test))
             for key, key2, limit in (("balanced_accuracy", "balanced_accuracy", GATE["balanced_accuracy_gt"]),
                                      ("kappa", "cohen_kappa", GATE["kappa_gt"]))}
    return {"gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"}

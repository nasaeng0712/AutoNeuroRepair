"""Unit tests for anr.baseline / anr.baseline_run (small synthetic canonical datasets)."""

from copy import deepcopy
from pathlib import Path
import subprocess
from unittest.mock import patch

import joblib
from mne.decoding import CSP
from mne.filter import filter_data
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anr import baseline as bl
from anr.baseline_run import clean_git_commit, model_artifact_id

SHA = "c" * 64
LABELS = ["left_hand", "right_hand", "both_feet", "tongue"]
SF = 250.0


def synthetic(*, rejected=((1, 0),), seed=0, per_run=8, missing=None):
    """6 runs x per_run trials, 22 EEG channels, 1500 samples; class-dependent channel power."""
    rng = np.random.default_rng(seed)
    X, y, meta = [], [], []
    for run in range(1, 7):
        for k in range(per_run):
            label = LABELS[k % 4]
            if missing and (run, label) == missing:
                label = LABELS[(k + 1) % 4]
            trial = rng.normal(size=(22, 1500)) * 1e-5
            c = LABELS.index(label)
            trial[c * 5:(c + 1) * 5] *= 3.0
            X.append(trial)
            y.append(label)
            meta.append({"trial_id": f"{SHA}:{run * 100000 + k * 1500}", "task_run_index": run,
                         "rejection_marker_present": (run, k) in rejected})
    return {"X_eeg": np.array(X), "X_eog": rng.normal(size=(len(X), 3, 1500)) * 1e-5,
            "y_semantic": np.array(y), "trial_metadata": meta,
            "dataset_metadata": {"sfreq": SF}, "canonical_dataset_id": f"{SHA}:canonical-trial-dataset-v1"}


@pytest.fixture(scope="module")
def dataset():
    return synthetic()


@pytest.fixture(scope="module")
def manifest(dataset):
    return bl.build_split_manifest(dataset)


@pytest.fixture(scope="module")
def model(dataset, manifest):
    return bl.fit_baseline(dataset, manifest)


def test_frozen_model_vocabulary_is_explicit_and_native_code_based():
    vocab = bl.frozen_model_vocabulary()
    assert [(v["model_class_index"], v["semantic_label"], v["native_event_code"]) for v in vocab] == [
        (0, "left_hand", "769"), (1, "right_hand", "770"), (2, "both_feet", "771"), (3, "tongue", "772")]
    # Order comes from the vocabulary, not from sorting the strings.
    assert bl.to_model_targets(["tongue", "left_hand", "both_feet", "right_hand"]).tolist() == [3, 0, 2, 1]
    assert sorted(LABELS) != LABELS
    with pytest.raises(bl.BaselineBlockedError):
        bl.to_model_targets(["feet"])


def test_deterministic_run_split_and_manifest_fields(dataset, manifest):
    assert manifest == bl.build_split_manifest(deepcopy(dataset))
    assert {r["run_id"]: r["partition"] for r in manifest} == {
        1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "VALIDATION", 6: "TEST"}
    assert set(manifest[0]) == {"trial_id", "run_id", "partition", "rejection_marker_present",
                                "model_eligible", "exclusion_reason"}
    assert [r["trial_id"] for r in manifest] == [m["trial_id"] for m in dataset["trial_metadata"]]


def test_no_split_leakage(manifest):
    parts = {p: {manifest[i]["trial_id"] for i in bl.clean_indices(manifest, p)}
             for p in ("TRAIN", "VALIDATION", "TEST")}
    assert not (parts["TRAIN"] & parts["VALIDATION"] or parts["TRAIN"] & parts["TEST"]
                or parts["VALIDATION"] & parts["TEST"])
    assert all(len(v) > 0 for v in parts.values())


def test_rejected_trials_are_listed_and_excluded_from_clean_sets(dataset, manifest):
    rejected = [r for r in manifest if r["rejection_marker_present"]]
    assert len(rejected) == 1 and len(manifest) == 48  # dataset keeps every trial
    assert rejected[0]["model_eligible"] is False and rejected[0]["exclusion_reason"]
    assert all(r["exclusion_reason"] is None for r in manifest if r["model_eligible"])
    clean = {manifest[i]["trial_id"] for p in ("TRAIN", "VALIDATION", "TEST")
             for i in bl.clean_indices(manifest, p)}
    assert rejected[0]["trial_id"] not in clean and len(clean) == 47


def test_every_clean_partition_has_all_four_classes_else_blocked(dataset, manifest):
    for partition in ("TRAIN", "VALIDATION", "TEST"):
        idx = bl.clean_indices(manifest, partition)
        assert set(dataset["y_semantic"][idx]) == set(LABELS)
    broken = synthetic(per_run=4, rejected=((5, 0),))  # run 5 loses its left_hand trial
    with pytest.raises(bl.BaselineBlockedError, match="VALIDATION.*left_hand"):
        bl.build_split_manifest(broken)


def test_split_requires_runs_one_to_six(dataset):
    bad = deepcopy(dataset)
    bad["trial_metadata"][-1]["task_run_index"] = 7
    with pytest.raises(bl.BaselineBlockedError, match="runs 1..6"):
        bl.build_split_manifest(bad)


def test_filter_config_and_resolved_parameters():
    assert bl.FILTER_CONFIG["l_freq"] == 8.0 and bl.FILTER_CONFIG["h_freq"] == 30.0
    assert (bl.FILTER_CONFIG["method"], bl.FILTER_CONFIG["phase"], bl.FILTER_CONFIG["fir_window"],
            bl.FILTER_CONFIG["fir_design"]) == ("fir", "zero", "hamming", "firwin")
    resolved = bl.resolved_filter_parameters(SF, 1500)
    assert resolved["filter_length_samples"] == 413 and resolved["filter_length_samples"] % 2 == 1
    assert (resolved["l_trans_bandwidth_hz"], resolved["h_trans_bandwidth_hz"]) == (2.0, 7.5)
    assert resolved["padding"] == "reflect_limited"


def test_window_is_exactly_1000_samples_half_open_and_matches_library_filter(dataset):
    assert bl.window_samples(SF) == (500, 1500)
    X = dataset["X_eeg"][:3]
    out = bl.preprocess_eeg(X, SF)
    assert out.shape == (3, 22, 1000)  # 4.0 s at 250 Hz: no resampling
    full = filter_data(X, SF, 8.0, 30.0, method="fir", phase="zero", fir_window="hamming",
                       fir_design="firwin", pad="reflect_limited", verbose="ERROR")
    assert np.array_equal(out, full[..., 500:1500])
    assert np.array_equal(out[..., 0], full[..., 500])      # first kept sample is t = 2.0 s
    assert np.array_equal(out[..., -1], full[..., 1499])    # last kept sample is t < 6.0 s
    assert not np.array_equal(out[..., 0], full[..., 499])  # off-by-one guard


def test_preprocessing_is_deterministic_per_trial_and_label_free(dataset):
    X = dataset["X_eeg"]
    assert np.array_equal(bl.preprocess_eeg(X[:6], SF), bl.preprocess_eeg(X[:6], SF))
    # Each trial is filtered independently: no statistic is shared across trials/partitions.
    assert np.array_equal(bl.preprocess_eeg(X[[5, 20]], SF), bl.preprocess_eeg(X, SF)[[5, 20]])
    assert "y" not in bl.preprocess_eeg.__code__.co_varnames[:bl.preprocess_eeg.__code__.co_argcount]
    with pytest.raises(bl.BaselineBlockedError):
        bl.preprocess_eeg(X[:1], 128.0)
    with pytest.raises(bl.BaselineBlockedError):
        bl.preprocess_eeg(X[:1, :, :1400], SF)


def test_csp_and_lda_exact_config(model):
    csp, lda = model["csp"], model["lda"]
    assert isinstance(csp, CSP) and isinstance(lda, LinearDiscriminantAnalysis)
    params = csp.get_params()
    for key, value in {"n_components": 4, "reg": None, "cov_est": "concat",
                       "transform_into": "average_power", "log": True, "norm_trace": False,
                       "component_order": "mutual_info"}.items():
        assert params[key] == value
    for key, value in {"solver": "svd", "shrinkage": None, "priors": None, "n_components": None,
                       "store_covariance": False, "tol": 1e-4, "covariance_estimator": None}.items():
        assert lda.get_params()[key] == value
    assert bl.CSP_CONFIG["n_components"] == 4 and bl.LDA_CONFIG["solver"] == "svd"


def test_csp_and_lda_are_fitted_once_on_train_clean_trials_only(dataset, manifest):
    calls = {"csp": [], "lda": []}
    csp_fit, lda_fit = CSP.fit, LinearDiscriminantAnalysis.fit

    def spy_csp(self, X, y, *args, **kwargs):
        calls["csp"].append((np.array(X), np.array(y)))
        return csp_fit(self, X, y, *args, **kwargs)

    def spy_lda(self, X, y, *args, **kwargs):
        calls["lda"].append((np.array(X), np.array(y)))
        return lda_fit(self, X, y, *args, **kwargs)

    with patch.object(CSP, "fit", spy_csp), patch.object(LinearDiscriminantAnalysis, "fit", spy_lda):
        fitted = bl.fit_baseline(dataset, manifest)
    train = bl.clean_indices(manifest, "TRAIN")
    assert len(calls["csp"]) == 1 and len(calls["lda"]) == 1
    X_csp, y_csp = calls["csp"][0]
    assert X_csp.shape == (len(train), 22, 1000)  # all 22 EEG channels, nothing else
    assert np.array_equal(X_csp, bl.preprocess_eeg(dataset["X_eeg"][train], SF))
    assert np.array_equal(y_csp, bl.to_model_targets(dataset["y_semantic"][train]))
    X_lda, y_lda = calls["lda"][0]
    assert X_lda.shape == (len(train), 4) and np.array_equal(y_lda, y_csp)
    assert np.array_equal(X_lda, fitted["csp"].transform(X_csp))
    assert fitted["fit_trial_ids"] == [manifest[i]["trial_id"] for i in train]
    outside = {m["trial_id"] for i, m in enumerate(manifest) if i not in set(train)}
    assert not outside & set(fitted["fit_trial_ids"])  # no validation/test/rejected trial id


def test_fit_ignores_rejected_validation_test_data_labels_and_eog(dataset, manifest, model):
    other = deepcopy(dataset)
    train = set(bl.clean_indices(manifest, "TRAIN"))
    outside = [i for i in range(len(manifest)) if i not in train]
    rng = np.random.default_rng(5)
    other["X_eeg"][outside] = np.nan                  # rejected, validation and test signals
    other["X_eog"][:] = np.nan                        # EOG is not a Baseline input
    for i in outside:                                 # validation/test labels shuffled
        other["y_semantic"][i] = LABELS[rng.integers(4)]
    refit = bl.fit_baseline(other, manifest)
    assert np.array_equal(refit["csp"].filters_, model["csp"].filters_)
    assert np.array_equal(refit["csp"].patterns_, model["csp"].patterns_)
    assert np.array_equal(refit["lda"].coef_, model["lda"].coef_)
    assert np.array_equal(refit["lda"].intercept_, model["lda"].intercept_)


def test_class_order_invariant_even_when_data_order_differs(dataset, manifest, model):
    assert [int(c) for c in model["lda"].classes_] == [0, 1, 2, 3]
    shuffled = deepcopy(dataset)
    train = bl.clean_indices(manifest, "TRAIN")
    order = np.array(train)[::-1]  # tongue-heavy end first: estimator sorting must not matter
    shuffled["X_eeg"][train] = dataset["X_eeg"][order]
    shuffled["y_semantic"][train] = dataset["y_semantic"][order]
    assert [int(c) for c in bl.fit_baseline(shuffled, manifest)["lda"].classes_] == [0, 1, 2, 3]
    wrong = LinearDiscriminantAnalysis().fit(np.random.default_rng(0).normal(size=(12, 3)),
                                             [3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    wrong.classes_ = np.array([3, 0, 1, 2])
    with pytest.raises(bl.BaselineInvariantError):
        bl.check_class_order(wrong)


def test_prediction_contract_probability_and_decision_invariants(dataset, manifest, model):
    idx = bl.clean_indices(manifest, "VALIDATION")
    ids = [manifest[i]["trial_id"] for i in idx]
    artifact = model_artifact_id(SHA)
    records, arrays = bl.predict_trials(model, dataset["X_eeg"][idx], ids, SF, artifact)
    assert artifact == f"{SHA}:csp-lda-baseline-v1"
    n = len(idx)
    assert arrays["probabilities"].shape == (n, 4) and arrays["decision_scores"].shape == (n, 4)
    assert np.isfinite(arrays["probabilities"]).all() and np.isfinite(arrays["decision_scores"]).all()
    assert arrays["probabilities"].min() >= 0 and arrays["probabilities"].max() <= 1
    assert np.allclose(arrays["probabilities"].sum(axis=1), 1.0, rtol=0, atol=1e-9)
    first = records[0]
    assert set(first) == {"trial_id", "model_artifact_id", "frozen_class_order",
                          "predicted_model_class_index", "predicted_semantic_label",
                          "probability_vector", "decision_score_vector",
                          "probability_semantics", "decision_score_semantics"}
    assert first["frozen_class_order"] == LABELS
    assert first["probability_semantics"] == "native_lda_model_estimate"
    assert first["decision_score_semantics"] == "lda_decision_function"
    assert first["predicted_semantic_label"] == LABELS[first["predicted_model_class_index"]]
    assert first["predicted_model_class_index"] == int(np.argmax(first["probability_vector"])) \
        == int(np.argmax(first["decision_score_vector"]))
    assert [r["trial_id"] for r in records] == ids
    lda = model["lda"]
    features = model["csp"].transform(bl.preprocess_eeg(dataset["X_eeg"][idx], SF))
    bad = arrays["probabilities"].copy()
    bad[0] = [0.5, 0.5, 0.5, 0.5]
    with pytest.raises(bl.BaselineInvariantError, match="sum"):
        bl.check_prediction_invariants(lda, features, bad, arrays["decision_scores"])
    bad = arrays["decision_scores"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(bl.BaselineInvariantError, match="non-finite"):
        bl.check_prediction_invariants(lda, features, arrays["probabilities"], bad)
    swapped = arrays["probabilities"][:, [1, 0, 2, 3]]
    with pytest.raises(bl.BaselineInvariantError, match="orders disagree"):
        bl.check_prediction_invariants(lda, features, swapped, arrays["decision_scores"])


def test_no_calibration_or_scaler_in_the_fitted_stack(model):
    assert sorted(model) == ["csp", "fit_trial_ids", "lda"]
    assert type(model["lda"]).__name__ == "LinearDiscriminantAnalysis"
    assert not any("calibrat" in type(v).__name__.lower() or "scaler" in type(v).__name__.lower()
                   for v in model.values())


def test_reload_reproduces_predictions_exactly(dataset, manifest, model, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"csp": model["csp"], "lda": model["lda"]}, path)
    reloaded = joblib.load(path)
    idx = bl.clean_indices(manifest, "TEST")
    ids = [manifest[i]["trial_id"] for i in idx]
    first = bl.predict_trials(model, dataset["X_eeg"][idx], ids, SF, "a")
    again = bl.predict_trials(reloaded, dataset["X_eeg"][idx], ids, SF, "a")
    assert first[0] == again[0]  # same labels, class order, probabilities, scores
    for key in first[1]:
        assert np.array_equal(first[1][key], again[1][key])  # tolerance 0: bitwise equal
    assert [int(c) for c in reloaded["lda"].classes_] == [0, 1, 2, 3]


def test_metrics_and_stage_gate():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    perfect = bl.evaluate(y, y)
    assert perfect["accuracy"] == perfect["balanced_accuracy"] == perfect["cohen_kappa"] == 1.0
    assert perfect["per_class_support"] == dict(zip(LABELS, [2, 2, 2, 2]))
    assert perfect["confusion_matrix"] == (np.eye(4, dtype=int) * 2).tolist()
    chance = bl.evaluate(y, np.zeros(8, dtype=int))
    assert chance["balanced_accuracy"] == 0.25 and chance["cohen_kappa"] == 0.0
    assert bl.qualify(perfect, perfect)["status"] == "PASS"
    assert bl.qualify(perfect, chance)["status"] == "FAIL"   # BA > 0.25 is strict
    assert bl.qualify(chance, perfect)["gates"]["validation_balanced_accuracy"] is False
    assert bl.qualify(perfect, perfect)["gates"]["test_kappa"] is True


def test_clean_git_gate(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(bl.BaselineBlockedError, match="not clean"):
        clean_git_commit(tmp_path)
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    sha = clean_git_commit(tmp_path)
    assert len(sha) == 40
    (tmp_path / "b.txt").write_text("y")
    with pytest.raises(bl.BaselineBlockedError):
        clean_git_commit(tmp_path)

"""Canonical Trial Dataset: X_eeg [N,22,T], X_eog [N,3,T], y_semantic [N], metadata.

Built from Canonical Trial Signal Extraction output after Channel
Canonicalization. Signals keep the canonical [0, 6) window and are otherwise
untouched (no filtering, crop or scaling). y_semantic holds Pipeline semantic
labels, not model class indices. source_processing_status is an immutable
snapshot of the provenance assessment at creation time; this module never
re-assesses provenance.
"""

import numpy as np

from .channel_canonicalization import canonicalize_channels
from .event_semantics import task_label_schema
from .validation.common import check, make_result

PIPELINE_SCHEMA_VERSION = "canonical-trial-dataset-v1"
VALIDATOR_VERSION = "1.0.0"


def canonical_dataset_id(source_sha256):
    return f"{source_sha256}:canonical-trial-dataset-v1"


def build_canonical_dataset(extraction, assembly, record, observed_label_coverage):
    """Assemble the dataset dict; raises ChannelCanonicalizationError on schema mismatch."""
    collection, epochs = extraction["collection"], extraction["epochs"]
    channels = canonicalize_channels(extraction["signals"], collection["channel_names"],
                                     collection["channel_types"])
    runs = {}
    for ordinal, run in enumerate(assembly["task_bearing_runs"], 1):
        for trial_id in run["trial_ids"]:
            runs[trial_id] = (ordinal, run["run_segment_index"])
    by_id = {t["trial_id"]: t for t in assembly["trials"]}
    trial_metadata = []
    for epoch in epochs:
        trial = by_id[epoch["trial_id"]]
        trial_metadata.append(dict(
            epoch, task_run_index=runs[epoch["trial_id"]][0],
            run_segment_index=runs[epoch["trial_id"]][1],
            cue_source_sample_index=trial["task_label_event"]["source_sample_index"],
            rejection_markers=[{"event_index": m["event_index"],
                                "source_sample_index": m["source_sample_index"]}
                               for m in trial["rejection_markers"]]))
    sha = record["source_sha256"]
    dataset_metadata = {
        "canonical_dataset_id": canonical_dataset_id(sha),
        "dataset_id": assembly["dataset_id"], "task_id": assembly["task_id"],
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "channel_schema_version": channels["channel_schema_version"],
        "source_sha256": sha, "source_artifact_id": record["artifact_id"],
        "source_provenance_ref": {"artifact_id": record["artifact_id"], "source_sha256": sha,
                                  "provenance_schema_version": record["schema_version"]},
        "source_processing_status": record["processing_status"],
        "pipeline_transformation_history": [
            {"step": "trial_signal_selection",
             "operation": "source sample selection of [0.0, 6.0) from each 768 anchor",
             "endpoint_convention": "half_open"},
            {"step": "channel_canonicalization",
             "operation": "selection into EEG/EOG groups and deterministic reorder"}],
        "task_label_schema": task_label_schema(),
        "observed_label_coverage": observed_label_coverage,
        "eeg_channel_schema": channels["eeg_channels"],
        "eog_channel_schema": channels["eog_channels"],
        "sfreq": collection["sfreq"],
        "signal_representation": {
            "unit_representation": collection["source_unit_representation"],
            "note": "values as returned by the loader; not modified by the pipeline"},
        "epoch_definition": {"tmin": collection["tmin"], "tmax": collection["tmax"],
                             "endpoint_convention": collection["endpoint_convention"],
                             "n_times": collection["n_times"]},
    }
    return {"canonical_dataset_id": dataset_metadata["canonical_dataset_id"],
            "X_eeg": channels["X_eeg"], "X_eog": channels["X_eog"],
            "y_semantic": np.array([e["ground_truth_semantic_label"] for e in epochs]),
            "trial_metadata": trial_metadata, "dataset_metadata": dataset_metadata}


def validate_canonical_dataset(dataset, record):
    """Dataset invariants as PASS/FAIL checks (same result schema as other validators)."""
    X_eeg, X_eog, y = dataset["X_eeg"], dataset["X_eog"], dataset["y_semantic"]
    meta, info = dataset["trial_metadata"], dataset["dataset_metadata"]
    sha = record["source_sha256"]
    epoch = info["epoch_definition"]

    def expect(observed, ok, reason):
        return observed, None if ok else reason

    def positive():
        shape = {"X_eeg": list(X_eeg.shape), "X_eog": list(X_eog.shape)}
        ok = (X_eeg.ndim == X_eog.ndim == 3 and X_eeg.shape[0] == X_eog.shape[0]
              and X_eeg.shape[2] == X_eog.shape[2] and min(X_eeg.shape + X_eog.shape) > 0)
        return expect(shape, ok, "N, C and T must be positive and shared by EEG and EOG")

    def finite():
        bad = int(np.count_nonzero(~np.isfinite(X_eeg)) + np.count_nonzero(~np.isfinite(X_eog)))
        return expect({"non_finite_values": bad}, bad == 0, "non-finite values found")

    def counts():
        n = {"N": X_eeg.shape[0], "y_semantic": len(y), "trial_metadata": len(meta)}
        return expect(n, len(set(n.values())) == 1, "label/metadata count differs from N")

    def trial_ids():
        ids = [m["trial_id"] for m in meta]
        ok = (len(set(ids)) == len(ids) and all(
            m["trial_id"] == f"{sha}:{m['anchor_source_sample_index']}" for m in meta))
        return expect({"n_unique": len(set(ids))}, ok, "trial ids not unique or not source_sha256:anchor")

    def common_epoch():
        keys = ("tmin", "tmax", "endpoint_convention", "n_times")
        ok = (all(all(m[k] == epoch[k] for k in keys) for m in meta)
              and epoch["n_times"] == X_eeg.shape[2]
              and all(m["signal_end_source_sample_exclusive"] - m["signal_start_source_sample"]
                      == epoch["n_times"] for m in meta))
        return expect(epoch, ok, "epoch definition differs between trials or from the arrays")

    def common_representation():
        ok = (all(m["sfreq"] == info["sfreq"] for m in meta)
              and len({m["source_artifact_id"] for m in meta}) == 1
              and all(m["source_artifact_id"] == info["source_artifact_id"] for m in meta)
              and all(m["source_processing_status"] == info["source_processing_status"] for m in meta))
        return expect({"sfreq": info["sfreq"]}, ok, "sfreq/source lineage/status differs between trials")

    def schema():
        eeg, eog = info["eeg_channel_schema"], info["eog_channel_schema"]
        ids = [c["canonical_channel_id"] for c in eeg + eog]
        ok = (len(eeg) == X_eeg.shape[1] and len(eog) == X_eog.shape[1]
              and len(set(ids)) == len(ids))
        return expect({"eeg": len(eeg), "eog": len(eog)}, ok, "channel schema does not match arrays")

    def lineage():
        ok = (dataset["canonical_dataset_id"] == info["canonical_dataset_id"]
              == f"{sha}:canonical-trial-dataset-v1"
              and info["source_sha256"] == sha and info["source_artifact_id"] == record["artifact_id"]
              and info["source_processing_status"] == record["processing_status"])
        return expect({"canonical_dataset_id": dataset["canonical_dataset_id"]}, ok,
                      "dataset lineage differs from the provenance record")

    def labels():
        allowed = set(info["task_label_schema"]["labels"])
        return expect(sorted(set(y.tolist())), set(y.tolist()) <= allowed,
                      "y_semantic holds a value outside the TaskLabelSchema")

    def no_model_fields():
        names = set(info) | {k for m in meta for k in m} | set(dataset)
        bad = sorted(n for n in names if "model_class" in n or "feature" in n or "baseline" in n)
        return expect(bad, not bad, "model-specific fields do not belong in the dataset")

    checks = [
        check("POSITIVE_DIMENSIONS", "N > 0, C > 0, T > 0; EEG and EOG share N and T", positive),
        check("FINITE_VALUES", "all signal values finite", finite),
        check("LABEL_METADATA_COUNTS", "len(y_semantic) == len(trial_metadata) == N", counts),
        check("STABLE_TRIAL_IDS", "unique trial ids equal source_sha256:anchor_sample", trial_ids),
        check("COMMON_EPOCH_DEFINITION", "identical canonical [0, 6) epoch for every trial", common_epoch),
        check("COMMON_SIGNAL_REPRESENTATION", "common sfreq, source lineage and status snapshot",
              common_representation),
        check("CHANNEL_SCHEMA", "EEG/EOG schema matches the arrays, ids unique", schema),
        check("DATASET_LINEAGE", "dataset id and lineage match the provenance record", lineage),
        check("SEMANTIC_LABELS", "y_semantic values belong to TaskLabelSchema", labels),
        check("NO_MODEL_FIELDS", "no model class index, features or Baseline filtering", no_model_fields),
    ]
    return make_result("canonical_trial_dataset", VALIDATOR_VERSION, record, checks)

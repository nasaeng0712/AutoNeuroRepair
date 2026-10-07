"""Trial Assembly Metadata v1: trial occurrences as metadata only.

Flow: decoded annotations -> Event Semantic records -> require_semantic_complete
-> deterministic trial metadata. No signal array is read, sliced or extracted;
no Baseline inclusion/exclusion decision is made (a rejection marker is only
recorded). Event Semantic and provenance are used as they are.

Rules (all positions are source-relative integer sample indices):
  * every native "768" (trial_anchor) is one trial; its event span is
    [anchor, earliest(next 768, next 32766, source end)), start inclusive,
    end exclusive, with "next" meaning a strictly larger sample index;
  * the span must hold exactly one task_label event; its semantic_label_id is
    the trial's ground truth. Membership depends on sample index only, so
    events sharing a sample (e.g. 768 and 1023) need no tie-break;
  * the label's cue must be exactly 2 * sfreq samples after the anchor;
  * every task_label and every 1023 must belong to exactly one span;
  * trial_id = "<source_sha256>:<anchor_source_sample_index>".
"""

from bisect import bisect_left, bisect_right

from .event_semantics import DATASET_ID, TASK_ID, interpret_events, require_semantic_complete
from .validation.common import check, make_result

TRIAL_ASSEMBLY_VERSION = "1.0.0"
CUE_SECONDS = 2


class TrialAssemblyError(ValueError):
    """Trial assembly rule violated; no trial metadata is produced."""


def source_sample_indices(onsets, sfreq, n_samples):
    """Exact integer sample index of each onset (seconds from source start).

    The reader derives onset as sample / sfreq, so the index is accepted only
    if index / sfreq reproduces the onset bit-for-bit; no tolerance is used.
    """
    indices = []
    for onset in onsets:
        index = round(float(onset) * sfreq)
        if float(index) / sfreq != float(onset):
            raise TrialAssemblyError(f"Onset {onset!r} is not an exact sample at sfreq={sfreq}")
        if not 0 <= index < n_samples:
            raise TrialAssemblyError(f"Event sample {index} outside source [0, {n_samples})")
        indices.append(index)
    return indices


def _event(index, sample, record):
    return {"event_index": index, "source_sample_index": sample, "event": dict(record)}


def assemble_trials(semantic_records, samples, *, sfreq, n_samples, source_sha256,
                    source_artifact_id):
    """Assemble trial metadata. Raises SemanticIncompleteError/TrialAssemblyError."""
    require_semantic_complete(semantic_records)  # fail closed before anything else
    records, samples = list(semantic_records), [int(s) for s in samples]
    if len(records) != len(samples):
        raise TrialAssemblyError("Every event needs exactly one source sample index")
    if any(b < a for a, b in zip(samples, samples[1:])):
        raise TrialAssemblyError("Events are not in source order")
    by_role = {}
    for i, record in enumerate(records):
        by_role.setdefault(record["event_role"], []).append(i)
    anchors = by_role.get("trial_anchor", [])
    anchor_samples = [samples[i] for i in anchors]
    run_starts = by_role.get("run_start", [])
    run_samples = [samples[i] for i in run_starts]
    if len(set(anchor_samples)) != len(anchors):
        raise TrialAssemblyError("Two trial anchors share a source sample")
    if len(set(run_samples)) != len(run_starts):
        raise TrialAssemblyError("Two run starts share a source sample")

    trials, used_labels, used_rejections = [], set(), set()
    for anchor in anchors:
        start = samples[anchor]
        later = [s for s in anchor_samples + run_samples if s > start]
        end = min(later + [n_samples])
        members = range(bisect_left(samples, start), bisect_left(samples, end))
        labels = [i for i in members if records[i]["event_role"] == "task_label"]
        if len(labels) != 1:
            raise TrialAssemblyError(
                f"Anchor at sample {start}: expected exactly 1 task label, found {len(labels)}")
        label = labels[0]
        delta = samples[label] - start
        if delta != CUE_SECONDS * sfreq:
            raise TrialAssemblyError(
                f"Anchor at sample {start}: cue delta {delta} != {CUE_SECONDS} * sfreq")
        rejections = [i for i in members if records[i]["event_role"] == "trial_rejection_marker"]
        used_labels.add(label)
        used_rejections.update(rejections)
        segment = bisect_right(run_samples, start)  # 0 = before the first run start
        trials.append({
            "trial_id": f"{source_sha256}:{start}",
            "dataset_id": DATASET_ID, "task_id": TASK_ID,
            "source_sha256": source_sha256, "source_artifact_id": source_artifact_id,
            "anchor": _event(anchor, start, records[anchor]),
            "task_label_event": _event(label, samples[label], records[label]),
            "semantic_label_id": records[label]["semantic_label_id"],
            "cue_delta_samples": delta,
            "span": {"start_sample": start, "end_sample_exclusive": end},
            "rejection_markers": [_event(i, samples[i], records[i]) for i in rejections],
            "rejection_marker_present": bool(rejections),
            "run_segment_index": segment,
            "run_start_event_index": run_starts[segment - 1] if segment else None,
        })

    labels_all, rejections_all = by_role.get("task_label", []), by_role.get("trial_rejection_marker", [])
    if set(labels_all) != used_labels or len(used_labels) != len(trials):
        raise TrialAssemblyError("Orphan or shared task label: not one-to-one with anchors")
    if set(rejections_all) != used_rejections:
        raise TrialAssemblyError("Orphan rejection marker: outside every trial span")
    if len({t["trial_id"] for t in trials}) != len(trials):
        raise TrialAssemblyError("Trial ids are not unique")

    runs = {}
    for trial in trials:
        runs.setdefault(trial["run_segment_index"], []).append(trial["trial_id"])
    return {
        "assembly_version": TRIAL_ASSEMBLY_VERSION, "dataset_id": DATASET_ID, "task_id": TASK_ID,
        "source_sha256": source_sha256, "source_artifact_id": source_artifact_id,
        "sfreq": sfreq, "n_samples": n_samples, "trials": trials,
        "task_bearing_runs": [
            {"run_segment_index": k, "run_start_event_index": run_starts[k - 1] if k else None,
             "trial_ids": ids} for k, ids in sorted(runs.items())],
        "evidence": {"observed_anchors": len(anchors), "associated_task_labels": len(used_labels),
                     "observed_task_labels": len(labels_all),
                     "observed_rejection_markers": len(rejections_all),
                     "associated_rejection_markers": len(used_rejections)},
    }


def assemble_from_raw(raw, record):
    """Read decoded annotations from a loaded Raw and assemble trial metadata."""
    annotations = raw.annotations
    if raw.first_samp != 0 or annotations.orig_time != raw.info.get("meas_date"):
        raise TrialAssemblyError("Annotation onsets are not relative to the source start")
    sfreq, n_samples = raw.info["sfreq"], int(raw.n_times)
    samples = source_sample_indices(annotations.onset, sfreq, n_samples)
    records = interpret_events([str(d) for d in annotations.description])
    return assemble_trials(records, samples, sfreq=sfreq, n_samples=n_samples,
                           source_sha256=record["source_sha256"],
                           source_artifact_id=record["artifact_id"])


def validate_a01t_trial_structure(assembly, record):
    """Normative BCIC IV 2a A01T criteria (PASS/FAIL), separate from assembly rules."""
    trials, runs = assembly["trials"], assembly["task_bearing_runs"]
    classes = ("left_hand", "right_hand", "both_feet", "tongue")
    by_id = {t["trial_id"]: t for t in trials}
    evidence = assembly["evidence"]

    def counts(items):
        return {c: sum(t["semantic_label_id"] == c for t in items) for c in classes}

    def expect(observed, expected, reason):
        return observed, None if observed == expected else reason

    per_run = {r["run_segment_index"]: counts([by_id[i] for i in r["trial_ids"]]) for r in runs}
    checks = [
        check("ANCHOR_COUNT", "288 trial anchors",
              lambda: expect(evidence["observed_anchors"], 288, "anchor count differs")),
        check("ASSOCIATED_LABEL_COUNT", "288 associated task labels, one per anchor",
              lambda: expect({"associated": evidence["associated_task_labels"],
                              "observed": evidence["observed_task_labels"], "trials": len(trials)},
                             {"associated": 288, "observed": 288, "trials": 288},
                             "label association count differs")),
        check("OVERALL_LABEL_COVERAGE", "72 trials for each of the four labels",
              lambda: expect(counts(trials), dict.fromkeys(classes, 72), "label coverage differs")),
        check("TASK_BEARING_RUN_COUNT", "6 task-bearing MI run segments",
              lambda: expect(len(runs), 6, "run count differs")),
        check("TRIALS_PER_RUN", "48 trials in every task-bearing run",
              lambda: expect([len(r["trial_ids"]) for r in runs], [48] * len(runs),
                             "trials per run differ")),
        check("PER_RUN_LABEL_COVERAGE", "12 trials per label in every task-bearing run",
              lambda: expect(list(per_run.values()), [dict.fromkeys(classes, 12)] * len(runs),
                             "per-run label coverage differs")),
        check("CUE_TIMING", "every cue exactly 2 seconds (2 * sfreq samples) after its anchor",
              lambda: expect(sorted({t["cue_delta_samples"] for t in trials}),
                             [CUE_SECONDS * assembly["sfreq"]], "cue timing differs")),
        check("TRIAL_ID_UNIQUENESS", "trial ids unique and equal source_sha256:anchor_sample",
              lambda: expect(len(by_id) == len(trials) and all(
                  t["trial_id"] == f"{t['source_sha256']}:{t['anchor']['source_sample_index']}"
                  for t in trials), True, "trial id rule violated")),
        check("REJECTION_MARKER_ASSOCIATION", "every 1023 belongs to exactly one trial span",
              lambda: expect(evidence["associated_rejection_markers"],
                             evidence["observed_rejection_markers"], "orphan 1023")),
    ]
    return make_result("a01t_trial_structure", TRIAL_ASSEMBLY_VERSION, record, checks)

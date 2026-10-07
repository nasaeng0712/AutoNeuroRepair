"""Event Semantic interpretation and observed coverage for BCIC IV 2a.

Input is the already decoded annotation description (``raw.annotations
.description``, strings such as ``"769"``); GDF decoding stays in the loader.
Output is interpretation metadata only: nothing here reads or changes the Raw
signal, the provenance record or processing_status.

Three separate things are kept apart:
  * the normative event schema / TaskLabelSchema (what each native code means),
  * observed coverage (what a given artifact actually contains),
  * model class index/order (NOT created here; a later Baseline-freeze concern).

A native code without an approved definition is never dropped or guessed: it
becomes a record with event_role "unknown_native_event" (task_id and
semantic_label_id None) and is counted in the coverage. An artifact containing
such a record is not semantic-complete; require_semantic_complete() is the
fail-closed check a later Trial stage must apply before using the events.
"""

from .loader import OFFICIAL_DOCUMENT

DATASET_ID = "BCIC_IV_2A"
SEMANTIC_SCHEMA_VERSION = "1.0.0"
TASK_ID = "motor_imagery"

# native code -> (event_role, task_id, semantic_label_id, human_readable_name)
_DEFINITIONS = {
    "276": ("auxiliary_recording_marker", None, None, "Idling EEG (eyes open)"),
    "277": ("auxiliary_recording_marker", None, None, "Idling EEG (eyes closed)"),
    "768": ("trial_anchor", TASK_ID, None, "Start of a trial"),
    "769": ("task_label", TASK_ID, "left_hand", "Left hand motor imagery"),
    "770": ("task_label", TASK_ID, "right_hand", "Right hand motor imagery"),
    "771": ("task_label", TASK_ID, "both_feet", "Both feet motor imagery"),
    "772": ("task_label", TASK_ID, "tongue", "Tongue motor imagery"),
    "783": ("unlabeled_task_cue", TASK_ID, None, "Unknown cue"),
    "1023": ("trial_rejection_marker", TASK_ID, None, "Rejected trial"),
    "1072": ("auxiliary_recording_marker", None, None, "Eye movements"),
    "32766": ("run_start", TASK_ID, None, "Start of a new run"),
}
UNKNOWN_EVENT_ROLE = "unknown_native_event"


def _code_order(code):
    return (not code.isdigit(), int(code) if code.isdigit() else 0, code)


class SemanticIncompleteError(ValueError):
    """Events contain a native code without an approved semantic definition."""


def _record(code, definition):
    role, task_id, label_id, name = definition
    return {"native_event_code": code, "event_role": role, "task_id": task_id,
            "semantic_label_id": label_id, "human_readable_name": name}


def event_semantic_schema():
    """Normative event definitions. Contains no observed counts and no class order."""
    return {"dataset_id": DATASET_ID, "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "normative_source": OFFICIAL_DOCUMENT,
            "events": {code: _record(code, d) for code, d in _DEFINITIONS.items()}}


def task_label_schema():
    """Supervised labels only (event_role == task_label); 783 etc. are excluded.

    ``labels`` maps semantic_label_id to its human-readable name; it carries no
    model class index and its key order has no meaning.
    """
    return {"dataset_id": DATASET_ID, "task_id": TASK_ID,
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "normative_source": OFFICIAL_DOCUMENT,
            "labels": {d[2]: d[3] for d in _DEFINITIONS.values() if d[0] == "task_label"}}


def interpret_event(native_code):
    """Return the semantic record for one decoded native code (str or str-like).

    An undefined code keeps its original value and gets the unknown role; its
    human_readable_name is None because no name is approved for it.
    """
    code = str(native_code)
    if code not in _DEFINITIONS:
        return _record(code, (UNKNOWN_EVENT_ROLE, None, None, None))
    return _record(code, _DEFINITIONS[code])


def interpret_events(descriptions):
    """One record per input event, in input order (position preserves linkage)."""
    return [interpret_event(code) for code in descriptions]


def require_semantic_complete(records):
    """Fail closed: raise if any record is an unknown native event.

    Takes the output of interpret_events(). Passing means only that every
    native code has an approved definition; nothing else is implied.
    """
    unknown = sorted({r["native_event_code"] for r in records
                      if r["event_role"] == UNKNOWN_EVENT_ROLE}, key=_code_order)
    if unknown:
        raise SemanticIncompleteError(
            f"Event semantics incomplete; undefined native event code(s): {unknown}")


def observe_coverage(descriptions):
    """Count what an artifact actually contains, separately from the schema.

    ``native_event_counts`` lists every observed code plus every defined code
    (0 when unobserved, e.g. 783); ``task_label_counts`` lists all four
    supervised labels. Codes without an approved definition are counted and
    listed in ``undefined_native_codes``, never dropped or interpreted.
    """
    counts = {}
    for code in descriptions:
        counts[str(code)] = counts.get(str(code), 0) + 1
    native = {code: counts.get(code, 0) for code in _DEFINITIONS}
    native.update(counts)
    labels = {d[2]: native[code] for code, d in _DEFINITIONS.items() if d[0] == "task_label"}
    return {"dataset_id": DATASET_ID, "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "total_events": sum(counts.values()),
            "native_event_counts": dict(sorted(native.items(), key=lambda kv: _code_order(kv[0]))),
            "task_label_counts": labels,
            "undefined_native_codes": sorted((c for c in counts if c not in _DEFINITIONS), key=_code_order)}

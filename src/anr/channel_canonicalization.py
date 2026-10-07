"""Channel Canonicalization: split [N, 25, T] into EEG [N, 22, T] and EOG [N, 3, T].

channel_schema_v1 is a dataset-scoped ordinal identity (bciciv2a:eeg:01..22,
bciciv2a:eog:01..03). No 10-20 electrode name is restored: each canonical id is
tied to the exact source channel name delivered by the loader for A01T
(order and types verified against the actual loader output). Only channel
selection and deterministic reorder are done; values are never modified.

Fail closed: missing, duplicate, ambiguous, type-mismatched, extra or
unvalidated-alias channels all raise ChannelCanonicalizationError.
"""

from collections import Counter

import numpy as np

CHANNEL_SCHEMA_VERSION = "channel_schema_v1"

# Exact source names, in canonical (= observed A01T loader) order.
_EEG_SOURCE_NAMES = (
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5", "EEG-C3", "EEG-6",
    "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9", "EEG-10", "EEG-11", "EEG-12",
    "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16")
_EOG_SOURCE_NAMES = ("EOG-left", "EOG-central", "EOG-right")


class ChannelCanonicalizationError(ValueError):
    """The source channels do not match the approved channel schema exactly."""


def channel_schema_v1():
    """Fresh list of schema records (22 EEG then 3 EOG)."""
    return ([{"canonical_channel_id": f"bciciv2a:eeg:{i:02d}", "source_channel_name": name,
              "channel_type": "eeg"} for i, name in enumerate(_EEG_SOURCE_NAMES, 1)]
            + [{"canonical_channel_id": f"bciciv2a:eog:{i:02d}", "source_channel_name": name,
                "channel_type": "eog"} for i, name in enumerate(_EOG_SOURCE_NAMES, 1)])


def _check_schema(schema):
    ids = [s["canonical_channel_id"] for s in schema]
    names = [s["source_channel_name"] for s in schema]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ChannelCanonicalizationError(
            "Ambiguous schema: a canonical id or source name is used more than once")
    if any(s["channel_type"] not in {"eeg", "eog"} for s in schema):
        raise ChannelCanonicalizationError("Schema channel type must be eeg or eog")


def canonicalize_channels(signals, channel_names, channel_types, schema=None):
    """Return EEG/EOG signal groups in canonical order with channel records."""
    schema = channel_schema_v1() if schema is None else schema
    _check_schema(schema)
    names, types = list(channel_names), list(channel_types)
    shape = np.shape(signals)
    if len(shape) != 3 or not (shape[1] == len(names) == len(types)):
        raise ChannelCanonicalizationError(
            "signals must be [trials, channels, samples] matching the channel names and types")
    problems = []
    duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
    known = {s["source_channel_name"]: s for s in schema}
    unknown = [n for n in names if n not in known]
    missing = [n for n in known if n not in names]
    mismatched = [n for n, t in zip(names, types) if n in known and known[n]["channel_type"] != t]
    if duplicates:
        problems.append(f"duplicate identity: {duplicates}")
    if unknown:
        problems.append(f"extra channel or unvalidated alias (no alias is accepted): {unknown}")
    if missing:
        problems.append(f"missing required channel: {missing}")
    if mismatched:
        problems.append(f"type mismatch: {mismatched}")
    if problems:
        raise ChannelCanonicalizationError("; ".join(problems))
    index = {n: i for i, n in enumerate(names)}
    signals = np.asarray(signals)
    out = {"channel_schema_version": CHANNEL_SCHEMA_VERSION}
    for group in ("eeg", "eog"):
        members = [dict(s, source_channel_index=index[s["source_channel_name"]])
                   for s in schema if s["channel_type"] == group]
        out[f"X_{group}"] = signals[:, [m["source_channel_index"] for m in members], :]
        out[f"{group}_channels"] = members
    return out

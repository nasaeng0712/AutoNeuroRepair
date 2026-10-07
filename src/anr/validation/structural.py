"""Structural validator: can the loaded data be used by a next computation step?

Not an EEG-quality check. Expected dataset values (25 channels, 250 Hz,
22 EEG + 3 EOG) are deliberately NOT used as a PASS oracle; the current Raw
is compared only with itself and with the load-time snapshot in the record.
"""

from collections import Counter
import math
from numbers import Real

import numpy as np

from .common import check, make_result

VALIDATOR_NAME = "structural_integrity"
VALIDATOR_VERSION = "1.0.0"


def _plain(value):
    """JSON-safe observed value (NaN/Inf and odd objects become strings)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real) and math.isfinite(value):
        return value
    return repr(value)


def validate_structural(raw, record):
    snapshot = record["metadata"]
    held = {}  # only the shape is kept; the signal array is not retained

    def shape_2d():
        shape = held.get("shape")
        if shape is None:
            raise ValueError("signal data is not accessible")
        if len(shape) != 2:
            raise ValueError(f"data is not 2-D (channel, sample): {len(shape)} dimension(s)")
        return shape

    def data_access():
        held["shape"] = tuple(int(n) for n in np.shape(raw.get_data()))
        return {"shape": list(held["shape"])}, None

    def data_dimension():
        if "shape" not in held:
            raise ValueError("signal data is not accessible")
        ndim = len(held["shape"])
        return {"ndim": ndim}, None if ndim == 2 else f"expected 2 dimensions, found {ndim}"

    def non_empty():
        shape = shape_2d()
        return ({"n_channels": shape[0], "n_samples": shape[1]},
                None if min(shape) > 0 else "channel and sample dimensions must be > 0")

    def channel_shape():
        counts = {"data_channels": shape_2d()[0], "ch_names": len(raw.ch_names),
                  "channel_types": len(raw.get_channel_types()),
                  "snapshot_n_channels": snapshot["n_channels"]}
        return counts, None if len(set(counts.values())) == 1 else "channel counts differ"

    def sample_shape():
        counts = {"data_samples": shape_2d()[1], "n_times": int(raw.n_times),
                  "snapshot_n_times": snapshot["n_times"]}
        return counts, None if len(set(counts.values())) == 1 else "sample counts differ"

    def sampling():
        sfreq, saved = raw.info.get("sfreq"), snapshot["sfreq"]
        observed = {"sfreq": _plain(sfreq), "snapshot_sfreq": _plain(saved)}
        if sfreq is None:
            reason = "sfreq is missing"
        elif isinstance(sfreq, bool) or not isinstance(sfreq, Real):
            reason = "sfreq is not numeric"
        elif not math.isfinite(sfreq):
            reason = "sfreq is not finite"
        elif sfreq <= 0:
            reason = "sfreq must be > 0"
        elif sfreq != saved:
            reason = "sfreq differs from the load-time snapshot"
        else:
            reason = None
        return observed, reason

    def channel_metadata():
        names, types = list(raw.ch_names), list(raw.get_channel_types())
        duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
        empty = [i for i, n in enumerate(names) if not isinstance(n, str) or not n.strip()]
        observed = {"n_names": len(names), "n_types": len(types),
                    "snapshot_n_names": len(snapshot["channel_names"]),
                    "snapshot_n_types": len(snapshot["channel_types"]),
                    "empty_name_indices": empty, "duplicate_names": duplicates}
        reasons = []
        if len({len(names), len(types), len(snapshot["channel_names"]),
                len(snapshot["channel_types"])}) != 1:
            reasons.append("channel name/type counts differ")
        if empty:
            reasons.append("empty channel name")
        if duplicates:
            reasons.append("duplicate channel name")
        if names != snapshot["channel_names"]:
            reasons.append("channel names differ from the load-time snapshot")
        if types != snapshot["channel_types"]:
            reasons.append("channel types differ from the load-time snapshot")
        return observed, "; ".join(reasons) or None

    # data_access must run first: later checks read the shape it stores.
    checks = [
        check("DATA_ACCESS", "signal data can be read from the Raw object", data_access),
        check("DATA_DIMENSION", "data is 2-D (channel, sample)", data_dimension),
        check("NON_EMPTY_DIMENSIONS", "channels > 0 and samples > 0", non_empty),
        check("CHANNEL_SHAPE_CONSISTENCY",
              "data channels == ch_names == channel types == snapshot n_channels",
              channel_shape),
        check("SAMPLE_SHAPE_CONSISTENCY",
              "data samples == raw.n_times == snapshot n_times", sample_shape),
        check("SAMPLING_INFORMATION_VALIDITY",
              "sfreq present, numeric, finite, > 0 and equal to the snapshot", sampling),
        check("CHANNEL_METADATA_CONSISTENCY",
              "name/type counts agree, no empty or duplicate names, equal to the snapshot",
              channel_metadata),
    ]
    return make_result(VALIDATOR_NAME, VALIDATOR_VERSION, record, checks)

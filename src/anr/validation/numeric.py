"""Finite-value validator: NaN / +Inf / -Inf in the whole signal array."""

import numpy as np

from .common import check, make_result

VALIDATOR_NAME = "finite_values"
VALIDATOR_VERSION = "1.0.0"


def validate_finite(raw, record):
    def finite_values():
        data = np.asarray(raw.get_data())
        if data.dtype.kind not in "fiu":
            return ({"dtype": str(data.dtype), "total_values": int(data.size)},
                    "signal data is not real-valued numeric")
        observed = {"total_values": int(data.size),
                    "nan_count": int(np.count_nonzero(np.isnan(data))),
                    "positive_inf_count": int(np.count_nonzero(np.isposinf(data))),
                    "negative_inf_count": int(np.count_nonzero(np.isneginf(data)))}
        bad = (observed["nan_count"] + observed["positive_inf_count"]
               + observed["negative_inf_count"])
        reason = (f"{observed['nan_count']} NaN, {observed['positive_inf_count']} +Inf, "
                  f"{observed['negative_inf_count']} -Inf value(s) found") if bad else None
        return observed, reason

    checks = [check("FINITE_VALUES", "no NaN, +Inf or -Inf in the signal data", finite_values)]
    return make_result(VALIDATOR_NAME, VALIDATOR_VERSION, record, checks)

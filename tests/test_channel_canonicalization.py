"""Unit tests for anr.channel_canonicalization (synthetic signals)."""

import numpy as np
import pytest

from anr.channel_canonicalization import (
    CHANNEL_SCHEMA_VERSION, ChannelCanonicalizationError, canonicalize_channels, channel_schema_v1,
)

SCHEMA = channel_schema_v1()
NAMES = [s["source_channel_name"] for s in SCHEMA]
TYPES = [s["channel_type"] for s in SCHEMA]


def signals(n_channels=25, trials=3, samples=20):
    # Every channel has distinct values so a wrong mapping cannot go unnoticed.
    base = np.arange(trials * n_channels * samples, dtype=float)
    return base.reshape(trials, n_channels, samples)


def test_valid_22_eeg_3_eog_split_shapes_and_ids():
    out = canonicalize_channels(signals(), NAMES, TYPES)
    assert out["channel_schema_version"] == CHANNEL_SCHEMA_VERSION == "channel_schema_v1"
    assert out["X_eeg"].shape == (3, 22, 20) and out["X_eog"].shape == (3, 3, 20)
    ids = [c["canonical_channel_id"] for c in out["eeg_channels"] + out["eog_channels"]]
    assert len(ids) == len(set(ids)) == 25
    assert ids[:22] == [f"bciciv2a:eeg:{i:02d}" for i in range(1, 23)]
    assert ids[22:] == [f"bciciv2a:eog:{i:02d}" for i in range(1, 4)]


def test_source_name_and_type_preserved_and_no_electrode_names_invented():
    out = canonicalize_channels(signals(), NAMES, TYPES)
    for record in out["eeg_channels"]:
        assert record["channel_type"] == "eeg" and record["source_channel_name"].startswith("EEG-")
    for record in out["eog_channels"]:
        assert record["channel_type"] == "eog" and record["source_channel_name"].startswith("EOG-")
    assert [c["source_channel_name"] for c in out["eeg_channels"]] == NAMES[:22]
    assert not any("source_channel_name" in c and c["source_channel_name"] == c["canonical_channel_id"]
                   for c in out["eeg_channels"])


def test_schema_has_exactly_22_eeg_and_3_eog():
    assert [s["channel_type"] for s in SCHEMA].count("eeg") == 22
    assert [s["channel_type"] for s in SCHEMA].count("eog") == 3
    assert len(SCHEMA) == 25


def test_order_is_deterministic():
    first = canonicalize_channels(signals(), NAMES, TYPES)
    second = canonicalize_channels(signals(), NAMES, TYPES)
    assert first["eeg_channels"] == second["eeg_channels"] and first["eog_channels"] == second["eog_channels"]
    assert np.array_equal(first["X_eeg"], second["X_eeg"])


def test_reordered_valid_input_gives_canonical_order_and_exact_values():
    data = signals()
    order = np.random.default_rng(1).permutation(25)
    names, types = [NAMES[i] for i in order], [TYPES[i] for i in order]
    reordered = canonicalize_channels(data[:, order, :], names, types)
    reference = canonicalize_channels(data, NAMES, TYPES)
    assert np.array_equal(reordered["X_eeg"], reference["X_eeg"])
    assert np.array_equal(reordered["X_eog"], reference["X_eog"])
    assert [c["canonical_channel_id"] for c in reordered["eeg_channels"]] == \
        [c["canonical_channel_id"] for c in reference["eeg_channels"]]
    assert reordered["eeg_channels"][0]["source_channel_index"] == int(np.where(order == 0)[0][0])


def test_values_exactly_match_corresponding_source_channels_and_eog_retained():
    data = signals()
    out = canonicalize_channels(data, NAMES, TYPES)
    for k, record in enumerate(out["eeg_channels"]):
        assert np.array_equal(out["X_eeg"][:, k, :], data[:, record["source_channel_index"], :])
    for k, record in enumerate(out["eog_channels"]):
        assert np.array_equal(out["X_eog"][:, k, :], data[:, record["source_channel_index"], :])
    assert np.array_equal(out["X_eog"], data[:, 22:, :])  # EOG not discarded
    used = sorted(c["source_channel_index"] for c in out["eeg_channels"] + out["eog_channels"])
    assert used == list(range(25))  # every source channel represented exactly once
    assert out["X_eeg"].dtype == data.dtype


def test_source_not_modified():
    data = signals()
    before = data.copy()
    canonicalize_channels(data, NAMES, TYPES)
    assert np.array_equal(data, before)


def test_missing_eeg_fails():
    keep = [i for i in range(25) if NAMES[i] != "EEG-Cz"]
    with pytest.raises(ChannelCanonicalizationError, match="missing required channel.*EEG-Cz"):
        canonicalize_channels(signals()[:, keep, :], [NAMES[i] for i in keep], [TYPES[i] for i in keep])


def test_missing_eog_fails():
    with pytest.raises(ChannelCanonicalizationError, match="missing required channel.*EOG-right"):
        canonicalize_channels(signals(24), NAMES[:24], TYPES[:24])


def test_duplicate_identity_fails():
    names = NAMES[:24] + [NAMES[0]]
    with pytest.raises(ChannelCanonicalizationError, match="duplicate identity.*EEG-Fz"):
        canonicalize_channels(signals(), names, TYPES)


def test_ambiguous_schema_fails():
    ambiguous = channel_schema_v1()
    ambiguous[1]["source_channel_name"] = ambiguous[0]["source_channel_name"]
    with pytest.raises(ChannelCanonicalizationError, match="Ambiguous"):
        canonicalize_channels(signals(), NAMES, TYPES, schema=ambiguous)


@pytest.mark.parametrize("swap", [("EEG-Fz", "eog"), ("EOG-left", "eeg")])
def test_type_mismatch_fails(swap):
    types = list(TYPES)
    types[NAMES.index(swap[0])] = swap[1]
    with pytest.raises(ChannelCanonicalizationError, match="type mismatch"):
        canonicalize_channels(signals(), NAMES, types)


def test_extra_channel_fails():
    with pytest.raises(ChannelCanonicalizationError, match="extra channel"):
        canonicalize_channels(signals(26), NAMES + ["EEG-extra"], TYPES + ["eeg"])


@pytest.mark.parametrize("alias", ["Fz", "eeg-fz", "EEG-Fz "])
def test_unvalidated_alias_fails(alias):
    names = list(NAMES)
    names[0] = alias
    with pytest.raises(ChannelCanonicalizationError, match="unvalidated alias"):
        canonicalize_channels(signals(), names, TYPES)


def test_shape_inconsistencies_fail():
    with pytest.raises(ChannelCanonicalizationError):
        canonicalize_channels(signals(24), NAMES, TYPES)  # 24 data channels, 25 names
    with pytest.raises(ChannelCanonicalizationError):
        canonicalize_channels(signals()[0], NAMES, TYPES)  # not [trials, channels, samples]

"""Minimal A01T.gdf loader with provenance, limited to ANR-T001 v1.2."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from uuid import uuid4
import warnings

from .provenance import (
    Evidence, ProcessingStep, assess, file_sha256, validate_record, validate_sha256,
)

OFFICIAL_DOCUMENT = "https://www.bbci.de/competition/iv/desc_2a.pdf"
OFFICIAL_SOURCE = "https://www.bbci.de/competition/iv/"
MNE_VERSION = "1.11.0"
MNE_SOURCE = "https://github.com/mne-tools/mne-python/blob/v1.11.0/mne/io/edf/edf.py"


@dataclass(frozen=True)
class SourceReference:
    """Caller-supplied trusted acquisition record, NOT the freshly computed hash.

    A local hash alone cannot authenticate an official download. Supply this
    only from an independently verified official acquisition record; its URL
    and verification note remain visible in JSON. ANR does not certify it.
    """

    sha256: str
    url: str
    verification: str

    def __post_init__(self):
        validate_sha256(self.sha256)
        if not self.url.startswith("https://") or not self.verification.strip():
            raise ValueError("Source reference needs HTTPS URL and verification note")


def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _audit_loader(raw, *, preload, version):
    """Inspect the pinned MNE header contract without reading/altering arrays.

    Private header access is isolated here. Missing or changed metadata makes
    the audit incomplete instead of silently concluding that nothing happened.
    """
    history, details = [], {}
    if version != MNE_VERSION:
        return history, details, False
    try:
        header = raw._raw_extras[0]
        decoded = bool(preload and raw.preload)
        selected = [int(i) for i in header["sel"]]
        if selected != list(range(25)):
            return history, details, False
        samples = [int(header["n_samps"][i]) for i in selected]
        maximum = int(header["max_samp"])
        if maximum <= 0 or any(s <= 0 or s > maximum for s in samples):
            return history, details, False
        details["samples_per_record"] = samples
        details["max_samples_per_record"] = maximum
        for key in ("cal", "offsets", "units"):
            values = [_number(v) for v in header[key]]
            if len(values) != 25 or any(v is None for v in values):
                return history, details, False
            details[key] = values
        history.append(ProcessingStep(
            "loader", "Digital-to-physical calibration, offset and SI unit conversion",
            "representation", decoded, MNE_SOURCE))
        history.append(ProcessingStep(
            "loader", "GDF events decoded to annotations; channels 22-24 typed EOG",
            "representation", True, MNE_SOURCE))
        if any(s != maximum for s in samples):
            history.append(ProcessingStep(
                "loader", "Implicit resampling of channels with fewer samples per record",
                "signal_processing", decoded, MNE_SOURCE))
        # Reading header metadata does not apply these acquisition filters.
        for key in ("highpass", "lowpass", "notch"):
            values = header.get(key)
            if values is not None:
                details[key] = [_number(v) for v in values]
        # A zero/negative header cutoff is retained but its semantics are not
        # assumed to mean either "missing" or "disabled". Require review.
        ambiguous_filter = any(v is not None and v <= 0
                               for key in ("highpass", "lowpass", "notch")
                               for v in details.get(key, []))
        if ambiguous_filter:
            details["unresolved_filter_encoding"] = True
            return history, details, False
        return history, details, bool(preload and raw.preload)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError):
        return history, details, False


def load_a01t(path, *, source_reference=None, preload=True,
              evidence=(), processing_history=(), reader=None, reader_version=None):
    """Return (Raw, JSON-ready provenance) without filtering or repairing.

    reader is a Gate A injection seam. Its results are labelled mock_reader,
    with mne_version=None; they are never actual GDF integration evidence.
    processing_history records caller-confirmed prior operations; it never
    applies processing. Provenance covers this load, not later Raw mutations.
    """
    path = Path(path).resolve()
    if path.name != "A01T.gdf":
        raise ValueError("T001 supports only A01T.gdf")
    if type(preload) is not bool:
        raise ValueError("preload must be boolean (not a memory-map output path)")
    if source_reference is not None and not isinstance(source_reference, SourceReference):
        raise TypeError("source_reference must be SourceReference or None")
    claims, history = list(evidence), list(processing_history)
    if any(not isinstance(c, Evidence) for c in claims):
        raise TypeError("evidence must contain Evidence objects")
    if any(not isinstance(s, ProcessingStep) for s in history):
        raise TypeError("processing_history must contain ProcessingStep objects")
    # Caller claims may challenge a built-in assertion, but cannot fabricate
    # the two positive prerequisites used by assess() to grant RAW.
    if any(c.stage == stage and c.item == item
           for c in claims for stage, item in
           (("source", "official_reference_match"), ("loader", "audit_complete"))):
        raise ValueError("Reserved evidence item; use source_reference or reader metadata")
    digest = file_sha256(path)  # Missing/unreadable files fail before invoking MNE.
    if reader is None:
        import mne
        reader = mne.io.read_raw_gdf
        version = mne.__version__
        mne_version = version
        scope = "actual_gdf"
    else:
        if not reader_version:
            raise ValueError("Injected reader requires its audited contract version")
        version, mne_version, scope = reader_version, None, "mock_reader"
    options = {"eog": [22, 23, 24], "misc": None, "stim_channel": None,
               "exclude": [], "include": None, "preload": preload, "verbose": "ERROR"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = reader(str(path), **options)
    try:
        if file_sha256(path) != digest:
            raise ValueError("Source file changed during loading")
        metadata = {"sfreq": _number(raw.info.get("sfreq")),
                    "n_channels": len(raw.ch_names), "channel_names": list(raw.ch_names),
                    "channel_types": list(raw.get_channel_types()),
                    "n_times": int(raw.n_times),
                    "annotation_count": len(raw.annotations),
                    "returned_highpass": _number(raw.info.get("highpass")),
                    "returned_lowpass": _number(raw.info.get("lowpass"))}
        if metadata["n_times"] <= 0:
            raise ValueError("Reader returned no samples")
        loader_history, header, audited = _audit_loader(raw, preload=preload, version=version)
        history.extend(loader_history)
        expected_types = ["eeg"] * 22 + ["eog"] * 3
        audited = (audited and metadata["channel_types"] == expected_types
                   and metadata["sfreq"] is not None)
        # Warnings are retained, but are not invented contradictions. A warning
        # leaves the audit incomplete pending review of the exact message.
        audited = audited and not caught
        claims.extend([
            Evidence("source", "official_reference_match",
                     None if source_reference is None else digest == source_reference.sha256,
                     "SHA-256 comparison with caller-supplied trusted reference", "direct"),
            Evidence("loader", "audit_complete", bool(audited), MNE_SOURCE, "direct"),
        ])
        if source_reference is not None:
            claims.extend([
                Evidence("source", "sha256", source_reference.sha256,
                         source_reference.url + " | " + source_reference.verification, "reported"),
                Evidence("source", "sha256", digest, str(path), "direct"),
            ])
        for item, expected in (("sfreq", 250.0), ("n_channels", 25)):
            claims.extend([
                Evidence("source", item, expected, OFFICIAL_DOCUMENT, "document"),
                Evidence("source", item, metadata[item], "Returned reader metadata", "metadata"),
            ])
        acquisition = []
        for item, key, expected in (("highpass_hz", "highpass", 0.5),
                                    ("lowpass_hz", "lowpass", 100.0),
                                    ("notch_hz", "notch", 50.0)):
            acquisition.append({"item": item, "value": expected, "source": OFFICIAL_DOCUMENT,
                                "basis": "official_document; not a signal measurement"})
            claims.append(Evidence("acquisition", item, expected, OFFICIAL_DOCUMENT, "document"))
            # raw.info filter summaries can contain defaults; do not use them
            # as explicit acquisition claims. Inspect original header instead.
            values = header.get(key, [])
            explicit = sorted({v for v in values if v is not None and v > 0})
            if explicit:
                for value in explicit:
                    claims.append(Evidence("acquisition", item, value,
                                           "GDF header via MNE " + version, "metadata"))
            else:
                claims.append(Evidence("acquisition", item, None,
                                       "No explicit usable GDF header value", "metadata"))
        artifact_id = str(uuid4())
        record = {
            "schema_version": 1, "dataset": "BCIC IV 2a/A01T",
            "artifact_id": artifact_id, "source_sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation_scope": scope,
            "source": {"path": str(path), "sha256": digest, "artifact_id": artifact_id,
                       "official_source": OFFICIAL_SOURCE,
                       "reference": None if source_reference is None else asdict(source_reference)},
            "loader": {"name": "mne.io.read_raw_gdf" if scope == "actual_gdf" else "injected_reader",
                       "mne_version": mne_version, "reader_contract_version": version,
                       "options": options, "header_audit": header,
                       "warnings": [str(w.message) for w in caught],
                       "preload_meaning": "Memory loading only; does not preserve original digital scale"},
            "metadata": metadata, "acquisition_history": acquisition,
            "processing_history": [asdict(s) for s in history],
            "evidence": [asdict(c) for c in claims],
            **assess(claims, history),
        }
        validate_record(record)
        return raw, record
    except Exception:
        raw.close()
        raise

"""ANR-T001 evidence assessment and JSON records (no signal processing)."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from uuid import UUID


STAGES = {"source", "acquisition", "loader", "post_distribution"}
KINDS = {"document", "metadata", "direct", "reported"}


@dataclass(frozen=True)
class Evidence:
    """A claim about one stage/item; None explicitly means information absent."""

    stage: str
    item: str
    value: str | float | int | bool | None
    source: str
    kind: str

    def __post_init__(self):
        if self.stage not in STAGES or self.kind not in KINDS:
            raise ValueError("Invalid evidence stage or kind")
        if not isinstance(self.item, str) or not isinstance(self.source, str) or not self.item or not self.source:
            raise ValueError("Evidence requires item and source")
        if self.value is not None and type(self.value) not in (str, float, int, bool):
            raise ValueError("Evidence value must be a JSON scalar or None")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Non-finite evidence is not supported; use None")


@dataclass(frozen=True)
class ProcessingStep:
    stage: str
    operation: str
    category: str  # representation or signal_processing
    confirmed: bool
    source: str

    def __post_init__(self):
        if self.stage not in {"loader", "post_distribution"}:
            raise ValueError("Acquisition belongs in acquisition_history")
        if self.category not in {"representation", "signal_processing"}:
            raise ValueError("Invalid processing category")
        if (type(self.confirmed) is not bool or not isinstance(self.operation, str)
                or not isinstance(self.source, str) or not self.operation or not self.source):
            raise ValueError("Processing step needs a boolean confirmation and source")


def file_sha256(path):
    """Hash source bytes in bounded chunks; this is not proof of authenticity."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters")
    return value


def assess(evidence, processing_history):
    """Compare explicit same-stage/item claims; missing values never conflict.

    DIRECT describes the basis of the decision, not scientific certainty.
    UNKNOWN uses INSUFFICIENT and retains every supporting/conflicting claim.
    RAW needs affirmative source and loader evidence, not merely an empty log.
    """
    evidence = list(evidence)
    processing_history = list(processing_history)
    conflicts = []
    groups = {}
    for index, claim in enumerate(evidence):
        if claim.value is not None:
            groups.setdefault((claim.stage, claim.item), []).append((index, claim))
    for (stage, item), claims in groups.items():
        values = {json.dumps(float(c.value) if type(c.value) in (int, float) else c.value,
                             sort_keys=True, allow_nan=False) for _, c in claims}
        if len(values) > 1:
            conflicts.append({"stage": stage, "item": item,
                              "evidence_indices": [i for i, _ in claims]})
    for stage in ("loader", "post_distribution"):
        steps = [i for i, s in enumerate(processing_history)
                 if s.stage == stage and s.confirmed and s.category == "signal_processing"]
        denials = [i for i, c in enumerate(evidence)
                   if c.stage == stage and c.item == "additional_signal_processing"
                   and c.value is False]
        if steps and denials:
            conflicts.append({"stage": stage, "item": "additional_signal_processing",
                              "evidence_indices": denials, "processing_history_indices": steps})

    # A recorded, confirmed operation is direct positive evidence even if other
    # claims disagree. This function does not execute that operation.
    processed = any(s.confirmed and s.category == "signal_processing"
                    for s in processing_history)
    processed |= any(c.stage in {"loader", "post_distribution"}
                     and c.item == "additional_signal_processing"
                     and c.value is True and c.kind == "direct" for c in evidence)

    def direct_true(stage, item):
        return any(c.stage == stage and c.item == item and c.value is True
                   and c.kind == "direct" for c in evidence)

    missing = []
    if not direct_true("source", "official_reference_match"):
        missing.append("Official source identity has not been established")
    if not direct_true("loader", "audit_complete"):
        missing.append("Loader transformations have not been fully checked")
    if any(not s.confirmed for s in processing_history):
        missing.append("Processing history contains an unconfirmed operation")
    if any(c.stage in {"loader", "post_distribution"}
           and c.item == "additional_signal_processing" and c.value is True
           and c.kind != "direct" for c in evidence) and not processed:
        missing.append("Additional processing is reported but not directly confirmed")

    if processed:
        status, level = "PROCESSED", "DIRECT"
    elif conflicts or missing:
        status, level = "UNKNOWN", "INSUFFICIENT"
    else:
        status, level = "RAW", "DIRECT"
    return {"processing_status": status, "evidence_level": level,
            "review_required": bool(conflicts or missing), "conflicts": conflicts,
            "unknowns": missing}


def validate_record(record):
    """Check persisted linkage and recompute decisions; JSON is not a signature."""
    try:
        if (type(record["schema_version"]) is not int or record["schema_version"] != 1
                or record["dataset"] != "BCIC IV 2a/A01T"):
            raise ValueError("Unsupported provenance record")
        UUID(record["artifact_id"])
        digest = validate_sha256(record["source_sha256"])
        if record["source"]["sha256"] != digest:
            raise ValueError("Inconsistent source hash linkage")
        if Path(record["source"]["path"]).name != "A01T.gdf":
            raise ValueError("Record is not for A01T.gdf")
        if record["source"]["artifact_id"] != record["artifact_id"]:
            raise ValueError("Inconsistent artifact linkage")
        claims = [Evidence(**c) for c in record["evidence"]]
        history = [ProcessingStep(**s) for s in record["processing_history"]]
        if not isinstance(record["acquisition_history"], list):
            raise ValueError("Missing acquisition history")
        if not isinstance(record["loader"], dict) or not isinstance(record["metadata"], dict):
            raise ValueError("Invalid loader/metadata record")
        loader = record["loader"]
        if not isinstance(loader["options"], dict) or not isinstance(loader["warnings"], list):
            raise ValueError("Missing loader options or warnings")
        if record["validation_scope"] == "mock_reader":
            if loader["mne_version"] is not None or loader["name"] != "injected_reader":
                raise ValueError("Mock results must not claim an actual MNE version")
        elif record["validation_scope"] == "actual_gdf":
            if not isinstance(loader["mne_version"], str) or loader["name"] != "mne.io.read_raw_gdf":
                raise ValueError("Actual GDF record requires an MNE version")
        else:
            raise ValueError("Invalid validation scope")
        reference = record["source"]["reference"]
        match = None
        if reference is not None:
            match = validate_sha256(reference["sha256"]) == digest
            if not reference["url"].startswith("https://") or not reference["verification"].strip():
                raise ValueError("Incomplete trusted source reference")
        matches = [c for c in claims if c.stage == "source" and c.item == "official_reference_match"]
        if len(matches) != 1 or matches[0].value is not match or matches[0].kind != "direct":
            raise ValueError("Source reference and match evidence disagree")
        expected = assess(claims, history)
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Persisted assessment does not match its evidence")
        # Also reject values JSON cannot represent faithfully.
        json.dumps(record, allow_nan=False)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed provenance record") from exc
    return record


def save_record(record, path):
    """Save only provenance/metadata, never a Raw object or a signal array."""
    validate_record(record)
    path = Path(path)
    if path.resolve() == Path(record["source"]["path"]).resolve():
        raise ValueError("JSON output must not overwrite the source GDF")
    # Exclusive creation protects an existing artifact and its source.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_record(path, *, source_path=None):
    """Reload JSON; when supplied, verify source bytes against the saved hash.

    source_path supports a relocated file. Without it only record consistency
    is checked; no claim is made that the source file still exists or matches.
    """
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream, parse_constant=lambda value: _invalid_json(value))
    validate_record(record)
    if source_path is not None:
        if file_sha256(source_path) != record["source_sha256"]:
            raise ValueError("Source file SHA-256 does not match the saved record")
    return record


def _invalid_json(value):
    raise ValueError(f"Non-finite JSON value: {value}")

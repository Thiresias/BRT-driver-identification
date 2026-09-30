"""Validated video manifests, independent of PyTorch and video decoding."""

import csv
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


REQUIRED_FIELDS = (
    "clip_id", "path", "kind", "split",
)
SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class VideoRecord:
    clip_id: str
    path: Path
    kind: str
    driver_id: str
    appearance_id: str
    driver_video_id: str
    split: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    source_asset_id: str = ""
    method: str = ""
    driver_scope: str = "named"
    poi_id: str = ""

    @property
    def recording_key(self):
        if not self.driver_video_id:
            return None
        if self.driver_scope == "non_poi":
            return ("non_poi", self.poi_id, self.driver_video_id)
        return ("named", self.driver_id, self.driver_video_id)

    @property
    def group_key(self):
        if self.kind == "genuine" and self.recording_key is not None:
            return (self.kind,) + self.recording_key
        return (self.kind, self.clip_id)

    def label_for(self, poi):
        if self.driver_scope == "non_poi":
            if poi != self.poi_id:
                raise ValueError(f"world records are scoped to POI {self.poi_id!r}, not {poi!r}")
            return 0
        return int(self.driver_id == poi)


@dataclass(frozen=True)
class VideoGroup:
    video_id: str
    records: Tuple[VideoRecord, ...]
    label: int


def _text(row, field, required=False):
    value = row.get(field)
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string (quote numeric identifiers in JSON)")
    value = value.strip()
    if required and not value:
        raise ValueError(f"missing required field: {field}")
    return value


def _record(row, base):
    if not isinstance(row, dict):
        raise ValueError("each record must be an object")
    values = {field: _text(row, field, required=True) for field in REQUIRED_FIELDS}
    values.update({field: _text(row, field) for field in
                   ("driver_id", "appearance_id", "driver_video_id", "poi_id")})
    values["driver_scope"] = _text(row, "driver_scope") or "named"
    if values["driver_scope"] == "named":
        if not values["driver_id"]:
            raise ValueError("named records require driver_id")
        if values["poi_id"]:
            raise ValueError("poi_id is only used with driver_scope=non_poi")
    elif values["driver_scope"] == "non_poi":
        if values["driver_id"] or not values["poi_id"]:
            raise ValueError("non_poi records require poi_id and an empty driver_id")
    else:
        raise ValueError("driver_scope must be named or non_poi")
    if values["kind"] not in {"genuine", "generated"}:
        raise ValueError("kind must be genuine or generated")
    if values["split"] not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    if (values["kind"] == "genuine" and values["appearance_id"]
            and values["driver_id"] != values["appearance_id"]):
        raise ValueError("genuine videos must have matching driver_id and appearance_id")

    start, end = row.get("start_sec"), row.get("end_sec")
    def missing(value):
        return value is None or (isinstance(value, str) and value.strip().lower() in {"", "nan"}) or (
            isinstance(value, float) and math.isnan(value))
    if missing(start) and missing(end):
        start = end = None
    else:
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            raise ValueError("provide both start_sec and end_sec as numbers") from None
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("timestamps must be finite and satisfy 0 <= start_sec < end_sec")

    path = Path(values.pop("path")).expanduser()
    if not path.is_absolute():
        path = base / path
    return VideoRecord(
        path=path.resolve(), start_sec=start, end_sec=end, **values,
        source_asset_id=_text(row, "source_asset_id"), method=_text(row, "method"),
    )


def load_metadata(manifests):
    """Read CSV or a JSON list; relative media paths use each manifest's directory.

    Files need not exist until the requested dataset split is constructed. This
    permits evaluation using a manifest whose training media is stored elsewhere.
    Timestamps describe provenance in the original recording, not crop offsets.
    """
    if isinstance(manifests, (str, Path)):
        manifests = [manifests]
    records, ids, paths, recording_splits = [], set(), set(), {}
    for manifest in manifests:
        manifest = Path(manifest).expanduser().resolve()
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            if manifest.suffix.lower() == ".csv":
                reader = csv.DictReader(handle)
                missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{manifest}: missing columns: {', '.join(sorted(missing))}")
                rows = list(reader)
            elif manifest.suffix.lower() == ".json":
                rows = json.load(handle)
                if not isinstance(rows, list):
                    raise ValueError(f"{manifest}: JSON must contain a list of records")
            else:
                raise ValueError(f"{manifest}: expected a .csv or .json manifest")
        for index, row in enumerate(rows, 1):
            try:
                record = _record(row, manifest.parent)
                if record.clip_id in ids:
                    raise ValueError(f"duplicate clip_id: {record.clip_id}")
                if record.path in paths:
                    raise ValueError(f"duplicate media path: {record.path}")
                old_split = recording_splits.get(record.recording_key) if record.recording_key else None
                if old_split is not None and old_split != record.split:
                    raise ValueError(
                        f"driver recording {record.recording_key} appears in both "
                        f"{old_split} and {record.split}"
                    )
            except ValueError as exc:
                raise ValueError(f"{manifest}, record {index}: {exc}") from exc
            ids.add(record.clip_id)
            paths.add(record.path)
            if record.recording_key is not None:
                recording_splits[record.recording_key] = record.split
            records.append(record)
    if not records:
        raise ValueError("metadata contains no records")
    unknown = sum(record.recording_key is None for record in records)
    if unknown:
        warnings.warn(f"{unknown} record(s) lack driver_video_id; recording-overlap checks "
                      "are incomplete for these assets", UserWarning, stacklevel=2)
    return records


def group_records(records, split, poi):
    """Group genuine recordings and retain each generated asset separately."""
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if not isinstance(poi, str) or not poi.strip():
        raise ValueError("poi must be a non-empty driver identity")
    # Check scope even for records outside the selected split.
    for record in records:
        record.label_for(poi)
    grouped = defaultdict(list)
    for record in records:
        if record.split == split:
            grouped[record.group_key].append(record)
    if not grouped:
        raise ValueError(f"metadata has no records for split {split!r}")
    groups = [
        VideoGroup(
            video_id=json.dumps(key, ensure_ascii=False, separators=(",", ":")),
            records=tuple(sorted(items, key=lambda r: r.clip_id)),
            label=items[0].label_for(poi),
        )
        for key, items in sorted(grouped.items())
    ]
    if {group.label for group in groups} != {0, 1}:
        raise ValueError(
            f"split {split!r} must contain both POI ({poi!r}) and non-POI drivers "
            "for binary training/evaluation; identity matching is case-sensitive"
        )
    return groups


def main():
    """Validate metadata without importing PyTorch or requiring video files."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument("--poi", required=True)
    args = parser.parse_args()
    try:
        records = load_metadata(args.metadata)
        summary = {}
        for split in sorted({record.split for record in records}):
            groups = group_records(records, split, args.poi)
            summary[split] = {
                "clips": sum(len(group.records) for group in groups),
                "groups": len(groups),
                "positive_groups": sum(group.label for group in groups),
                "negative_groups": sum(1 - group.label for group in groups),
            }
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

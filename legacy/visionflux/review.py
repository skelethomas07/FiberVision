from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import math
import uuid
import zipfile

import numpy as np
import pandas as pd

_ACTIVE_STATUSES = {"active", "accepted"}
_DELETE_ALL_AUTO_TOKEN = "__VISIONFLUX_DELETE_ALL_AUTO__"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distance_measurement(
    p1: tuple[float, float],
    p2: tuple[float, float],
    analysis_scale: float = 1.0,
    nm_per_px: float | None = None,
) -> dict[str, float | None]:
    """Measure a line in the displayed analysis image and convert to original units."""
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    analysis_width = float(math.hypot(x2 - x1, y2 - y1))
    original_width = analysis_width / float(analysis_scale)
    width_nm = None if nm_per_px is None else original_width * float(nm_per_px)
    return {
        "analysis_width_px": analysis_width,
        "original_width_px": original_width,
        "width_nm": width_nm,
    }


def _find_row(df: pd.DataFrame, measurement_id: str) -> int:
    matches = df.index[df["measurement_id"].astype(str) == str(measurement_id)].tolist()
    if len(matches) != 1:
        raise KeyError(f"measurement_id must identify exactly one row: {measurement_id}")
    return int(matches[0])


def reject_measurement(
    measurements: pd.DataFrame,
    measurement_id: str,
    reason: str = "rejected",
) -> tuple[pd.DataFrame, dict]:
    updated = measurements.copy(deep=True)
    idx = _find_row(updated, measurement_id)
    old_status = str(updated.at[idx, "status"])
    updated.at[idx, "status"] = "rejected"
    updated.at[idx, "review_label"] = reason
    event = {
        "timestamp": _utc_now(),
        "action": reason,
        "measurement_id": str(measurement_id),
        "old_status": old_status,
        "new_status": "rejected",
    }
    return updated, event


def accept_measurement(
    measurements: pd.DataFrame,
    measurement_id: str,
) -> tuple[pd.DataFrame, dict]:
    updated = measurements.copy(deep=True)
    idx = _find_row(updated, measurement_id)
    updated.at[idx, "status"] = "accepted"
    updated.at[idx, "review_label"] = "accepted"
    return updated, {
        "timestamp": _utc_now(),
        "action": "accepted",
        "measurement_id": str(measurement_id),
    }


def replace_with_manual(
    measurements: pd.DataFrame,
    measurement_id: str,
    p1: tuple[float, float],
    p2: tuple[float, float],
    analysis_scale: float = 1.0,
    nm_per_px: float | None = None,
) -> tuple[pd.DataFrame, dict, str]:
    updated = measurements.copy(deep=True)
    idx = _find_row(updated, measurement_id)
    old = updated.loc[idx].to_dict()
    metrics = distance_measurement(p1, p2, analysis_scale, nm_per_px)
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    new_id = f"manual-{uuid.uuid4().hex[:12]}"

    updated.at[idx, "status"] = "corrected"
    updated.at[idx, "review_label"] = "manual_replaced"

    new_row = dict(old)
    new_row.update(
        measurement_id=new_id,
        source="manual",
        status="active",
        review_label="manual",
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=0.5 * (x1 + x2),
        center_y=0.5 * (y1 + y2),
        xm=0.5 * (x1 + x2),
        ym=0.5 * (y1 + y2),
        width_px=metrics["analysis_width_px"],
        width_original_px=metrics["original_width_px"],
        width_nm=np.nan if metrics["width_nm"] is None else metrics["width_nm"],
        grade="USER",
        confidence=1.0,
        sem_agreement=1.0,
    )
    updated = pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)
    event = {
        "timestamp": _utc_now(),
        "action": "manual_replace",
        "measurement_id": str(measurement_id),
        "replacement_measurement_id": new_id,
        "fiber_region_id": old.get("fiber_region_id"),
        "old_width_px": float(old.get("width_px", np.nan)),
        "new_width_px": float(metrics["analysis_width_px"]),
        "new_width_original_px": float(metrics["original_width_px"]),
        "new_width_nm": metrics["width_nm"],
        "p1": [x1, y1],
        "p2": [x2, y2],
    }
    return updated, event, new_id


def _next_region_id(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return int(finite.max()) + 1 if len(finite) else 1


def split_region_at_measurement(
    measurements: pd.DataFrame,
    measurement_id: str,
) -> tuple[pd.DataFrame, dict, int]:
    updated = measurements.copy(deep=True)
    idx = _find_row(updated, measurement_id)
    row = updated.loc[idx]
    old_region = row["fiber_region_id"]
    split_order = float(row.get("region_sample_index", 0))
    new_region = _next_region_id(updated["fiber_region_id"])
    same_region = updated["fiber_region_id"].astype(str) == str(old_region)
    order = pd.to_numeric(updated.get("region_sample_index", 0), errors="coerce").fillna(0)
    move = same_region & (order >= split_order)
    updated.loc[move, "fiber_region_id"] = new_region
    event = {
        "timestamp": _utc_now(),
        "action": "split_region",
        "measurement_id": str(measurement_id),
        "old_region_id": old_region,
        "new_region_id": new_region,
        "split_region_sample_index": split_order,
        "moved_measurements": int(move.sum()),
    }
    return updated, event, new_region


def _segment_cost(values: np.ndarray, i: int, j: int) -> float:
    segment = values[i:j]
    if not len(segment):
        return float("inf")
    median = float(np.median(segment))
    return float(np.sum((segment - median) ** 2))


def _contiguous_segments(
    widths: np.ndarray,
    max_segments: int = 3,
    min_segment_samples: int = 3,
    split_rel_range: float = 0.30,
    min_improvement: float = 0.22,
    min_rep_difference_frac: float = 0.14,
) -> list[tuple[int, int]]:
    n = len(widths)
    if n < max(2 * min_segment_samples, 8):
        return [(0, n)]
    p10, p50, p90 = np.percentile(widths, [10, 50, 90])
    if (p90 - p10) / max(float(p50), 1e-9) < split_rel_range:
        return [(0, n)]

    one_cost = _segment_cost(widths, 0, n)
    if one_cost <= 1e-12:
        return [(0, n)]
    best_segments = [(0, n)]
    best_penalized = one_cost
    max_k = min(max_segments, n // min_segment_samples)
    for k in range(2, max_k + 1):
        dp = np.full((k + 1, n + 1), np.inf)
        prev = np.full((k + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0.0
        for kk in range(1, k + 1):
            for j in range(kk * min_segment_samples, n + 1):
                start_min = (kk - 1) * min_segment_samples
                start_max = j - min_segment_samples
                for i in range(start_min, start_max + 1):
                    candidate = dp[kk - 1, i] + _segment_cost(widths, i, j)
                    if candidate < dp[kk, j]:
                        dp[kk, j] = candidate
                        prev[kk, j] = i
        raw_cost = float(dp[k, n])
        if not np.isfinite(raw_cost):
            continue
        cuts: list[tuple[int, int]] = []
        j = n
        for kk in range(k, 0, -1):
            i = int(prev[kk, j])
            if i < 0:
                cuts = []
                break
            cuts.append((i, j))
            j = i
        cuts.reverse()
        if not cuts:
            continue
        reps = [float(np.median(widths[i:j])) for i, j in cuts]
        distinct = all(
            abs(a - b) >= min_rep_difference_frac * max(0.5 * (a + b), 1e-9)
            for a, b in zip(reps[:-1], reps[1:])
        )
        improvement = (one_cost - raw_cost) / one_cost
        penalized = raw_cost + 0.11 * one_cost * (k - 1)
        if distinct and improvement >= min_improvement and penalized < best_penalized:
            best_penalized = penalized
            best_segments = cuts
    return best_segments


def _ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    if "region_sample_index" in group.columns:
        order = pd.to_numeric(group["region_sample_index"], errors="coerce")
        if order.notna().any():
            return group.assign(_order=order.fillna(order.max() + 1)).sort_values("_order")
    return group.sort_values(["center_x", "center_y"], na_position="last")


def _point_weights(group: pd.DataFrame) -> np.ndarray:
    n = len(group)
    if n == 1:
        return np.ones(1, float)
    x = pd.to_numeric(group.get("center_x", pd.Series(np.arange(n))), errors="coerce").to_numpy(float)
    y = pd.to_numeric(group.get("center_y", pd.Series(np.zeros(n))), errors="coerce").to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.ones(n, float)
    ds = np.hypot(np.diff(x), np.diff(y))
    positive = ds[ds > 1e-6]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    ds = np.where(ds > 1e-6, ds, fallback)
    return np.r_[0.5 * ds[0], 0.5 * (ds[:-1] + ds[1:]), 0.5 * ds[-1]]


def recompute_representatives(
    measurements: pd.DataFrame,
    analysis_scale: float = 1.0,
    nm_per_px: float | None = None,
) -> pd.DataFrame:
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    if measurements.empty:
        return pd.DataFrame()
    active = measurements[measurements["status"].astype(str).isin(_ACTIVE_STATUSES)].copy()
    active["width_px"] = pd.to_numeric(active["width_px"], errors="coerce")
    active = active[np.isfinite(active["width_px"]) & (active["width_px"] > 0)]
    rows: list[dict] = []
    for region_id, raw_group in active.groupby("fiber_region_id", sort=False, dropna=False):
        group = _ordered_group(raw_group).reset_index(drop=True)
        widths = group["width_px"].to_numpy(float)
        segments = _contiguous_segments(widths)
        pweights = _point_weights(group)
        total = float(pweights.sum()) if pweights.sum() > 0 else float(len(group))
        for subregion_id, (i, j) in enumerate(segments, start=1):
            vals = widths[i:j]
            weight = float(pweights[i:j].sum() / total)
            rep_analysis = float(np.median(vals))
            rep_original = rep_analysis / analysis_scale
            rows.append({
                "fiber_region_id": region_id,
                "subregion_id": subregion_id,
                "representative_width_px": rep_analysis,
                "representative_width_original_px": rep_original,
                "representative_width_nm": np.nan if nm_per_px is None else rep_original * nm_per_px,
                "fiber_count_weight": weight,
                "region_length_fraction": weight,
                "n_local_samples": int(j - i),
                "start_region_sample_index": group.iloc[i].get("region_sample_index", i),
                "end_region_sample_index": group.iloc[j - 1].get("region_sample_index", j - 1),
                "min_width_px": float(np.min(vals)),
                "max_width_px": float(np.max(vals)),
                "p10_width_px": float(np.percentile(vals, 10)),
                "p90_width_px": float(np.percentile(vals, 90)),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    totals = result.groupby("fiber_region_id", sort=False)["fiber_count_weight"].transform("sum")
    result["fiber_count_weight"] = result["fiber_count_weight"] / totals.replace(0, 1)
    result["region_length_fraction"] = result["fiber_count_weight"]
    return result.reset_index(drop=True)


def build_session_zip(
    image_name: str,
    measurements: pd.DataFrame,
    representatives: pd.DataFrame,
    feedback: list[dict],
    analysis_summary: dict | None = None,
    *,
    measurement_table: pd.DataFrame | None = None,
    imagej_results: pd.DataFrame | None = None,
    direction_table: pd.DataFrame | None = None,
    annotated_png: bytes | None = None,
    annotated_unlabeled_png: bytes | None = None,
    unit_metadata: dict | None = None,
) -> bytes:
    stem = image_name.rsplit(".", 1)[0]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{stem}_corrected_measurements.csv",
            measurements.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            f"{stem}_region_representatives.csv",
            representatives.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            f"{stem}_feedback.json",
            json.dumps(feedback, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )
        if analysis_summary is not None:
            archive.writestr(
                f"{stem}_analysis_summary.json",
                json.dumps(analysis_summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            )
        if measurement_table is not None:
            archive.writestr(
                f"{stem}_measurements.csv",
                measurement_table.to_csv(index=False).encode("utf-8-sig"),
            )
        if imagej_results is not None:
            archive.writestr(
                f"{stem}_ImageJ_results.csv",
                imagej_results.to_csv(index=False).encode("utf-8-sig"),
            )
        if direction_table is not None:
            archive.writestr(
                f"{stem}_fiber_directions.csv",
                direction_table.to_csv(index=False).encode("utf-8-sig"),
            )
        if annotated_png is not None:
            archive.writestr(f"{stem}_labeled_thickness.png", annotated_png)
        if annotated_unlabeled_png is not None:
            archive.writestr(f"{stem}_thickness.png", annotated_unlabeled_png)
        if unit_metadata is not None:
            archive.writestr(
                f"{stem}_measurement_units.json",
                json.dumps(unit_metadata, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            )
    return buffer.getvalue()


def _manual_region_id() -> str:
    return f"manual-{uuid.uuid4().hex[:12]}"


def apply_canvas_edits(
    measurements: pd.DataFrame,
    new_measurements: list[dict] | None,
    delete_ids: list[str] | None,
    analysis_scale: float = 1.0,
    nm_per_px: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply one browser-side edit batch after the user presses 'Apply'."""
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    updated = measurements.copy(deep=True)
    events: list[dict] = []
    ids = {str(value) for value in (delete_ids or [])}
    delete_all_auto = _DELETE_ALL_AUTO_TOKEN in ids
    ids.discard(_DELETE_ALL_AUTO_TOKEN)

    if delete_all_auto and not updated.empty:
        source = updated.get("source", pd.Series("", index=updated.index)).astype(str)
        status = updated.get("status", pd.Series("", index=updated.index)).astype(str)
        auto_mask = (source != "manual") & status.isin(_ACTIVE_STATUSES)
        updated.loc[auto_mask, "status"] = "rejected"
        updated.loc[auto_mask, "review_label"] = "auto_cleared"
        events.append({
            "timestamp": _utc_now(),
            "action": "erase_all_automatic_measurements",
            "count": int(auto_mask.sum()),
        })

    if ids and not updated.empty:
        mask = updated["measurement_id"].astype(str).isin(ids)
        updated.loc[mask, "status"] = "rejected"
        updated.loc[mask, "review_label"] = "erased"
        events.append({
            "timestamp": _utc_now(),
            "action": "erase_measurements",
            "measurement_ids": sorted(ids),
            "count": int(mask.sum()),
        })

    rows: list[dict] = []
    for item in new_measurements or []:
        p1 = item.get("p1")
        p2 = item.get("p2")
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2):
            continue
        x1, y1 = map(float, p1)
        x2, y2 = map(float, p2)
        metrics = distance_measurement((x1, y1), (x2, y2), analysis_scale, nm_per_px)
        if not np.isfinite(metrics["analysis_width_px"]) or metrics["analysis_width_px"] <= 0:
            continue
        measurement_id = f"manual-{uuid.uuid4().hex[:12]}"
        region_id = str(item.get("fiber_region_id") or _manual_region_id())
        chord_direction = np.rad2deg(np.arctan2(-(y2 - y1), x2 - x1))
        tangent_direction = float((chord_direction + 180.0) % 180.0 - 90.0)
        try:
            supplied_direction = float(item.get("direction_deg"))
            if np.isfinite(supplied_direction):
                tangent_direction = supplied_direction
        except (TypeError, ValueError):
            pass
        row = {
            "measurement_id": measurement_id,
            "fiber_region_id": region_id,
            "fiber_path_id": item.get("fiber_path_id"),
            "region_sample_index": 0,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": 0.5 * (x1 + x2),
            "center_y": 0.5 * (y1 + y2),
            "xm": 0.5 * (x1 + x2),
            "ym": 0.5 * (y1 + y2),
            "width_px": metrics["analysis_width_px"],
            "width_original_px": metrics["original_width_px"],
            "width_nm": np.nan if metrics["width_nm"] is None else metrics["width_nm"],
            "direction_deg": tangent_direction,
            "local_orientation_deg": np.nan,
            "local_coherency": np.nan,
            "orientation_error_deg": np.nan,
            "orientation_score": 1.0,
            "grade": "USER",
            "confidence": 1.0,
            "sem_agreement": 1.0,
            "bundle_score": 0.0,
            "status": "active",
            "source": "manual",
            "review_label": "corrected" if item.get("replacement_for") not in (None, "") else "manual",
            "replacement_for": item.get("replacement_for"),
        }
        rows.append(row)
        events.append({
            "timestamp": _utc_now(),
            "action": "manual_add",
            "measurement_id": measurement_id,
            "fiber_region_id": region_id,
            "fiber_path_id": item.get("fiber_path_id"),
            "replacement_for": item.get("replacement_for"),
            "p1": [x1, y1],
            "p2": [x2, y2],
            "width_original_px": metrics["original_width_px"],
            "width_nm": metrics["width_nm"],
        })
    if rows:
        updated = pd.concat([updated, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return updated, events


def build_representative_lines(
    measurements: pd.DataFrame,
    representatives: pd.DataFrame,
) -> list[dict]:
    """Select one editable chord for each fibre-region representative/subregion."""
    if measurements is None or measurements.empty or representatives is None or representatives.empty:
        return []
    active = measurements[measurements["status"].astype(str).isin(_ACTIVE_STATUSES)].copy()
    active["width_px"] = pd.to_numeric(active["width_px"], errors="coerce")
    lines: list[dict] = []
    for _, rep in representatives.iterrows():
        region = rep.get("fiber_region_id")
        def _region_text(value):
            try:
                number = float(value)
                if np.isfinite(number) and number.is_integer():
                    return str(int(number))
            except (TypeError, ValueError):
                pass
            return str(value)
        region_text = _region_text(region)
        group = active[active["fiber_region_id"].map(_region_text) == region_text].copy()
        if group.empty:
            continue
        if "region_sample_index" in group.columns:
            order = pd.to_numeric(group["region_sample_index"], errors="coerce")
            start = pd.to_numeric(pd.Series([rep.get("start_region_sample_index")]), errors="coerce").iloc[0]
            end = pd.to_numeric(pd.Series([rep.get("end_region_sample_index")]), errors="coerce").iloc[0]
            if np.isfinite(start) and np.isfinite(end):
                within = group[order.between(float(start), float(end), inclusive="both")]
                if not within.empty:
                    group = within
        target = float(rep.get("representative_width_px", np.nan))
        distance = np.abs(pd.to_numeric(group["width_px"], errors="coerce") - target)
        confidence = pd.to_numeric(group.get("confidence", 0.5), errors="coerce").fillna(0.5)
        choice = group.assign(_rank=distance - 0.05 * confidence).sort_values("_rank").iloc[0]
        source = str(choice.get("source", "auto"))
        erase_ids = group["measurement_id"].astype(str).tolist()
        ordered_path = _ordered_group(group)
        path_points = [
            [float(x), float(y)]
            for x, y in zip(
                pd.to_numeric(ordered_path.get("center_x"), errors="coerce"),
                pd.to_numeric(ordered_path.get("center_y"), errors="coerce"),
            )
            if np.isfinite(x) and np.isfinite(y)
        ]
        lines.append({
            "id": f"rep-{region}-{rep.get('subregion_id', 1)}",
            "measurement_id": str(choice.get("measurement_id")),
            "fiber_region_id": region_text,
            "subregion_id": int(rep.get("subregion_id", 1)),
            "x1": float(choice.get("x1")),
            "y1": float(choice.get("y1")),
            "x2": float(choice.get("x2")),
            "y2": float(choice.get("y2")),
            "width_analysis_px": float(choice.get("width_px")),
            "width_original_px": float(rep.get("representative_width_original_px", rep.get("representative_width_px"))),
            "width_nm": None if not np.isfinite(float(rep.get("representative_width_nm", np.nan))) else float(rep.get("representative_width_nm")),
            "direction_deg": None if not np.isfinite(float(choice.get("direction_deg", np.nan))) else float(choice.get("direction_deg")),
            "source": source,
            "status": str(choice.get("status", "active")),
            "review_label": str(choice.get("review_label", "")),
            "replacement_for": choice.get("replacement_for"),
            "erase_ids": erase_ids,
            "path_points": path_points,
        })
    # Visible labels are regenerated after every edit. Sorting from top to bottom and
    # left to right guarantees there are no gaps and makes exports reproducible.
    lines.sort(key=lambda line: (
        0.5 * (float(line["y1"]) + float(line["y2"])),
        0.5 * (float(line["x1"]) + float(line["x2"])),
    ))
    for label, line in enumerate(lines, start=1):
        line["label"] = label
    return lines

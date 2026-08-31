"""Metrics, per-specimen aggregation and PASS/REVIEW/FAIL status."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (aggregate_by_specimen, distribution_metrics, fiber_level_recall, match_sites,
                         matched_site_metrics)
from src.quality import FAIL, PASS, REVIEW, QualityThresholds, field_status, publication_set


def _sites(widths, xs, ys, angles):
    return pd.DataFrame({"center_x_px": xs, "center_y_px": ys, "width_px": widths,
                         "fiber_angle_raster_deg": angles})


def test_distribution_metrics_identity_and_scale():
    rng = np.random.default_rng(0)
    gt = rng.lognormal(np.log(9.0), 0.4, 500)
    same = distribution_metrics(gt, gt)
    assert same["wasserstein"] == pytest.approx(0.0)
    assert same["sd_ratio"] == pytest.approx(1.0)
    assert same["median_relative_error"] == pytest.approx(0.0)
    for k in ("gt_p90", "gt_p95", "gt_iqr", "gt_sd", "ks_statistic"):
        assert k in same
    half = distribution_metrics(gt, 0.5 * gt)
    assert half["median_relative_error"] == pytest.approx(-0.5, abs=1e-6)
    assert half["sd_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert half["p90_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert half["wasserstein"] > 0
    # too few samples -> only counts, no fabricated statistics
    tiny = distribution_metrics([1, 2], gt)
    assert set(tiny) == {"n_gt", "n_pred"}


def test_matching_is_one_to_one_and_reports_bias_vs_width():
    n = 40
    rng = np.random.default_rng(1)
    xs = rng.uniform(20, 480, n)
    ys = rng.uniform(20, 480, n)
    w = rng.uniform(5, 30, n)
    ang = rng.uniform(-90, 90, n)
    gt = _sites(w, xs, ys, ang)
    # predictions: +10 % width bias, 1 px jitter, one duplicate and one far-away spurious site
    pred = _sites(np.r_[1.08 * w, 1.08 * w[0], 12.0],
                  np.r_[xs + 1.0, xs[0] + 1.5, 5000.0],
                  np.r_[ys - 0.5, ys[0], 5000.0],
                  np.r_[ang, ang[0], 0.0])
    m = match_sites(gt, pred)
    assert len(m) == n
    assert m["gt_index"].is_unique and m["pred_index"].is_unique
    assert (m["distance"] < 2.0).all()
    met = matched_site_metrics(m, n_gt=n, n_pred=len(pred))
    assert met["n_matched"] == n and met["matched_fraction_of_gt"] == 1.0
    assert met["median_relative_error"] == pytest.approx(0.08, abs=1e-6)
    assert met["bias_px"] > 0
    assert met["within_20pct"] == 1.0 and met["within_10pct"] == 1.0
    assert met["pearson_r"] > 0.99
    assert met["bias_vs_width"] and all(r["n"] >= 3 for r in met["bias_vs_width"])
    rec = fiber_level_recall(gt, pred)
    assert rec["fiber_recall"] == 1.0 and rec["n_pred"] == len(pred)


def test_angle_mismatch_blocks_a_match():
    gt = _sites([10.0], [100.0], [100.0], [0.0])
    pred = _sites([10.0], [100.0], [100.5], [80.0])
    assert len(match_sites(gt, pred, max_angle_deg=30.0)) == 0
    pred_ok = _sites([10.0], [100.0], [100.5], [88.0])   # wraps to 2 deg from -90/90 side? no: 88 vs 0 = 88
    assert len(match_sites(gt, pred_ok, max_angle_deg=30.0)) == 0
    pred_wrap = _sites([10.0], [100.0], [100.5], [-88.0])
    gt_wrap = _sites([10.0], [100.0], [100.0], [89.0])   # 89 vs -88 is 3 deg apart across the wrap
    assert len(match_sites(gt_wrap, pred_wrap, max_angle_deg=30.0)) == 1


def _per_field(n_fields, specimens, rng):
    per, spec_of = {}, {}
    for i in range(n_fields):
        f = f"F{i}"
        per[f] = {"distribution_sites": {"median_relative_error": float(rng.normal(0.05, 0.02)),
                                         "sd_ratio": float(rng.normal(0.9, 0.05))},
                  "roll_up": {"unassigned_fraction": float(rng.uniform(0.05, 0.2))}}
        spec_of[f] = specimens[i % len(specimens)]
    return per, spec_of


def test_bootstrap_ci_requires_enough_specimen_groups():
    rng = np.random.default_rng(2)
    per, spec_of = _per_field(6, ["A", "B", "C"], rng)
    few = aggregate_by_specimen(per, spec_of, n_boot=200, min_groups_for_ci=5)
    assert few["n_specimens"] == 3 and few["ci_warning"]
    for entry in few["metrics"].values():
        assert "ci95_over_specimens" not in entry
        assert len(entry["per_specimen"]) == 3
    per, spec_of = _per_field(12, ["A", "B", "C", "D", "E", "F"], rng)
    many = aggregate_by_specimen(per, spec_of, n_boot=200, min_groups_for_ci=5)
    assert many["ci_warning"] is None
    for name, entry in many["metrics"].items():
        lo, hi = entry["ci95_over_specimens"]
        assert lo <= entry["mean_over_specimens"] <= hi, name
    # per-specimen aggregation weights specimens, not fields
    per = {"F0": {"roll_up": {"unassigned_fraction": 0.0}}, "F1": {"roll_up": {"unassigned_fraction": 0.0}},
           "F2": {"roll_up": {"unassigned_fraction": 0.0}}, "F3": {"roll_up": {"unassigned_fraction": 1.0}}}
    agg = aggregate_by_specimen(per, {"F0": "A", "F1": "A", "F2": "A", "F3": "B"}, n_boot=0)
    assert agg["metrics"]["unassigned_fraction"]["mean_over_specimens"] == pytest.approx(0.5)


def _status(**kw):
    base = dict(seg_area=0.35, separability_auc=0.9, coherent_fraction=0.6, boundary_agreement=0.85,
                unassigned_fraction=0.1, n_fibres=60, accepted_fraction=0.8, ood_z=0.5, ood=[],
                calibration_valid=True)
    base.update(kw)
    return field_status(**base)


def test_field_status_pass_review_fail_and_nm_policy():
    ok = _status()
    assert ok["status"] == PASS and ok["nm_status"] == "valid" and not ok["fail_reasons"]
    rv = _status(boundary_agreement=0.6, n_fibres=20)
    assert rv["status"] == REVIEW and len(rv["review_reasons"]) == 2
    fl = _status(seg_area=0.95)
    assert fl["status"] == FAIL and fl["fail_reasons"][0].startswith("segmentation_area")
    assert _status(ood_z=6.0)["status"] == FAIL
    assert _status(ood_z=3.5, ood=[{"metric": "contrast", "z": 3.5}])["status"] == REVIEW
    uncal = _status(calibration_valid=False, calibration_reason="unresolved")
    assert uncal["status"] == PASS and uncal["nm_status"] == "calibration_invalid"
    thr = QualityThresholds(min_fibres_fail=100)
    assert _status(thr=thr)["status"] == FAIL
    pub = publication_set({"a": ok, "b": rv, "c": fl})
    assert pub == ["a"]
    assert publication_set({"a": ok, "b": rv, "c": fl}, include_review=True) == ["a", "b"]

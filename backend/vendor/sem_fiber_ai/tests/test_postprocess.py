"""Post-processing and roll-up: an oracle model must give back the known widths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.tests.helpers import oracle_maps
from sem_fiber_ai.src.metrics import (distribution_metrics, fiber_level_recall, match_sites,
                                      matched_site_metrics)
from sem_fiber_ai.src.postprocess import (REJECT_CODES, SITE_COLUMNS, PostConfig,
                                          decode_predictions, rejection_summary,
                                          suppress_duplicates)
from sem_fiber_ai.src.rollup import RollupConfig, distribution_summary, field_summary, rollup
from sem_fiber_ai.src.synthetic import make_field


@pytest.fixture(scope="module")
def oracle(synth_field):
    maps = oracle_maps(synth_field, exact_distance=True)
    sites = decode_predictions(maps, image_id="F1", nm_per_pixel=synth_field.nm_per_px,
                               calibration_valid=True, cfg=PostConfig(spacing_px=12.0))
    fib, sites2, info = rollup(sites, synth_field.mask)
    return synth_field, maps, sites, fib, sites2, info


def test_site_table_schema_and_rejection_codes(oracle):
    _f, _m, sites, *_ = oracle
    assert list(sites.columns) == SITE_COLUMNS
    codes = set(sites["rejected_reason"].unique()) - {""}
    assert codes <= set(REJECT_CODES) | {"duplicate_same_fiber"}
    assert rejection_summary(sites)["accepted"] == int((sites.rejected_reason == "").sum())


def test_oracle_widths_are_unbiased(oracle):
    f, _m, sites, fib, _s2, info = oracle
    acc = sites[sites.rejected_reason == ""]
    mt = match_sites(f.annotations, acc)
    mm = matched_site_metrics(mt, len(f.annotations), len(acc))
    assert mm["n_matched"] >= 30
    assert abs(mm["median_relative_error"]) < 0.03, mm["median_relative_error"]
    assert mm["angle_median_abs_error_deg"] < 1.0
    dm = distribution_metrics(f.annotations.width_px, fib.width_px)
    assert 0.9 < dm["pred_median"] / dm["gt_median"] < 1.1
    assert info["unassigned_fraction"] < 0.1
    assert fiber_level_recall(f.annotations, acc)["fiber_recall"] > 0.6


def test_nm_policy(oracle):
    f, maps, *_ = oracle
    ok = decode_predictions(maps, image_id="F1", nm_per_pixel=2.0, calibration_valid=True)
    bad = decode_predictions(maps, image_id="F1", nm_per_pixel=2.0, calibration_valid=False)
    assert np.allclose(ok.width_nm, ok.width_px * 2.0)
    assert bad.width_nm.isna().all() and not bad.calibration_valid.any()


def test_spacing_controls_site_density(oracle):
    f, maps, *_ = oracle
    a = decode_predictions(maps, cfg=PostConfig(spacing_px=8.0))
    b = decode_predictions(maps, cfg=PostConfig(spacing_px=24.0))
    assert len(a) > 1.8 * len(b)


def test_baseline_duplicate_suppression():
    df = pd.DataFrame({"center_x_px": [10.0, 12.0, 60.0], "center_y_px": [10.0, 11.0, 60.0],
                       "fiber_angle_raster_deg": [30.0, 32.0, 30.0], "width_px": [8.0, 8.0, 8.0],
                       "confidence": [0.9, 0.8, 0.7]})
    kept = suppress_duplicates(df, PostConfig(mode="baseline"))
    assert len(kept) == 2 and 0 in kept.index and 2 in kept.index


def test_rollup_merges_across_junctions_and_reports_unassigned():
    f = make_field(5, H=256, W=256, n_fibres=6, n_annotations=40, image_id="R")
    maps = oracle_maps(f)
    sites = decode_predictions(maps, cfg=PostConfig(spacing_px=10.0))
    fib_merge, _, info_m = rollup(sites, f.mask, RollupConfig(merge_junctions=True))
    fib_cut, _, info_c = rollup(sites, f.mask, RollupConfig(merge_junctions=False))
    assert len(fib_merge) <= len(fib_cut)
    assert info_m["n_unassigned"] + info_m["n_assigned"] == info_m["n_sites"]
    summ = field_summary(fib_merge, sites, info_m, nm_valid=False)
    assert "number_weighted_px" in summ and "length_weighted_px" in summ
    assert "number_weighted_nm" not in summ


def test_length_weighted_distribution():
    d = distribution_summary(np.array([1.0, 10.0]), weights=np.array([1.0, 99.0]))
    assert d["median"] > 5.0 and d["weighting"] == "length"

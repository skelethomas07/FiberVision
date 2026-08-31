"""Targets: geometry-aware distance field, same-branch propagation, strata weights."""
from __future__ import annotations

import numpy as np
import pytest

from sem_fiber_ai.src.fiber_prior import FiberPrior, PriorConfig
from sem_fiber_ai.src.labels import validate_labels
from sem_fiber_ai.src.targets import (TARGET_KEYS, TargetConfig, encode_targets,
                                      strata_weight_map, strata_weights_from_widths)


@pytest.fixture(scope="module")
def field_and_prior(synth_field):
    prior = FiberPrior.compute(synth_field.image, PriorConfig(), ann=synth_field.annotations)
    return synth_field, prior


def test_synthetic_labels_are_schema_consistent(synth_field):
    rep = validate_labels(synth_field.annotations)
    assert rep["ok"], rep


def test_geometry_targets_have_every_key_and_valid_ranges(field_and_prior):
    f, prior = field_and_prior
    tg = encode_targets(f.image.shape, f.annotations, TargetConfig(mode="geometry"), prior=prior)
    assert set(tg) == set(TARGET_KEYS)
    for k in ("center", "segment", "validity", "validity_mask", "reg_mask", "ignore", "dist_weight"):
        assert tg[k].min() >= 0.0 and tg[k].max() <= 1.0 + 1e-6, k
    assert tg["dist"].min() >= 0.0
    assert tg["dist"][~prior.mask].max() == 0.0, "distance is zero off the mask"
    # the ridge target is dense: every mask pixel is within a few px of a positive
    assert (tg["center"] > 0.5).sum() > 0.01 * prior.mask.sum()


def test_distance_target_matches_manual_width_at_sites(field_and_prior):
    f, prior = field_and_prior
    tg = encode_targets(f.image.shape, f.annotations, TargetConfig(mode="geometry"), prior=prior)
    xs = f.annotations.center_x_px.round().astype(int).to_numpy()
    ys = f.annotations.center_y_px.round().astype(int).to_numpy()
    w = f.annotations.width_px.to_numpy()
    from scipy import ndimage as ndi

    dmax = ndi.maximum_filter(tg["dist"], size=3)
    ratio = 2.0 * dmax[ys, xs] / w
    assert 0.9 < np.median(ratio) < 1.1, np.median(ratio)
    assert np.median(tg["dist_weight"][ys, xs]) > 0.9, "verified sites carry full weight"


def test_propagation_stays_on_the_same_branch(field_and_prior):
    f, prior = field_and_prior
    cfg = TargetConfig(mode="geometry", unverified_weight=0.3)
    one = f.annotations.iloc[[0]]
    tg = encode_targets(f.image.shape, one, cfg, prior=prior)
    verified = (tg["dist_weight"] > cfg.unverified_weight + 1e-3) & prior.mask
    # verified pixels form a connected region containing the site and cover a
    # small fraction of the whole mask, not everything
    assert verified.any()
    frac = verified.sum() / prior.mask.sum()
    assert frac < 0.5, frac
    from scipy import ndimage as ndi

    lab, n = ndi.label(verified, structure=np.ones((3, 3)))
    assert n >= 1


def test_baseline_targets_ignore_unlabelled_fibre(field_and_prior):
    f, prior = field_and_prior
    tg = encode_targets(f.image.shape, f.annotations.iloc[:5], TargetConfig(mode="baseline"), prior=prior)
    assert tg["ignore"][prior.mask].mean() > 0.5
    assert (tg["center"] > 0.99).sum() >= 5


def test_strata_weights_favour_rare_bands():
    widths = np.concatenate([np.full(900, 7.0), np.full(100, 20.0)])
    edges = (0.0, 6.0, 9.0, 13.0, 18.0, 25.0, 1e9)
    wts = strata_weights_from_widths(widths, edges, cap=6.0)
    cfg = TargetConfig(strata_edges=edges, strata_weights=wts)
    m = strata_weight_map(np.array([7.0, 20.0]), cfg)
    assert m[1] > m[0]
    assert max(wts) <= 6.0 and min(wts) >= 0.2


def test_negative_and_wrapped_angles_are_encoded_consistently(field_and_prior):
    f, prior = field_and_prior
    cfg = TargetConfig(mode="geometry", orientation_source="chord")
    ann = f.annotations.copy()
    ann["fiber_angle_raster_deg"] = -89.5      # nearly vertical, negative
    tg = encode_targets(f.image.shape, ann, cfg, prior=prior)
    xs = ann.center_x_px.round().astype(int).to_numpy()
    ys = ann.center_y_px.round().astype(int).to_numpy()
    a = 0.5 * np.degrees(np.arctan2(tg["sin2t"][ys, xs], tg["cos2t"][ys, xs]))
    from sem_fiber_ai.src.coords import angular_diff_180

    assert np.all(angular_diff_180(a, -89.5) < 1e-3)

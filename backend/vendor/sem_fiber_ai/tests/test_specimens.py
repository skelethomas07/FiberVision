"""Specimen grouping, sealed splits, LOSO folds and leakage detection."""
from __future__ import annotations

import numpy as np
import pytest

from sem_fiber_ai.src import specimens as S


@pytest.mark.parametrize("image_id, specimen, series", [
    ("40s_58-4", "40s_58", "40s"),
    ("40s_58-1", "40s_58", "40s"),
    ("A_8", "A", "A"),
    ("48_3", "48", "48"),
    ("B_2", "B", "B"),
    ("3-8", "3", "3"),
    ("SEM_MG_23_6", "SEM_MG_23", "SEM"),
    ("plain", "plain", "plain"),
])
def test_specimen_key_strips_exactly_one_trailing_field_token(image_id, specimen, series):
    assert S.specimen_key(image_id) == specimen
    assert S.specimen_key(image_id, level="series") == series
    assert S.specimen_key(image_id, level="image") == image_id


def test_v6_bug_is_fixed_series_fields_share_a_group():
    # v6 used rsplit('-') so A_8 / 48_3 / B_2 each became a "specimen"
    ids = ["A_8", "A_9", "48_3", "48_4", "B_2", "B_3"]
    g = S.specimen_groups(ids)
    assert g["A_8"] == g["A_9"] and g["48_3"] == g["48_4"] and g["B_2"] == g["B_3"]
    assert len(set(g.values())) == 3


def test_override_map_wins_over_pattern():
    g = S.specimen_groups(["A_8", "A_9"], override={"A_9": "other"})
    assert g["A_8"] == "A" and g["A_9"] == "other"


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        S.specimen_key("A_8", level="whatever")


def _ids(n_spec=6, per=3):
    return [f"S{s}-{k}" for s in range(n_spec) for k in range(1, per + 1)]


def test_sealed_split_is_grouped_deterministic_and_leak_free():
    ids = _ids()
    groups = S.specimen_groups(ids)
    a = S.sealed_split(ids, groups, val_frac=0.2, test_frac=0.2, seed=1)
    b = S.sealed_split(ids, groups, val_frac=0.2, test_frac=0.2, seed=1)
    assert a["train"] == b["train"] and a["val"] == b["val"] and a["test"] == b["test"]
    S.assert_no_leakage(a)
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(ids)
    assert not (set(a["train"]) & set(a["test"])) and not (set(a["val"]) & set(a["test"]))
    # every specimen sits in exactly one part
    for part in ("train", "val", "test"):
        gs = {groups[i] for i in a[part]}
        for other in ("train", "val", "test"):
            if other != part:
                assert not (gs & {groups[i] for i in a[other]})
    assert a["test_groups"] and a["val_groups"]
    assert S.split_digest(a) == S.split_digest(b)
    assert S.split_digest(a) != S.split_digest(S.sealed_split(ids, groups, seed=2))


def test_sealed_split_refuses_when_nothing_can_be_held_out():
    ids = ["only-1", "only-2", "only-3"]
    groups = S.specimen_groups(ids)          # a single group
    with pytest.raises((RuntimeError, ValueError)):
        S.sealed_split(ids, groups)


def test_stratified_split_places_every_stratum_in_test_when_possible():
    ids = _ids(8, 2)
    groups = S.specimen_groups(ids)
    strata = {i: ("coarse" if int(i[1]) % 2 else "fine") for i in ids}
    sp = S.sealed_split(ids, groups, strata=strata, val_frac=0.25, test_frac=0.25, seed=3)
    S.assert_no_leakage(sp)
    assert {strata[i] for i in sp["test"]} == {"coarse", "fine"}


def test_loso_folds_cover_every_non_test_group_once():
    ids = _ids(5, 2)
    groups = S.specimen_groups(ids)
    folds = S.loso_folds(ids, groups, exclude_groups=["S4"])
    held = [f["held_out_group"] for f in folds]
    assert sorted(held) == ["S0", "S1", "S2", "S3"]
    for f in folds:
        S.assert_no_leakage(f, groups)
        assert all(groups[i] == f["held_out_group"] for i in f["val"])
        assert not any(groups[i] == "S4" for i in f["train"] + f["val"])


def test_assert_no_leakage_detects_image_and_group_leaks():
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        S.assert_no_leakage({"train": ["a-1"], "val": [], "test": ["a-1"]}, {"a-1": "a"})
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        S.assert_no_leakage({"train": ["a-1"], "val": ["a-2"], "test": []}, {"a-1": "a", "a-2": "a"})


def test_near_duplicates_are_clustered_and_merged_into_grouping():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(96, 96)).astype(np.uint8)
    from scipy import ndimage as ndi

    base = ndi.gaussian_filter(base.astype(float), 3.0)
    base = (255 * (base - base.min()) / (np.ptp(base) + 1e-9)).astype(np.uint8)
    near = np.clip(base.astype(int) + rng.integers(-3, 4, size=base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, size=(96, 96)).astype(np.uint8)
    dup, pairs = S.near_duplicate_groups({"X-1": base, "Y-1": near, "Z-1": other})
    assert dup["X-1"] == dup["Y-1"] and dup["Z-1"] != dup["X-1"]
    assert pairs and {pairs[0]["a"], pairs[0]["b"]} == {"X-1", "Y-1"}
    merged = S.merge_groups(S.specimen_groups(["X-1", "Y-1", "Z-1"]), dup)
    assert merged["X-1"] == merged["Y-1"] != merged["Z-1"]


def test_calibration_strata_buckets_nm_per_px():
    st = S.calibration_strata({"a": 1.0, "b": 1.1, "c": 4.0, "d": None})
    assert st["a"] == st["b"] and st["a"] != st["c"]
    assert st["d"] is not None

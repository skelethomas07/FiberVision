"""Regression tests for v6 publication-grade leakage barriers."""
import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.pseudolabel import calibrate_widths
from sem_fiber_ai.src.two_stage import _assert_no_specimen_leakage


def _widths(iid, med, n=30):
    return pd.DataFrame({"image_id":[iid]*n,
                         "width_px":np.linspace(med-0.5,med+0.5,n)})


def test_calibration_strict_rejects_nonconstant_factor():
    pseudo=pd.concat([_widths("A-1",10),_widths("B-1",10)],ignore_index=True)
    manual=pd.concat([_widths("A-1",10),_widths("B-1",16)],ignore_index=True)
    with pytest.raises(RuntimeError, match="not transferable"):
        calibrate_widths(pseudo,manual,strict=True,max_relative_range=0.25,min_images=2)


def test_calibration_uses_only_supplied_manual_allowlist():
    pseudo=pd.concat([_widths("A-1",10),_widths("B-1",10),_widths("TEST-1",10)],ignore_index=True)
    manual=pd.concat([_widths("A-1",10),_widths("B-1",10)],ignore_index=True)
    _,info=calibrate_widths(pseudo,manual,strict=True,min_images=2)
    assert set(info["per_image_ratio"])=={"A-1","B-1"}
    assert "TEST-1" not in info["per_image_ratio"]


def test_specimen_audit_catches_pseudo_test_leak():
    with pytest.raises(RuntimeError,match="specimen leakage"):
        _assert_no_specimen_leakage(train_ids=["A-1"],val_ids=["B-1"],
            test_ids=["T-1"],pseudo_ids=["A-9","T-8"],calibration_ids=["A-1"])


def test_specimen_audit_clean_nested_split():
    out=_assert_no_specimen_leakage(train_ids=["A-1","A-2"],val_ids=["B-1"],
        test_ids=["T-1"],pseudo_ids=["A-9"],calibration_ids=["A-1","A-2"])
    assert out["passed"] is True
    assert out["test_specimens"]==["T"]


def test_prior_cache_changes_when_pixels_change(tmp_path):
    from sem_fiber_ai.src.fiber_prior import FiberPrior, PriorConfig
    a=np.zeros((48,48),np.uint8); a[:,20:25]=220
    b=np.zeros((48,48),np.uint8); b[20:25,:]=220
    base=tmp_path/"same_id_prior"
    pa=FiberPrior.load_or_compute(a,base,PriorConfig(polarity="bright"),ann=None)
    pb=FiberPrior.load_or_compute(b,base,PriorConfig(polarity="bright"),ann=None)
    assert not np.array_equal(pa.mask,pb.mask)
    assert len(list(tmp_path.glob("same_id_prior.*.npz")))==2

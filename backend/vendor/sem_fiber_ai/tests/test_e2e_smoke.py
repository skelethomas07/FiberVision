"""End-to-end CPU smoke test: synthetic data -> split -> train -> resume -> select -> evaluate -> infer.

This proves the code path runs and that the guarantees (sealed split, NaN nm
without calibration, refusal of split members, exact resume) hold on a tiny
synthetic set.  It is not evidence about real SEM images.
"""
import json
from pathlib import Path

import pandas as pd

from src.selftest import run_selftest


def test_cpu_selftest_end_to_end(tmp_path):
    rep = run_selftest(tmp_path, keep=True, n_specimens=4, fields_per_specimen=2, H=192, W=192, verbose=False)
    assert rep["ok"]
    for step in ("synthetic dataset", "sealed split", "smoke training (2 epochs)", "resume check",
                 "select + evaluate", "manifest", "infer new images", "split member refusal"):
        assert rep["steps"][step]["ok"], step
    split = rep["split"]
    assert set(split["train"]).isdisjoint(split["test"]) and set(split["val"]).isdisjoint(split["test"])
    assert rep["training"]["epochs_run"] == 2
    run = Path(tmp_path) / "run"
    man = json.loads((run / "manifest.json").read_text())
    assert man["format"] == "sem_fiber_ai_manifest_v7" and man["package_version"] == "7.0.0"
    assert man["run_mode"] == "FAST_SMOKE_TEST" and man["split"]["digest"]
    assert "best.pt" in man["checkpoints"] and "last.pt" in man["checkpoints"]
    assert (run / "selection.json").exists() and (run / "eval_test" / "metrics_test.json").exists()
    sel = json.loads((run / "selection.json").read_text())
    assert sel["selected_on_split"] == "val", "selection must never look at the test split"
    assert set(sel["selected_on"]) <= set(split["val"]) and not (set(sel["selected_on"]) & set(split["test"]))
    mt = json.loads((run / "eval_test" / "metrics_test.json").read_text())
    assert "per_field" in mt and "aggregate_all_fields" in mt and "quality" in mt
    # inference wrote a machine-readable site table with rejection codes, and NaN nm when uncalibrated
    s1 = pd.read_csv(tmp_path / "infer_out" / "NEW-1_sites.csv")
    assert "rejected_reason" in s1.columns and s1["width_nm"].isna().all()
    q = json.loads((tmp_path / "infer_out" / "NEW-1_summary.json").read_text())["quality"]
    assert q["nm_status"] == "calibration_invalid" and q["status"] in ("PASS", "REVIEW", "FAIL")

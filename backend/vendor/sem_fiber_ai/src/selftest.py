"""CPU end-to-end smoke test on synthetic data (v7).

Runs the whole pipeline in FAST_SMOKE_TEST mode on a synthetic dataset with
known widths and angles.  It proves the code path executes and that the
plumbing (splits, calibration policy, tiling, resume, selection, evaluation,
inference, manifest) behaves; it is NOT evidence about real SEM images.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def run_selftest(work_dir: str | Path | None = None, *, keep: bool = False,
                 n_specimens: int = 5, fields_per_specimen: int = 2, H: int = 256, W: int = 256,
                 verbose: bool = True) -> dict[str, Any]:
    from .infer import load_run, measure_folder
    from .pipeline import (load_dataset, make_split, plan_physical, run_training, select_and_evaluate,
                           write_run_manifest)
    from .synthetic import write_synthetic_dataset
    from .train import load_config
    from .utils import package_version

    t0 = time.time()
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="sem_fiber_selftest_"))
    root.mkdir(parents=True, exist_ok=True)
    pkg = Path(__file__).resolve().parents[1]
    cfg = load_config(pkg / "config" / "default.yaml")
    report: dict[str, Any] = {"package_version": package_version(), "work_dir": str(root), "steps": {}}

    def step(name, fn):
        t = time.time()
        out = fn()
        report["steps"][name] = {"ok": True, "seconds": round(time.time() - t, 2)}
        if verbose:
            print(f"  [selftest] {name:<28s} ok  ({time.time() - t:5.1f}s)")
        return out

    ds = step("synthetic dataset", lambda: write_synthetic_dataset(
        root / "data", n_specimens=n_specimens, fields_per_specimen=fields_per_specimen, H=H, W=W,
        n_annotations=40))
    run_dir = root / "run"
    run_dir.mkdir(exist_ok=True)
    records = step("load records", lambda: load_dataset(ds["labels_csv"], ds["image_dir"], cfg,
                                                        prior_cache_dir=run_dir / "prior_cache"))
    split = step("sealed split", lambda: make_split(records, cfg, run_dir))
    assert not (set(split["train"]) & set(split["test"])), "split leakage"
    phys = step("physical plan", lambda: plan_physical(records, split, cfg, run_dir))
    if phys["factors"]:
        records = step("reload resampled", lambda: load_dataset(
            ds["labels_csv"], ds["image_dir"], cfg, prior_cache_dir=run_dir / "prior_cache",
            resample_factors=phys["factors"]))
        for r in records:
            r.specimen = split["groups"].get(r.image_id, r.image_id)
    res = step("smoke training (2 epochs)", lambda: run_training(
        cfg, records, split, run_dir, run_mode="FAST_SMOKE_TEST", resume=False, user="selftest"))
    # exact resume: continuing a finished run must be a no-op with identical state
    res2 = step("resume check", lambda: run_training(
        cfg, records, split, run_dir, run_mode="FAST_SMOKE_TEST", resume=True, user="selftest"))
    assert res2["manifest"]["resumed_from"] is not None, "resume did not pick up last.pt"
    assert len(res2["history"]["train_loss"]) == len(res["history"]["train_loss"]), "resume changed history"
    ev = step("select + evaluate", lambda: select_and_evaluate(cfg, records, split, run_dir, label="selftest"))
    assert (run_dir / "selection.json").exists()
    assert "test" in ev and ev["test"]["n_fields"] > 0
    man = step("manifest", lambda: write_run_manifest(
        cfg, run_id="selftest", user="selftest", run_mode="FAST_SMOKE_TEST", split=split, records=records,
        package_dir=pkg, run_dir=run_dir, labels_csv=ds["labels_csv"], calibration=None,
        selection=ev["selection"], train_result=res, evaluation=ev,
        notes=["synthetic smoke test; not a scientific result"]))
    # inference on new images (synthetic, not in the split) + refusal of split members
    new_dir = root / "new_images"
    new_dir.mkdir(exist_ok=True)
    from .synthetic import make_field
    import cv2

    for k in range(2):
        f = make_field(900 + k, H=H, W=W, n_fibres=18, image_id=f"NEW-{k}")
        cv2.imwrite(str(new_dir / f"NEW-{k}.png"), f.image.astype(np.uint8))
    run = step("load run", lambda: load_run(run_dir))
    batch = step("infer new images", lambda: measure_folder(run, new_dir, root / "infer_out",
                                                            calib_table={"NEW-0": 2.0}))
    assert batch["n_images"] == 2
    r0 = batch["results"][0]
    assert r0["calibration"]["valid"] is True and r0["quality"]["nm_status"] == "valid"
    r1 = batch["results"][1]
    assert r1["calibration"]["valid"] is False and r1["quality"]["nm_status"] == "calibration_invalid"
    import pandas as pd

    s1 = pd.read_csv(root / "infer_out" / "NEW-1_sites.csv")
    assert s1["width_nm"].isna().all(), "nm must be NaN without valid calibration"
    refused = False
    try:
        from .infer import measure_image

        measure_image(run, Path(ds["image_dir"]) / f"{split['train'][0]}.png", root / "infer_out")
    except RuntimeError:
        refused = True
    assert refused, "split members must be refused"
    report["steps"]["split member refusal"] = {"ok": True}
    report.update({"split": {k: split[k] for k in ("train", "val", "test")},
                   "physical": {"enabled": phys["enabled"], "reference_nm_per_px": phys["reference_nm_per_px"]},
                   "training": {"epochs_run": res["manifest"]["epochs_run"], "best_epoch": res["best_epoch"],
                                "stop_reason": res["stop_reason"]},
                   "selection": {k: ev["selection"][k] for k in ("spacing_px", "min_validity", "seg_threshold")},
                   "test_fields": ev["test"]["n_fields"], "manifest_sha_pkg": man["package_tree_sha256"][:16],
                   "seconds": round(time.time() - t0, 1), "ok": True})
    (root / "selftest_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"  [selftest] ALL STEPS PASSED in {report['seconds']}s  (work dir: {root})")
    if not keep and work_dir is None:
        shutil.rmtree(root, ignore_errors=True)
    return report


if __name__ == "__main__":                                  # pragma: no cover
    run_selftest(keep=True)

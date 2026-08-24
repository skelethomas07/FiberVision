"""The parser must survive real ImageJ exports, which vary a lot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sem_fiber_ai.src.csv_parser import infer_length_quantum, parse_measurement_csv


def _write(tmp_path, text, name="r.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_reads_the_uploaded_schema(tmp_path):
    p = _write(tmp_path, "\ufefflabel,Area,Mean,Min,Max,Angle,Length\n"
                         "1,32.0,151.5,121.0,169.1,-116.66,13.8666\n"
                         "2,44.0,144.4,114.5,170.7,-111.10,18.1333\n")
    out = parse_measurement_csv(p)
    assert out.n_rows == 2
    assert out.column_map["Length"] == "length"
    assert out.column_map["Angle"] == "angle"
    assert not out.has_coordinates          # this export carries no positions


def test_recognises_coordinates_when_present(tmp_path):
    p = _write(tmp_path, " ,BX,BY,Width,Height,Angle,Length\n"
                         "1,10,20,6,8,45,10\n2,30,40,6,8,-45,12\n")
    out = parse_measurement_csv(p)
    assert out.has_coordinates
    assert out.frame["cx"].iloc[0] == pytest.approx(13.0)
    assert out.frame["cy"].iloc[0] == pytest.approx(24.0)


def test_endpoint_schema_gives_centre(tmp_path):
    p = _write(tmp_path, "Label,X1,Y1,X2,Y2,Length,Angle\n"
                         "1,0,0,10,0,10,0\n")
    out = parse_measurement_csv(p)
    assert out.has_endpoints
    assert out.frame["cx"].iloc[0] == pytest.approx(5.0)


def test_bad_rows_are_reported_not_silently_dropped(tmp_path):
    p = _write(tmp_path, "label,Length,Angle\n1,10,0\n2,-3,0\n3,,0\n1,7,0\n")
    out = parse_measurement_csv(p)
    reasons = {e["reason"] for e in out.errors}
    assert "invalid_length" in reasons
    assert "duplicate_label" in reasons
    assert out.n_dropped == 3


def test_missing_length_column_is_an_error_not_a_crash(tmp_path):
    p = _write(tmp_path, "label,Foo,Bar\n1,2,3\n")
    out = parse_measurement_csv(p)
    assert any(e["reason"] == "missing_length_column" for e in out.errors)
    with pytest.raises(ValueError):
        parse_measurement_csv(p, strict=True)


def test_quantum_inference_finds_a_pixel_lattice():
    # whole-pixel measurements scaled by 16/15 nm per pixel
    lengths = np.arange(5, 60) * (16 / 15)
    res = infer_length_quantum(lengths)
    assert res["best"] is not None
    assert res["best"]["quantum"] == pytest.approx(16 / 15, rel=1e-6)
    assert res["best"]["frac_explained"] > 0.95


def test_quantum_inference_declines_on_continuous_data():
    rng = np.random.default_rng(0)
    res = infer_length_quantum(rng.uniform(5, 60, 400))
    assert res["best"] is None

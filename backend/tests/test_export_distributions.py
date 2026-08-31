from types import SimpleNamespace
import csv, io

from app.services.exports import build_orientation_distribution_csv, build_thickness_range_csv


def row(angle_deg=0.0, width_nm=50.0, active=True):
    return SimpleNamespace(
        id='x', source_model_measurement_id='m', x1=0, y1=0, x2=1, y2=1,
        width_px=10.0, width_nm=width_nm, angle_deg=angle_deg, active=active,
        edited=False, source='ai'
    )


def test_orientation_distribution_has_exactly_181_degree_columns_and_pairs():
    data = build_orientation_distribution_csv([
        row(angle_deg=-90.0),   # fiber 0
        row(angle_deg=-89.0),   # fiber 1
        row(angle_deg=89.6),    # fiber 179.6 -> 180
        row(angle_deg=0.0, active=False),
    ]).decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(data)))
    assert len(rows) == 2
    assert len(rows[0]) == 181
    assert rows[0][0] == '0°' and rows[0][180] == '180°'
    assert rows[1][0] == '1/3 (33.33%)'
    assert rows[1][1] == '1/3 (33.33%)'
    assert rows[1][180] == '1/3 (33.33%)'


def test_thickness_range_reports_fraction_of_all_active_fibers():
    data = build_thickness_range_csv([
        row(width_nm=40), row(width_nm=50), row(width_nm=60), row(width_nm=80), row(width_nm=55, active=False)
    ], 50, 60).decode('utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(data)))
    assert rows == [{
        'normal_min_nm': '50.000000', 'normal_max_nm': '60.000000',
        'total_fibers': '4', 'within_range_fibers': '2', 'within_range_percent': '50.00', 'within_range_pair': '2/4 (50.00%)'
    }]



def test_thickness_range_rejects_missing_nm_values():
    import pytest
    with pytest.raises(ValueError, match="nm thickness is unavailable"):
        build_thickness_range_csv([row(width_nm=None)], 40, 80)


def test_thickness_range_is_inclusive_at_both_edges():
    data = build_thickness_range_csv([row(width_nm=50), row(width_nm=60), row(width_nm=61)], 50, 60).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(data)))
    assert rows[0]["within_range_fibers"] == "2"
    assert rows[0]["total_fibers"] == "3"
    assert rows[0]["within_range_percent"] == "66.67"

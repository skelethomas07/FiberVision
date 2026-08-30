from io import BytesIO

from PIL import Image, ImageDraw

from app.services.scale_calibration import (
    detect_scale_bar,
    detect_scale_calibration,
    parse_scale_label,
    resolve_nm_per_pixel,
)


def _sem_with_footer(bar_width: int = 120) -> bytes:
    image = Image.new("L", (600, 400), color=145)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 330, 599, 399), fill=0)
    draw.rectangle((210, 342, 210 + bar_width - 1, 349), fill=255)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_parse_scale_label_supports_nm_um_and_micro_ocr_variant():
    assert parse_scale_label("500 nm") == (500.0, "500 nm")
    assert parse_scale_label("2.5 um") == (2500.0, "2.5 um")
    assert parse_scale_label("1µm") == (1000.0, "1µm")
    assert parse_scale_label("ee 1pm YONSEI") == (1000.0, "1pm")


def test_detect_scale_bar_finds_long_white_bar_in_dark_footer():
    bar = detect_scale_bar(_sem_with_footer(120))
    assert bar is not None
    assert abs(bar.width_px - 120) <= 1
    assert bar.y0 >= 330


def test_detect_scale_calibration_combines_bar_width_and_ocr_label():
    result = detect_scale_calibration(_sem_with_footer(125), ocr_runner=lambda _: "1pm YONSEI")
    assert result is not None
    assert result.source == "scale_bar"
    assert result.scale_value_nm == 1000.0
    assert abs(result.scale_bar_px - 125) <= 1
    assert abs(result.nm_per_pixel - 8.0) < 0.1


def test_manual_nm_per_pixel_overrides_automatic_detection():
    result = resolve_nm_per_pixel(_sem_with_footer(125), 3.25, ocr_runner=lambda _: "1pm")
    assert result.source == "manual"
    assert result.nm_per_pixel == 3.25
    assert result.scale_bar_px is None


def test_detection_failure_keeps_pixel_only_mode():
    blank = BytesIO()
    Image.new("L", (300, 200), color=128).save(blank, format="PNG")
    result = resolve_nm_per_pixel(blank.getvalue(), None, ocr_runner=lambda _: "")
    assert result.source == "none"
    assert result.nm_per_pixel is None

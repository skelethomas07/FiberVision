from io import BytesIO

from PIL import Image, ImageDraw

from app.services.visionflux_import import inspect_sem_upload


def _png(draw_fn=None):
    image = Image.new('RGB', (160, 120), (118, 118, 118))
    if draw_fn:
        draw = ImageDraw.Draw(image)
        draw_fn(draw)
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def test_plain_sem_creates_browser_preview_without_seed_measurements():
    result = inspect_sem_upload(_png(), 'sem.png')
    assert result.is_visionflux_annotated is False
    assert result.measurements == []
    assert result.preview_content_type == 'image/png'
    preview = Image.open(BytesIO(result.preview_bytes))
    assert preview.size == (160, 120)


def test_yellow_and_blue_visionflux_lines_are_imported_as_editable_measurements():
    def draw(draw):
        draw.line((20, 30, 70, 30), fill=(255, 210, 40), width=3)
        draw.ellipse((17, 27, 23, 33), fill=(255, 210, 40))
        draw.ellipse((67, 27, 73, 33), fill=(255, 210, 40))
        draw.line((90, 70, 130, 95), fill=(26, 220, 235), width=3)
        draw.ellipse((87, 67, 93, 73), fill=(26, 220, 235))
        draw.ellipse((127, 92, 133, 98), fill=(26, 220, 235))

    result = inspect_sem_upload(_png(draw), 'sem.png')
    assert result.is_visionflux_annotated is True
    assert len(result.measurements) == 2
    assert {m.source for m in result.measurements} == {'visionflux_auto', 'visionflux_manual'}
    assert all(m.width_px > 30 for m in result.measurements)
    preview = Image.open(BytesIO(result.preview_bytes)).convert('RGB')
    # Browser preview is neutralised so the dynamic overlay is not doubled.
    px = preview.getpixel((40, 30))
    assert max(px) - min(px) < 8


def test_small_isolated_coloured_pixels_do_not_trigger_import_mode():
    def draw(draw):
        draw.point((10, 10), fill=(255, 210, 40))
        draw.point((12, 10), fill=(26, 220, 235))

    result = inspect_sem_upload(_png(draw), 'sem.png')
    assert result.is_visionflux_annotated is False
    assert result.measurements == []


def test_coloured_square_or_label_blob_is_not_mistaken_for_measurement_line():
    def draw(draw):
        draw.rectangle((40, 40, 54, 54), fill=(255, 210, 40))
    result = inspect_sem_upload(_png(draw), 'sem.png')
    assert result.measurements == []
    assert result.is_visionflux_annotated is False

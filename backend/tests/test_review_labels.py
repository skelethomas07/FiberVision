from app.services.review_labels import supervision_label


def test_imported_blue_visionflux_measurement_remains_manual_add_when_kept():
    assert supervision_label(has_model=True, model_source='visionflux_manual', active=True, edited=False) == ('MANUAL_ADD', True, True)


def test_imported_measurement_can_still_be_removed_or_corrected():
    assert supervision_label(has_model=True, model_source='visionflux_manual', active=False, edited=False) == ('AUTO_REMOVE', None, False)
    assert supervision_label(has_model=True, model_source='visionflux_manual', active=True, edited=True) == ('MANUAL_CORRECT', True, True)

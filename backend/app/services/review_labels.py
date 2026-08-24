from __future__ import annotations


def supervision_label(*, has_model: bool, model_source: str | None, active: bool, edited: bool) -> tuple[str, bool | None, bool]:
    if not has_model:
        return "MANUAL_ADD", True, True
    if not active:
        return "AUTO_REMOVE", None, False
    if edited:
        return "MANUAL_CORRECT", True, True
    if model_source == "visionflux_manual":
        return "MANUAL_ADD", True, True
    return "AUTO_KEEP", True, True

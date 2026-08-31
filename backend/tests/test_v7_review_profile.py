from dataclasses import dataclass
from types import SimpleNamespace

from app.inference.sem_fiber_engine import _apply_fibervision_review_profile


@dataclass
class Post:
    seg_threshold: float = 0.5
    min_validity: float = 0.3
    spacing_px: float = 12.0
    boundary_tol: float = 0.35
    junction_clear_scale: float = 0.6
    min_seg_confidence: float = 0.5


def test_review_profile_relaxes_unselected_postprocessing_for_human_review():
    run = {"selection": None, "post": Post()}
    changed = _apply_fibervision_review_profile(run)
    assert changed is True
    assert run["post"].seg_threshold == 0.4
    assert run["post"].min_validity == 0.0
    assert run["post"].min_seg_confidence == 0.0
    assert run["post"].junction_clear_scale == 0.0
    assert run["post"].boundary_tol == 0.9
    assert run["post"].spacing_px == 12.0


def test_review_profile_preserves_validation_selected_settings():
    post = Post(seg_threshold=0.6, min_validity=0.4, boundary_tol=0.5)
    run = {"selection": {"selected_on_split": "val"}, "post": post}
    changed = _apply_fibervision_review_profile(run)
    assert changed is False
    assert run["post"] is post

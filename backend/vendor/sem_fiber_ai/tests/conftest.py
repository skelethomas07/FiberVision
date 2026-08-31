"""Shared fixtures for the v7 test suite (CPU-only, synthetic data)."""
from __future__ import annotations

import logging
import pathlib
import sys

import pytest

# Make ``import sem_fiber_ai`` work no matter where pytest is invoked from.
_PKG_PARENT = pathlib.Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

logging.disable(logging.INFO)

from sem_fiber_ai.tests.helpers import oracle_maps  # noqa: E402,F401  (re-exported for tests)


@pytest.fixture(scope="session")
def synth_field():
    from sem_fiber_ai.src.synthetic import make_field

    return make_field(1, H=448, W=448, n_fibres=22, n_annotations=100, image_id="F1")


@pytest.fixture(scope="session")
def synth_dataset(tmp_path_factory):
    from sem_fiber_ai.src.synthetic import write_synthetic_dataset

    root = tmp_path_factory.mktemp("synth")
    return write_synthetic_dataset(root, n_specimens=4, fields_per_specimen=2, H=192, W=192,
                                   n_annotations=30)

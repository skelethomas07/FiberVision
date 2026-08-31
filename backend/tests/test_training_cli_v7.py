import pytest

from app.training.cli import build_parser


def test_training_cli_does_not_expose_legacy_v6_retrain_command():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["retrain"])


def test_training_cli_prepare_remains_available_for_review_exports():
    args = build_parser().parse_args(["prepare"])

    assert args.command == "prepare"

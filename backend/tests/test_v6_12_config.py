from app.config import Settings


def test_default_model_settings_target_v6_12_run():
    settings = Settings(_env_file=None)

    assert settings.model_checkpoint.as_posix().endswith("models/v6_12/best_full.pt")
    assert settings.model_version == "v6.12"


def test_analysis_job_default_version_is_v6_12():
    from app.models import AnalysisJob

    default = AnalysisJob.__table__.c.model_version.default.arg
    assert default == "v6.12"

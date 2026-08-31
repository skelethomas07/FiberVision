from app.config import Settings


def test_default_model_settings_target_v7_run():
    settings = Settings(_env_file=None)

    assert settings.model_checkpoint.as_posix().endswith("models/v7/best.pt")
    assert settings.model_version == "v7.0.0"


def test_analysis_job_default_version_is_v7():
    from app.models import AnalysisJob

    default = AnalysisJob.__table__.c.model_version.default.arg
    assert default == "v7.0.0"

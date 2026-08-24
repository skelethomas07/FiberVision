from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.analyzer import AnalysisResult, run_uploaded_analysis
from pipeline.batch import BatchInput, BatchProgress, format_elapsed, run_batch
from pipeline.compute import detect_compute_backend
from pipeline.exports import build_export_bundle
from pipeline.review import (
    apply_canvas_edits,
    build_representative_lines,
    build_session_zip,
)
from pipeline.review_state import ReviewItem, build_review_item, recompute_review_item
from pipeline.resume import build_measurement_table, try_build_resume_analysis
from services.collaboration import (
    CollaborationConfig,
    SupabaseRepository,
    apply_snapshot_to_item,
    collaboration_enabled,
    serialize_snapshot,
)
from services.notifications import (
    CompletionReport,
    EmailConfig,
    send_completion_email,
    validate_email_address,
)
from ui.figures import (
    build_distribution_figure,
    build_direction_segment_figure,
    build_orientation_histogram,
    build_orientation_rose,
    build_thickness_direction_3d,
    build_thickness_direction_heatmap,
)
from ui.live_timer import live_elapsed_timer
from ui.measurement_canvas import measurement_canvas, normalize_canvas_payload

SUPPORTED_TYPES = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]


@st.cache_data(max_entries=12, show_spinner=False)
def _cached_analysis(
    image_bytes: bytes,
    filename: str,
    nm_per_px: float | None,
    max_dimension: int | None,
    prefer_gpu: bool,
    auto_calibrate: bool,
) -> AnalysisResult:
    """Run cacheable analysis without replaying Streamlit UI callbacks."""
    return run_uploaded_analysis(
        image_bytes,
        filename,
        nm_per_px=nm_per_px,
        max_dimension=max_dimension,
        prefer_gpu=prefer_gpu,
        progress_callback=None,
        auto_calibrate=auto_calibrate,
    )


def _init_state() -> None:
    defaults = {
        "batch_items": {},
        "selected_item_id": None,
        "last_batch_report": None,
        "collab_project": None,
        "collab_images": [],
        "collab_worker": "",
        "collab_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _current_item() -> ReviewItem | None:
    items: dict[str, ReviewItem] = st.session_state.batch_items
    selected = st.session_state.selected_item_id
    if selected in items:
        return items[selected]
    if items:
        selected = next(iter(items))
        st.session_state.selected_item_id = selected
        return items[selected]
    return None


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return float("nan")
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _secret_section(name: str) -> dict:
    try:
        section = st.secrets[name]
        return dict(section)
    except Exception:
        return {}


def _collaboration_config() -> CollaborationConfig | None:
    values = _secret_section("supabase")
    if not collaboration_enabled(values):
        return None
    try:
        return CollaborationConfig.from_mapping(values)
    except (TypeError, ValueError):
        return None


def _collaboration_repo() -> SupabaseRepository | None:
    config = _collaboration_config()
    if config is None:
        return None
    return SupabaseRepository(config)


def _refresh_shared_images(repo: SupabaseRepository) -> list[dict]:
    project = repo.ensure_project(name="VisionFlux Shared Review")
    images = repo.list_images(project["id"])
    st.session_state.collab_project = project
    st.session_state.collab_images = images
    return images


def _load_shared_image(
    repo: SupabaseRepository,
    row: dict,
    worker: str,
    *,
    max_dimension: int | None,
    prefer_gpu: bool,
) -> None:
    if not worker.strip():
        raise ValueError("공동 작업자 이름을 입력해 주세요.")
    lock = repo.acquire_lock(str(row["id"]), worker.strip())
    editable = bool(lock.get("acquired"))
    image_bytes = repo.download_image(str(row["storage_path"]))
    result = _cached_analysis(
        image_bytes, str(row["image_name"]), None, max_dimension, prefer_gpu, True,
    )
    detected = result.summary.get("nm_per_original_px")
    item_id = f"shared-{row['id']}"
    item = build_review_item(
        item_id, result,
        nm_per_px=None if detected is None else float(detected),
    )
    snapshot_row = repo.load_snapshot(str(row["id"]))
    if snapshot_row and snapshot_row.get("snapshot"):
        item.canvas_state = apply_snapshot_to_item(item, snapshot_row["snapshot"])
    item.collaboration_image_id = str(row["id"])
    item.collaboration_worker = worker.strip()
    item.collaboration_editable = editable
    st.session_state.batch_items[item_id] = item
    st.session_state.selected_item_id = item_id
    if editable:
        st.session_state.collab_notice = f"{row['image_name']} 편집 잠금을 획득했습니다."
    else:
        owner = lock.get("locked_by") or row.get("locked_by") or "다른 작업자"
        st.session_state.collab_notice = f"{owner} 님이 작업 중이어서 읽기 전용으로 열었습니다."


def _save_shared_item(
    item: ReviewItem,
    *,
    status: str = "in_progress",
    canvas_state: dict | None = None,
) -> bool:
    if not item.collaboration_image_id or not item.collaboration_worker:
        return False
    repo = _collaboration_repo()
    if repo is None:
        return False
    state = dict(item.canvas_state or {})
    if canvas_state:
        state.update(canvas_state)
    item.canvas_state = state
    snapshot = serialize_snapshot(
        item, worker_name=item.collaboration_worker, status=status, canvas_state=state,
    )
    repo.save_snapshot(
        image_id=item.collaboration_image_id,
        worker_name=item.collaboration_worker,
        snapshot=snapshot,
        status=status,
    )
    return True


def _upload_shared_artifacts(item: ReviewItem, export, zip_bytes: bytes, measurement_table: pd.DataFrame) -> None:
    if not item.collaboration_image_id or not item.collaboration_worker:
        return
    repo = _collaboration_repo()
    if repo is None:
        raise RuntimeError("Supabase 설정을 읽지 못했습니다.")
    stem = Path(item.analysis.image_name).stem
    repo.upload_artifact(
        image_id=item.collaboration_image_id, kind="measurement_csv",
        filename=f"{stem}_measurements.csv",
        data=measurement_table.to_csv(index=False).encode("utf-8-sig"),
        content_type="text/csv", worker_name=item.collaboration_worker,
    )
    repo.upload_artifact(
        image_id=item.collaboration_image_id, kind="labeled_png",
        filename=f"{stem}_labeled_thickness.png", data=export.annotated_labeled_png,
        content_type="image/png", worker_name=item.collaboration_worker,
    )
    repo.upload_artifact(
        image_id=item.collaboration_image_id, kind="unlabeled_png",
        filename=f"{stem}_thickness.png", data=export.annotated_unlabeled_png,
        content_type="image/png", worker_name=item.collaboration_worker,
    )
    repo.upload_artifact(
        image_id=item.collaboration_image_id, kind="review_zip",
        filename=f"{stem}_visionflux_review.zip", data=zip_bytes,
        content_type="application/zip", worker_name=item.collaboration_worker,
    )


def _email_config() -> EmailConfig | None:
    values = _secret_section("email")
    sender = str(values.get("sender", "")).strip()
    password = str(values.get("app_password", "")).strip()
    if not sender or not password:
        return None
    return EmailConfig(
        sender=sender,
        app_password=password,
        smtp_host=str(values.get("smtp_host", "smtp.gmail.com")),
        smtp_port=int(values.get("smtp_port", 465)),
        timeout_seconds=float(values.get("timeout_seconds", 20.0)),
    )


def _app_url() -> str | None:
    value = str(_secret_section("app").get("url", "")).strip()
    return value or None


def _make_batch_inputs(uploaded_files) -> list[BatchInput]:
    items: list[BatchInput] = []
    used: set[str] = set()
    for index, uploaded in enumerate(uploaded_files):
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()[:16]
        item_id = digest
        suffix = 1
        while item_id in used:
            suffix += 1
            item_id = f"{digest}-{suffix}"
        used.add(item_id)
        items.append(BatchInput(item_id=item_id, filename=uploaded.name, data=data))
    return items


def _send_batch_email(
    recipient: str,
    *,
    total: int,
    succeeded: int,
    failed_names: list[str],
    elapsed_seconds: float,
) -> tuple[str, str | None]:
    if not recipient:
        return "not_requested", None
    config = _email_config()
    if config is None:
        return "not_configured", "Streamlit Secrets에 발신 Gmail 설정이 없습니다."
    report = CompletionReport(
        total_files=total,
        succeeded_files=succeeded,
        failed_files=len(failed_names),
        elapsed_seconds=elapsed_seconds,
        completed_at=datetime.now(timezone.utc),
        failed_filenames=tuple(failed_names),
        app_url=_app_url(),
    )
    try:
        send_completion_email(config, recipient, report)
        return "sent", None
    except Exception as exc:
        return "failed", f"{type(exc).__name__}: {exc}"


def _run_batch_analysis(
    uploaded_files,
    nm_per_px: float | None,
    max_dimension: int | None,
    prefer_gpu: bool,
    recipient: str,
    auto_calibrate: bool,
) -> None:
    batch_inputs = _make_batch_inputs(uploaded_files)
    if not batch_inputs:
        return

    started_perf = time.perf_counter()
    started_unix = time.time()
    progress_bar = st.sidebar.progress(0, text="0% · 분석 준비")
    stage_slot = st.sidebar.empty()
    elapsed_slot = st.sidebar.empty()
    with st.sidebar:
        live_elapsed_timer(started_unix)

    def on_progress(event: BatchProgress) -> None:
        percent = int(round(100 * event.overall_fraction))
        progress_bar.progress(
            percent,
            text=f"{percent}% · {event.file_index}/{event.total_files} · {event.message}",
        )
        stage_slot.caption(event.filename)
        elapsed_slot.caption(f"서버 경과 시간 {format_elapsed(event.elapsed_seconds)}")

    def analyze(item: BatchInput, report) -> AnalysisResult:
        report(0.01, "이미지 확인")
        resumed = try_build_resume_analysis(
            item.data, item.filename, nm_per_px=nm_per_px,
        )
        if resumed is not None:
            count = int(resumed.summary.get("imported_measurements", 0))
            report(0.85, f"기존 측정선 {count}개 복원 · 자동 재분석 생략")
            report(1.0, "이어하기 준비 완료")
            return resumed
        report(0.04, "새 SEM 이미지 분석 준비")
        result = run_uploaded_analysis(
            item.data,
            item.filename,
            nm_per_px=nm_per_px,
            max_dimension=max_dimension,
            prefer_gpu=prefer_gpu,
            progress_callback=report,
            auto_calibrate=auto_calibrate,
        )
        report(1.0, "결과 준비 완료")
        return result

    outcomes = run_batch(batch_inputs, analyze, on_progress=on_progress)
    elapsed = time.perf_counter() - started_perf

    review_items: dict[str, ReviewItem] = {}
    failed_names: list[str] = []
    errors: dict[str, str] = {}
    for outcome in outcomes:
        if outcome.result is None:
            failed_names.append(outcome.filename)
            errors[outcome.filename] = outcome.error or "알 수 없는 오류"
            continue
        detected_nm_per_px = outcome.result.summary.get("nm_per_original_px")
        review_items[outcome.item_id] = build_review_item(
            outcome.item_id,
            outcome.result,
            nm_per_px=(None if detected_nm_per_px is None else float(detected_nm_per_px)),
            duration_seconds=outcome.duration_seconds,
        )

    email_status, email_error = _send_batch_email(
        recipient,
        total=len(outcomes),
        succeeded=len(review_items),
        failed_names=failed_names,
        elapsed_seconds=elapsed,
    )

    st.session_state.batch_items = review_items
    st.session_state.selected_item_id = next(iter(review_items), None)
    st.session_state.last_batch_report = {
        "total": len(outcomes),
        "succeeded": len(review_items),
        "failed": len(failed_names),
        "failed_names": failed_names,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "email_recipient": recipient,
        "email_status": email_status,
        "email_error": email_error,
        "requested_backend": "GPU" if prefer_gpu else "CPU",
    }
    st.session_state.pop("selected_result", None)


def _sidebar() -> None:
    st.sidebar.markdown("## 분석")
    uploaded_files = st.sidebar.file_uploader(
        "SEM 이미지",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="새 SEM은 순서대로 분석합니다. VisionFlux 노란/청록 측정선이 들어간 완료 이미지는 자동으로 이어하기 모드로 열고 재분석하지 않습니다.",
    )

    recipient = st.sidebar.text_input(
        "완료 알림 이메일",
        placeholder="name@example.com",
        help="주소를 입력하면 모든 이미지 분석이 끝난 뒤 한 번만 완료 메일을 보냅니다. 비워 두면 발송하지 않습니다.",
    ).strip()

    with st.sidebar.expander("이미지 품질 안내", expanded=False):
        st.caption(
            "분석 영역의 긴 변은 1200px 이상을 권장합니다. 가장 얇은 fiber가 "
            "최소 6px, 가능하면 8px 이상으로 보여야 edge 측정이 안정적입니다."
        )

    scale_mode = st.sidebar.radio(
        "길이 단위",
        ["스케일바 자동 감지", "nm/px 직접 입력", "픽셀 단위"],
        index=0,
        help="하단 정보 영역은 fiber 분석에서 제외합니다. 자동 감지는 스케일바와 옆 단위를 읽어 nm/px를 계산합니다.",
    )
    auto_calibrate = scale_mode == "스케일바 자동 감지"
    nm_per_px = None
    if scale_mode == "nm/px 직접 입력":
        nm_per_px = st.sidebar.number_input(
            "원본 이미지 nm/px",
            min_value=0.000001,
            value=1.0,
            format="%.6f",
            help="스케일바로 확인한 원본 이미지의 nm/px 값입니다.",
        )

    with st.sidebar.expander("분석 설정", expanded=False):
        resolution = st.radio(
            "처리 크기",
            ["빠름 · 최대 1200 px", "균형 · 최대 1600 px", "원본 · 로컬 권장"],
            index=0,
            help="Cloud에서는 1200 px가 가장 안정적입니다. 두께는 원본 픽셀 크기로 환산됩니다.",
        )
        prefer_gpu = st.toggle(
            "가능하면 GPU 사용",
            value=True,
            help="로컬 CUDA/CuPy 환경에서는 방향 tensor 계산을 GPU로 처리합니다. 사용할 수 없으면 자동으로 CPU로 전환합니다.",
        )
    max_dimension = {
        "빠름 · 최대 1200 px": 1200,
        "균형 · 최대 1600 px": 1600,
        "원본 · 로컬 권장": None,
    }[resolution]

    backend = detect_compute_backend(prefer_gpu=prefer_gpu)
    if backend.gpu_available:
        st.sidebar.caption(f"계산 장치: GPU · {backend.detail}")
    else:
        st.sidebar.caption(f"계산 장치: CPU · {backend.detail}")

    analyze_disabled = not uploaded_files
    if st.sidebar.button(
        "분석 시작",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
        help="업로드한 모든 이미지를 순서대로 분석합니다. 진행률과 경과 시간이 아래에 표시됩니다.",
    ):
        if recipient:
            try:
                recipient = validate_email_address(recipient)
            except ValueError as exc:
                st.sidebar.error(str(exc))
                return
        try:
            _run_batch_analysis(
                uploaded_files,
                nm_per_px,
                max_dimension,
                prefer_gpu,
                recipient,
                auto_calibrate,
            )
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"분석 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")

    config = _collaboration_config()
    with st.sidebar.expander("Supabase 공동 작업", expanded=config is not None):
        if config is None:
            st.caption("Streamlit Secrets에 [supabase] 설정을 추가하면 5명이 같은 이미지와 수정 결과를 공유할 수 있습니다.")
        else:
            worker = st.text_input(
                "작업자 이름", value=st.session_state.collab_worker,
                key="collab-worker-input", help="잠금과 수정 기록에 표시할 이름입니다.",
            ).strip()
            st.session_state.collab_worker = worker
            try:
                repo = _collaboration_repo()
                if repo is None:
                    raise RuntimeError("Supabase 설정을 읽지 못했습니다.")
                if st.button("공유 목록 새로고침", use_container_width=True):
                    _refresh_shared_images(repo)
                if not st.session_state.collab_images:
                    _refresh_shared_images(repo)
                if uploaded_files and st.button(
                    "업로드 파일을 공동 프로젝트에 추가", use_container_width=True,
                    disabled=not worker, help="원본 이미지를 private Supabase Storage에 저장합니다.",
                ):
                    project = st.session_state.collab_project or repo.ensure_project(name="VisionFlux Shared Review")
                    for uploaded in uploaded_files:
                        repo.upload_image(
                            project_uuid=project["id"], filename=uploaded.name,
                            data=uploaded.getvalue(), uploaded_by=worker,
                        )
                    _refresh_shared_images(repo)
                    st.success(f"{len(uploaded_files)}개 이미지를 공유했습니다.")
                shared_images = list(st.session_state.collab_images or [])
                if shared_images:
                    options = {str(row["id"]): row for row in shared_images}
                    selected_shared = st.selectbox(
                        "공유 이미지", options=list(options),
                        format_func=lambda image_id: (
                            f"{options[image_id]['image_name']} · {options[image_id].get('status', 'pending')}"
                            + (f" · {options[image_id].get('locked_by')} 작업중" if options[image_id].get('locked_by') else "")
                        ),
                    )
                    if st.button(
                        "공유 이미지 열기", use_container_width=True, disabled=not worker,
                        help="비어 있거나 만료된 잠금을 획득합니다. 잠긴 이미지는 읽기 전용으로 열립니다.",
                    ):
                        _load_shared_image(
                            repo, options[selected_shared], worker,
                            max_dimension=max_dimension, prefer_gpu=prefer_gpu,
                        )
                        st.rerun()
                else:
                    st.caption("공유 이미지가 아직 없습니다.")
            except Exception as exc:
                st.warning(f"공동 작업 연결 오류: {type(exc).__name__}: {exc}")

    items: dict[str, ReviewItem] = st.session_state.batch_items
    if items:
        st.sidebar.divider()
        ids = list(items)
        current = st.session_state.selected_item_id
        index = ids.index(current) if current in ids else 0
        selected = st.sidebar.selectbox(
            "검토할 이미지",
            options=ids,
            index=index,
            format_func=lambda item_id: items[item_id].analysis.image_name,
            key="selected_result",
            help="여러 이미지를 분석했다면 여기서 한 장씩 선택해 두께를 수정합니다.",
        )
        st.session_state.selected_item_id = selected
        st.sidebar.caption(
            "노란 선은 자동 대표 두께, 하늘색 선은 방향 기반 보완, 청록 선은 사용자가 추가한 두께입니다."
        )


def _batch_report_banner() -> None:
    report = st.session_state.last_batch_report
    if not report:
        return
    text = (
        f"{report['total']}개 분석 · 성공 {report['succeeded']}개 · "
        f"실패 {report['failed']}개 · {format_elapsed(report['elapsed_seconds'])}"
    )
    if report["failed"]:
        st.warning(text)
        with st.expander("실패한 파일 확인", expanded=False):
            for filename, error in report["errors"].items():
                st.code(f"{filename}: {error}")
    else:
        st.success(text)

    status = report.get("email_status")
    if status == "sent":
        st.caption(f"완료 알림을 {report['email_recipient']}로 보냈습니다.")
    elif status == "not_configured":
        st.info("분석 결과는 저장되었습니다. 완료 메일을 보내려면 Streamlit Secrets에 발신 Gmail을 설정하세요.")
    elif status == "failed":
        st.warning(f"분석은 완료됐지만 이메일 발송에 실패했습니다: {report.get('email_error')}")


def _set_item_calibration(item: ReviewItem, nm_per_px: float | None) -> None:
    item.nm_per_px = None if nm_per_px is None else float(nm_per_px)
    if "width_original_px" in item.measurements.columns:
        values = pd.to_numeric(item.measurements["width_original_px"], errors="coerce")
        item.measurements["width_nm"] = (
            np.nan if item.nm_per_px is None else values * item.nm_per_px
        )
    item.revision += 1
    recompute_review_item(item)


def _image_info_panel(item: ReviewItem) -> None:
    analysis = item.analysis
    quality = analysis.quality
    summary = analysis.summary
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "이미지 품질",
        quality.label if quality is not None else "—",
        help="해상도, 선명도, 대비와 포화 픽셀을 종합한 안내값입니다. 측정 합격/불합격 판정은 아닙니다.",
    )
    removed = int(summary.get("footer_removed_px", 0) or 0)
    c2.metric(
        "분석 제외 하단",
        f"{removed}px" if removed else "없음",
        help="배율, 전압, 날짜와 스케일바가 있는 하단 정보 영역은 fiber 검출에서 제외됩니다.",
    )
    scale_text = "픽셀 단위" if item.nm_per_px is None else f"{item.nm_per_px:.6f} nm/px"
    c3.metric(
        "길이 보정",
        scale_text,
        help="자동 스케일바 인식값 또는 사용자가 수정한 원본 이미지 기준 nm/px입니다.",
    )

    with st.expander("화질과 스케일 확인", expanded=False):
        if quality is not None:
            st.write(
                f"분석 영역: {quality.width_px}×{quality.height_px}px · "
                f"대비 {quality.contrast:.2f} · 선명도 {quality.sharpness:.1f}"
            )
            if quality.estimated_min_fiber_width_px is not None:
                st.write(f"추정 최소 구조 폭: 약 {quality.estimated_min_fiber_width_px:.1f}px")
            for message in quality.messages:
                st.warning(message)
            if not quality.messages:
                st.success("권장 해상도·선명도·대비 조건을 대체로 만족합니다.")

        calibration = analysis.calibration
        if calibration is not None and calibration.bar_length_px is not None:
            label = (
                f"{calibration.scale_value:g} {calibration.scale_unit}"
                if calibration.scale_value is not None and calibration.scale_unit
                else "단위 인식 실패"
            )
            st.write(
                f"스케일바 감지 길이: {calibration.bar_length_px:.1f}px · "
                f"표시값: {label} · 신뢰도 {calibration.confidence:.2f}"
            )
        corrected = st.number_input(
            "이 이미지의 원본 nm/px",
            min_value=0.000001,
            value=float(item.nm_per_px or 1.0),
            format="%.6f",
            key=f"calibration-{item.item_id}",
            help="자동 감지값이 틀렸다면 수정한 뒤 아래 버튼을 누르세요.",
        )
        a, b = st.columns(2)
        if a.button("보정값 반영", key=f"apply-cal-{item.item_id}", use_container_width=True):
            _set_item_calibration(item, float(corrected))
            st.rerun()
        if b.button("픽셀 단위로 전환", key=f"clear-cal-{item.item_id}", use_container_width=True):
            _set_item_calibration(item, None)
            st.rerun()


def _summary_metrics(item: ReviewItem) -> None:
    reps = item.representatives
    measurements = item.measurements
    use_nm = item.nm_per_px is not None
    value_col = "representative_width_nm" if use_nm else "representative_width_original_px"
    values = pd.to_numeric(reps.get(value_col, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    weights = pd.to_numeric(
        reps.get("fiber_count_weight", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).to_numpy(float)
    median = _weighted_median(values, weights) if len(values) else float("nan")
    manual_count = int((measurements.get("source", pd.Series(dtype=str)).astype(str) == "manual").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 이미지", item.analysis.image_name, help="현재 두께와 방향 결과를 확인하고 있는 이미지입니다.")
    c2.metric("대표 두께 수", int(len(reps)), help="같은 fiber 영역의 반복 측정은 대표값으로 묶습니다.")
    c3.metric(
        "중앙 두께",
        "—" if not np.isfinite(median) else f"{median:.2f} {'nm' if use_nm else 'px'}",
        help="각 fiber 영역이 같은 총 가중치를 갖도록 계산한 중앙값입니다.",
    )
    c4.metric("수동 추가", manual_count, help="두 점 클릭 후 전체 반영한 수동 두께선 수입니다.")


def _handle_canvas_result(result, item: ReviewItem) -> None:
    if result is None:
        return

    raw_autosave = getattr(result, "autosave", None)
    if raw_autosave is not None:
        autosave_payload = normalize_canvas_payload(raw_autosave)
        token = json.dumps(autosave_payload, sort_keys=True, default=str)
        token_key = f"autosave-token-{item.item_id}"
        if token != st.session_state.get(token_key):
            item.canvas_state = dict(autosave_payload.get("canvas_state", {}) or {})
            if item.collaboration_image_id and item.collaboration_editable:
                try:
                    _save_shared_item(item, canvas_state=item.canvas_state)
                    st.toast("공유 프로젝트에 5분 자동저장했습니다.")
                except Exception as exc:
                    st.warning(f"공유 자동저장 실패: {type(exc).__name__}: {exc}")
            st.session_state[token_key] = token

    raw_payload = getattr(result, "apply", None)
    if raw_payload is None:
        return
    payload = normalize_canvas_payload(raw_payload)
    if not payload["new_measurements"] and not payload["delete_ids"]:
        return
    token = f"{item.revision}:" + json.dumps(payload, sort_keys=True, default=str)
    if token == item.last_apply_token:
        return
    updated, events = apply_canvas_edits(
        item.measurements, payload["new_measurements"], payload["delete_ids"],
        analysis_scale=item.analysis.analysis_scale, nm_per_px=item.nm_per_px,
    )
    item.measurements = updated
    item.feedback.extend(events)
    item.last_apply_token = token
    item.revision += 1
    state = dict(payload.get("canvas_state", {}) or {})
    state.update({"revision": item.revision, "pending": [], "delete_ids": [], "savedAt": int(time.time() * 1000)})
    item.canvas_state = state
    recompute_review_item(item)
    if item.collaboration_image_id and item.collaboration_editable:
        try:
            _save_shared_item(item, canvas_state=state)
        except Exception as exc:
            st.warning(f"공유 저장 실패: {type(exc).__name__}: {exc}")
    st.toast(f"수동 측정 {len(payload['new_measurements'])}개 추가·수정 · 기존 표시 {len(payload['delete_ids'])}개 삭제")
    st.rerun()


def _thickness_tab(item: ReviewItem) -> None:
    analysis = item.analysis
    reps = item.representatives
    representative_lines = build_representative_lines(item.measurements, reps)

    if bool(analysis.summary.get("resume_mode")):
        imported = int(analysis.summary.get("imported_measurements", 0))
        st.info(f"이어하기 모드 · 기존 측정선 {imported}개를 복원했습니다. 자동 두께 탐지는 다시 실행하지 않았습니다.")

    with st.expander("측정값 고치는 방법", expanded=False):
        st.markdown(
            """
1. 마우스 휠로 확대하고 **이동·선택**으로 원하는 위치를 찾습니다.  
2. **두께 추가**에서 edge 위에 1.5초 머물면 모델이 검출한 fiber 경로만 강조됩니다. 첫 edge를 클릭하면 가능한 경우 법선이 표시됩니다.  
3. **두께 수정**에서 기존 선의 한쪽 끝을 클릭하면 같은 방향의 안내선이 표시됩니다. 새 edge를 클릭해 교체합니다.  
4. 잘못된 선은 **지우개**로 선택한 뒤 **전체 반영**을 누릅니다.
            """
        )

    canvas_result = measurement_canvas(
        analysis.image,
        representative_lines,
        analysis_scale=analysis.analysis_scale,
        nm_per_px=item.nm_per_px,
        revision=item.revision,
        key=f"visionflux-canvas-{item.item_id}",
        autosave_key=f"{item.item_id}-{analysis.image_name}",
        initial_state=item.canvas_state,
        editable=item.collaboration_editable,
        hover_delay_ms=1500,
    )
    _handle_canvas_result(canvas_result, item)

    st.markdown("### 두께 분포")
    use_nm = item.nm_per_px is not None
    st.plotly_chart(
        build_distribution_figure(reps, use_nm=use_nm),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    stem = Path(analysis.image_name).stem
    export_image = analysis.original_image if analysis.original_image is not None else analysis.image
    export = build_export_bundle(
        export_image,
        representative_lines,
        analysis_scale=analysis.analysis_scale,
        nm_per_px=item.nm_per_px,
        image_coordinates_are_original=analysis.original_image is not None,
    )
    measurement_table = build_measurement_table(
        representative_lines,
        analysis_scale=analysis.analysis_scale,
        image_coordinates_are_original=analysis.original_image is not None,
    )
    unit_metadata = {
        "coordinates": "original image pixels",
        "width": "original image pixels",
        "angle": "local fiber tangent direction in degrees, axial [-90, 90)",
        "source": "auto or manual",
        "status": "keep, corrected, or ambiguous for visible measurements; removed measurements are omitted",
    }
    zip_bytes = build_session_zip(
        analysis.image_name,
        item.measurements,
        reps,
        item.feedback,
        analysis_summary=analysis.summary,
        measurement_table=measurement_table,
        annotated_png=export.annotated_labeled_png,
        annotated_unlabeled_png=export.annotated_unlabeled_png,
        unit_metadata=unit_metadata,
    )
    st.caption(
        "CSV는 원본 이미지 픽셀 좌표를 저장합니다. "
        "measurement_id, x1, y1, x2, y2, center_x, center_y, width, angle, source, status 순서입니다."
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "측정 데이터 CSV", measurement_table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}_measurements.csv", mime="text/csv", use_container_width=True,
        help="현재 화면에 남아 있는 대표 두께선을 좌표·폭·방향·출처·상태와 함께 저장합니다.",
    )
    d2.download_button(
        "라벨 포함 이미지", export.annotated_labeled_png,
        file_name=f"{stem}_labeled_thickness.png", mime="image/png", use_container_width=True,
        help="최종 두께선과 연속 라벨을 함께 표시합니다.",
    )
    d3.download_button(
        "라벨 없는 이미지", export.annotated_unlabeled_png,
        file_name=f"{stem}_thickness.png", mime="image/png", use_container_width=True,
        help="최종 두께선만 표시하고 라벨은 숨깁니다.",
    )
    d4.download_button(
        "전체 결과 ZIP", zip_bytes, file_name=f"{stem}_visionflux_review.zip",
        mime="application/zip", use_container_width=True,
        help="11컬럼 측정 CSV, 두 종류의 표시 이미지, 수정 기록을 함께 저장합니다.",
    )

    if item.collaboration_image_id:
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button(
            "공유 프로젝트에 지금 저장", key=f"shared-save-{item.item_id}",
            use_container_width=True, disabled=not item.collaboration_editable,
        ):
            try:
                _save_shared_item(item, status="in_progress")
                _upload_shared_artifacts(item, export, zip_bytes, measurement_table)
                st.success("Supabase에 스냅샷과 결과 파일을 저장했습니다.")
            except Exception as exc:
                st.error(f"공유 저장 실패: {type(exc).__name__}: {exc}")
        if c2.button(
            "작업 완료 및 잠금 해제", key=f"shared-done-{item.item_id}",
            use_container_width=True, disabled=not item.collaboration_editable, type="primary",
        ):
            try:
                _save_shared_item(item, status="done")
                _upload_shared_artifacts(item, export, zip_bytes, measurement_table)
                repo = _collaboration_repo()
                if repo is not None:
                    repo.release_lock(item.collaboration_image_id, item.collaboration_worker or "", completed=True)
                item.collaboration_editable = False
                st.success("완료 처리하고 잠금을 해제했습니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"완료 처리 실패: {type(exc).__name__}: {exc}")


def _orientation_tab(item: ReviewItem) -> None:
    analysis = item.analysis
    orientation = analysis.orientation
    if orientation is None:
        if bool(analysis.summary.get("resume_mode")):
            st.info("이어하기 모드에서는 기존 측정선을 복원하고 수동 추가만 이어가므로 방향 분석을 다시 실행하지 않습니다.")
        else:
            st.info("방향 결과가 없습니다. 이미지를 다시 분석해 주세요.")
        return
    representative_lines = build_representative_lines(item.measurements, item.representatives)
    status = (
        item.measurements["status"].astype(str)
        if "status" in item.measurements.columns
        else pd.Series("active", index=item.measurements.index)
    )
    active = item.measurements[status.isin(["active", "accepted"])]
    path_count = int(active["fiber_path_id"].nunique()) if "fiber_path_id" in active.columns else 0
    segment_count = int(active[["fiber_path_id", "direction_segment_id"]].drop_duplicates().shape[0]) if {"fiber_path_id", "direction_segment_id"}.issubset(active.columns) else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "주방향",
        "—" if not np.isfinite(orientation.dominant_direction_deg) else f"{orientation.dominant_direction_deg:+.1f}°",
        help="Coherency와 energy 조건을 통과한 구조 tensor를 모두 합쳐 계산한 방향입니다.",
    )
    m2.metric(
        "정렬도 S",
        "—" if not np.isfinite(orientation.order_parameter) else f"{orientation.order_parameter:.3f}",
        help="0은 방향이 고르게 퍼진 상태, 1은 거의 한 방향으로 정렬된 상태입니다.",
    )
    m3.metric(
        "검출 경로",
        path_count,
        help="Centerline graph에서 분리된 연속 fiber 경로 수입니다. 교차점에서는 경로가 나뉠 수 있습니다.",
    )
    m4.metric(
        "방향 구간",
        segment_count,
        help="긴 fiber가 휘어 방향이 지속적으로 달라지면 같은 경로 안에서 여러 방향 구간으로 나눕니다.",
    )

    st.caption(
        f"계산 장치: {orientation.compute_backend} · {orientation.compute_backend_detail} · "
        "0°는 수평이며 fiber 방향은 180° 주기입니다."
    )
    st.image(
        orientation.color_map,
        caption="색상은 방향, 채도는 방향 신뢰도(coherency), 밝기는 원본 SEM 명암을 나타냅니다.",
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        build_orientation_histogram(orientation),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    c2.plotly_chart(
        build_orientation_rose(orientation),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        build_direction_segment_figure(item.measurements),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("### 두께·방향·개수")
    use_nm = item.nm_per_px is not None
    g1, g2 = st.tabs(["3D 그래프", "2D Heatmap"])
    with g1:
        st.plotly_chart(
            build_thickness_direction_3d(representative_lines, use_nm=use_nm),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )
    with g2:
        st.plotly_chart(
            build_thickness_direction_heatmap(representative_lines, use_nm=use_nm),
            use_container_width=True,
            config={"displayModeBar": False},
        )


def _collaboration_tab(item: ReviewItem) -> None:
    config = _collaboration_config()
    if config is None:
        st.info("Streamlit Secrets에 Supabase 설정을 추가하면 공동 작업 현황이 표시됩니다.")
        return
    repo = _collaboration_repo()
    if repo is None:
        st.warning("Supabase 연결을 만들지 못했습니다.")
        return
    if st.button("현황 새로고침", key="collab-main-refresh"):
        try:
            _refresh_shared_images(repo)
        except Exception as exc:
            st.error(f"새로고침 실패: {type(exc).__name__}: {exc}")
    rows = list(st.session_state.collab_images or [])
    if rows:
        table = pd.DataFrame([{
            "이미지": row.get("image_name"),
            "상태": row.get("status"),
            "작업자": row.get("locked_by") or "",
            "업데이트": row.get("updated_at"),
        } for row in rows])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.caption("공유 이미지가 없습니다.")
    if item.collaboration_image_id:
        mode = "편집 가능" if item.collaboration_editable else "읽기 전용"
        st.info(f"현재 이미지: {item.analysis.image_name} · {mode} · 작업자 {item.collaboration_worker or '—'}")


def main() -> None:
    st.set_page_config(
        page_title="VisionFlux — Surface to Volume",
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    st.title("VisionFlux")
    st.markdown("#### Surface to Volume")
    st.caption("SEM 이미지에서 fiber 두께와 방향을 정리해 3D 구조 생성에 사용할 입력 분포를 만듭니다.")

    _sidebar()
    _batch_report_banner()
    if st.session_state.collab_notice:
        st.info(st.session_state.collab_notice)
        st.session_state.collab_notice = None
    item = _current_item()
    if item is None:
        st.info("왼쪽에서 SEM 이미지를 올린 뒤 **분석 시작**을 눌러 주세요.")
        return

    _image_info_panel(item)
    _summary_metrics(item)
    thickness_tab, orientation_tab, collaboration_tab = st.tabs(["두께", "방향", "공동 작업"])
    with thickness_tab:
        _thickness_tab(item)
    with orientation_tab:
        _orientation_tab(item)
    with collaboration_tab:
        _collaboration_tab(item)


if __name__ == "__main__":
    main()

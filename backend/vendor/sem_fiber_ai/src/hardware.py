"""Hardware detection and performance-only adaptation (v7).

The scientific protocol -- data split, model definition, target construction,
loss, tile size, EFFECTIVE batch size, epochs, patience, evaluation criteria --
never depends on what is in this module.  Hardware may only change:

* micro-batch size and gradient-accumulation steps (effective batch fixed);
* mixed-precision dtype (bf16 > fp16+GradScaler > fp32);
* data-loader workers, pinned memory, persistent workers;
* inference tile size / tiles per forward pass (blending keeps results invariant);
* optional ``torch.compile`` (tested against eager output; falls back).
"""
from __future__ import annotations

import gc
import os
from dataclasses import asdict, dataclass
from typing import Any

from .utils import get_logger

LOG = get_logger(__name__)


@dataclass
class Hardware:
    device: str = "cpu"             # "cuda" | "cpu"
    gpu_name: str | None = None
    vram_gb: float = 0.0
    cuda_version: str | None = None
    compute_capability: tuple[int, int] | None = None
    bf16_supported: bool = False
    cpu_count: int = 1
    profile: str = ""               # "a100" | "l4" | "t4" | "gpu_generic" | "cpu"

    def __post_init__(self) -> None:
        if not self.profile:
            self.profile = _profile_name(self.device, self.gpu_name, self.vram_gb)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability) if self.compute_capability else None
        return d


def _profile_name(device: str, gpu_name: str | None, vram_gb: float = 0.0) -> str:
    if device != "cuda":
        return "cpu"
    low = (gpu_name or "").lower()
    if "a100" in low or "h100" in low or ("a10" in low and vram_gb > 30):
        return "a100"
    if "l4" in low:
        return "l4"
    if "t4" in low:
        return "t4"
    return "gpu_generic"


def detect() -> Hardware:
    import torch

    cpu = os.cpu_count() or 1
    if not torch.cuda.is_available():
        return Hardware("cpu", None, 0.0, None, None, False, cpu, "cpu")
    props = torch.cuda.get_device_properties(0)
    name = props.name
    vram = float(props.total_memory) / 2 ** 30
    cc = (int(props.major), int(props.minor))
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:                                    # noqa: BLE001
        bf16 = cc[0] >= 8
    prof = _profile_name("cuda", name, vram)
    return Hardware("cuda", name, vram, getattr(torch.version, "cuda", None), cc, bf16, cpu, prof)


#: performance profiles -- identical scientific config, different throughput knobs
PROFILES: dict[str, dict[str, Any]] = {
    "a100": {"micro_batch_candidates": (16, 8, 4, 2, 1), "num_workers": 4,
             "infer_tile": 1024, "infer_tile_batch": 8},
    "l4": {"micro_batch_candidates": (8, 4, 2, 1), "num_workers": 4,
           "infer_tile": 768, "infer_tile_batch": 4},
    "t4": {"micro_batch_candidates": (8, 4, 2, 1), "num_workers": 2,
           "infer_tile": 512, "infer_tile_batch": 4},
    "gpu_generic": {"micro_batch_candidates": (8, 4, 2, 1), "num_workers": 2,
                    "infer_tile": 512, "infer_tile_batch": 2},
    "cpu": {"micro_batch_candidates": (2, 1), "num_workers": 0,
            "infer_tile": 512, "infer_tile_batch": 1},
}


def profile_for(hw: Hardware) -> dict[str, Any]:
    p = dict(PROFILES[hw.profile])
    p["num_workers"] = min(p["num_workers"], max(0, hw.cpu_count - 1))
    return p


def choose_precision(hw: Hardware, preference: str = "auto") -> str:
    """bf16 when reliably supported, else fp16 (+GradScaler), else fp32."""
    if hw.device != "cuda":
        return "fp32"
    if preference in ("bf16", "fp16", "fp32"):
        if preference == "bf16" and not hw.bf16_supported:
            LOG.warning("bf16 requested but not supported on %s; using fp16", hw.gpu_name)
            return "fp16"
        return preference
    if hw.bf16_supported:
        return "bf16"
    return "fp16"


def autocast_dtype(precision: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


def is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cudnn_status_alloc_failed" in msg


def clear_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def probe_micro_batch(model, *, tile: int, candidates, device, precision: str,
                      effective_batch: int, loss_fn=None, target_maker=None) -> int:
    """Largest micro-batch (<= effective batch) whose forward+backward fits.

    Runs a real training step on random tiles for each candidate, largest
    first, and returns the first that survives.  Never changes tile or model.
    """
    import torch

    model.train()
    dtype = autocast_dtype(precision)
    for mb in candidates:
        mb = int(mb)
        if mb > effective_batch or effective_batch % mb != 0:
            continue
        try:
            clear_cuda()
            x = torch.randn(mb, 1, tile, tile, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(x)
                if loss_fn is not None and target_maker is not None:
                    loss, _ = loss_fn(out, target_maker(mb, tile, device))
                else:
                    loss = sum(v.float().pow(2).mean() for v in out.values())
            loss.backward()
            model.zero_grad(set_to_none=True)
            del x, out, loss
            clear_cuda()
            LOG.info("VRAM probe: micro-batch %d fits at tile %d (%s)", mb, tile, precision)
            return mb
        except Exception as exc:                          # noqa: BLE001
            if is_oom(exc):
                LOG.info("VRAM probe: micro-batch %d OOM at tile %d", mb, tile)
                model.zero_grad(set_to_none=True)
                clear_cuda()
                continue
            raise
    raise RuntimeError(f"no micro-batch in {list(candidates)} fits at tile {tile}; "
                       "the protocol tile size is fixed -- use a larger GPU")


def try_compile(model, *, example: "Any", tol: float = 1e-2):
    """Return (compiled_or_eager_model, note).  Compiled output must match eager."""
    import torch

    if not hasattr(torch, "compile"):
        return model, "torch.compile unavailable"
    try:
        model.eval()
        with torch.no_grad():
            ref = model(example)
        cm = torch.compile(model)
        with torch.no_grad():
            got = cm(example)
        worst = max(float((got[k].float() - ref[k].float()).abs().max()) for k in ref)
        if worst > tol:
            LOG.warning("torch.compile output differs from eager by %.4g > %.4g; using eager",
                        worst, tol)
            return model, f"compile rejected (max diff {worst:.3g})"
        return cm, f"compiled (max diff vs eager {worst:.2e})"
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("torch.compile failed (%s); using eager", exc)
        return model, f"compile failed: {type(exc).__name__}"


def cuda_memory_stats() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"peak_allocated_gb": float(torch.cuda.max_memory_allocated() / 2 ** 30),
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / 2 ** 30)}

"""Opt-in inference acceleration with automatic fallback to the stock bf16 path.

Parts (each gated separately; see README "Acceleration"):
  fp8_conv_lq   LQ projector conv3d in FP8 (cuDNN graph API)
  fp8_dit       DiT linears / FFN in FP8 (torch._scaled_mm + Triton quantization)
  fused_dit     DiT RMSNorm+RoPE and AdaLN fused Triton kernels (bf16)
  fp8_conv_tcd  TCDecoder convolutions in FP8 (cuDNN graph API) -- opt-in only

Environment: FLASHVSR_ACCEL=1 requests the first three (run.py --accel sets it);
FLASHVSR_FP8_CONV / FLASHVSR_FP8_DIT / FLASHVSR_FUSED_DIT =1 request one of them,
=0 removes it even under FLASHVSR_ACCEL=1. fp8_conv_tcd is left out of
FLASHVSR_ACCEL and runs only with FLASHVSR_FP8_CONV_TCD=1: the TCDecoder
synthesizes the output pixels, and FP8's 3-bit mantissa turns smooth feature
gradients into steps, visible as contour banding on skin and other flat areas
(its warping-error / flat-area error was 2-3x that of the other parts).
Nothing set: nothing changes.

Every part changes the output, so a part that can't run must leave the stock
code path untouched (not a rewritten bf16 imitation of the fast path): a run
whose parts all fell back is bit-identical to a run without acceleration.

  preflight()  cheap checks, no model needed (env, CUDA, sm89+, bf16, imports,
               probe builds) -> the plan. Runs before the --resume manifest.
  install()    patch a built pipeline and warm up every part -> the active set.
  demote()     a part failed at runtime: restore the stock path for it for the
               rest of the process. Called by the parts themselves: tiny-long's
               __call__ swallows exceptions, so the caller never sees them.
"""
import os

from .common import log

import torch  # noqa: E402  (vsrlib.common first: allocator env var)

# Bump a part's version whenever its output changes (kernels, calibration table):
# the version is part of the --resume manifest, so tiles made by an older
# implementation are not reused.
PART_VERSIONS = {"fp8_conv_tcd": 1, "fp8_conv_lq": 1, "fp8_dit": 1, "fused_dit": 1}
PARTS = tuple(PART_VERSIONS)
_ENV = {"fp8_conv_tcd": "FLASHVSR_FP8_CONV_TCD", "fp8_conv_lq": "FLASHVSR_FP8_CONV",
        "fp8_dit": "FLASHVSR_FP8_DIT", "fused_dit": "FLASHVSR_FUSED_DIT"}
_OPT_IN = frozenset({"fp8_conv_tcd"})  # never enabled by FLASHVSR_ACCEL / enable_all
_CONV = ("fp8_conv_tcd", "fp8_conv_lq")
_DEFAULT_SHAPE = (512, 512)  # warmup output size when the caller doesn't know it


class _State:
    def __init__(self):
        self.undo = {}        # part -> callable restoring the stock path
        self.release = []     # callables freeing resources without restoring
        self.demoted = set()
        self.warned = set()
        self.warming = False  # install()'s warmup handles its own failures


_state = _State()


def _warn(msg):
    if msg not in _state.warned:
        _state.warned.add(msg)
        log(msg, message_type='warning')


def _device_capability(dev):
    return torch.cuda.get_device_capability(dev)


def requested_parts(enable_all=False):
    """Parts requested by the environment (or by enable_all), minus =0. Opt-in
    parts need their own variable set to 1."""
    all_on = enable_all or os.environ.get("FLASHVSR_ACCEL") == "1"
    parts = set()
    for p in PARTS:
        v = os.environ.get(_ENV[p])
        if v == "1" or (all_on and v != "0" and p not in _OPT_IN):
            parts.add(p)
    return parts


def preflight(device, dtype, mode, version, enable_all=False):
    """Checks 1-5 of the design, cheapest first. Returns the frozenset of parts
    that may be installed; every dropped part gets a one-line warning."""
    parts = requested_parts(enable_all)
    if not parts:
        return frozenset()

    def drop(which, reason):
        which = [p for p in PARTS if p in which and p in parts]
        if which:
            parts.difference_update(which)
            _warn(f"[FlashVSR] accel: {', '.join(which)} disabled: {reason}; using the standard path.")

    dev = torch.device(device)
    if str(version) != "11":
        # Validated (quality gate, calibration) on FlashVSR v1.1 only.
        drop(PARTS, "supports FlashVSR v1.1 (-v 11) only")
    elif dev.type != "cuda":
        drop(PARTS, "needs a CUDA device")
    elif dtype != torch.bfloat16:
        drop(PARTS, "needs --dtype bf16")
    else:
        cc = _device_capability(dev)
        if cc < (8, 9):
            drop(PARTS, f"needs an FP8-capable GPU (sm89+, RTX 40 series or newer), found sm{cc[0]}{cc[1]}")
    if mode == "full":
        parts.discard("fp8_conv_tcd")  # full mode decodes with the Wan VAE, not the TCDecoder
    if parts & set(_CONV):
        try:
            from . import fp8_conv
            if not fp8_conv.has_table(version):
                drop(_CONV, f"no FP8 calibration table for model version {version}")
            else:
                fp8_conv.probe(dev)
        except Exception as e:
            drop(_CONV, f"cuDNN FP8 convolution unavailable ({type(e).__name__}: {e})")
    if "fp8_dit" in parts:
        try:
            from . import accel_dit
            accel_dit.probe_scaled_mm(dev)
        except Exception as e:
            drop(("fp8_dit",), f"FP8 GEMM unavailable ({type(e).__name__}: {e})")
    return frozenset(parts)


def _guard(part):
    def wrap(fn):
        def guarded(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except torch.OutOfMemoryError:
                raise  # transient, and the FP8 path uses less memory than the fallback
            except Exception as e:
                if not _state.warming:
                    demote(part, e)
                raise
        return guarded
    return wrap


def demote(part, exc=None):
    """Restore the stock path for `part` for the rest of this process."""
    undo = _state.undo.pop(part, None)
    if undo is None:
        return
    try:
        undo()
    finally:
        _state.demoted.add(part)
        if exc is not None:
            _warn(f"[FlashVSR] accel: {part} failed at runtime ({type(exc).__name__}: {exc}); switched "
                  f"to the standard path for the rest of this run. To turn it off: {_ENV[part]}=0")


def _forget():
    """Drop the previous install without restoring it (its pipeline is being
    replaced); global hooks are reset so they can't leak into the new one."""
    from src.models import wan_video_dit
    wan_video_dit.FUSED_ROPE_FN = None
    wan_video_dit.FUSED_ADALN = None
    for rel in _state.release:
        try:
            rel()
        except Exception:
            pass
    _state.undo.clear()
    _state.release.clear()
    _state.demoted.clear()


def _install_part(part, pipe, version, device, guard):
    if part == "fp8_dit":
        from .accel_dit import install_fp8_dit
        return install_fp8_dit(pipe.dit, device, guard), None
    if part == "fused_dit":
        from .accel_dit import install_fused_dit
        undo = install_fused_dit(guard)
        return undo, undo
    from . import fp8_conv
    if part == "fp8_conv_lq":
        undo, runner = fp8_conv.install_lq(pipe.dit.LQ_proj_in, version, device, guard)
    else:
        undo, runner = fp8_conv.install_tcd(pipe.TCDecoder, version, device, guard)
    return undo, runner.release


@torch.no_grad()
def _warmup(part, pipe, shape):
    th, tw = shape
    dev = pipe.device
    if part in ("fp8_dit", "fused_dit"):
        from .accel_dit import warmup_dit
        warmup_dit(pipe.dit, 2 * (th // 16) * (tw // 16))  # tokens of a steady 2-latent-frame chunk
    elif part == "fp8_conv_lq":
        lq = pipe.dit.LQ_proj_in
        lq.clear_cache()
        for t in (1, 4, 4):  # first chunk (cache warm-up) and two steady chunks
            lq.stream_forward(torch.zeros(1, 3, t, th, tw, dtype=torch.bfloat16, device=dev))
        lq.clear_cache()
    else:
        tcd = pipe.TCDecoder
        tcd.clean_mem()
        tcd.decode_video(torch.zeros(1, 2, 16, th // 8, tw // 8, dtype=torch.bfloat16, device=dev),
                         parallel=False, cond=torch.zeros(1, 3, 8, th, tw, dtype=torch.bfloat16, device=dev))
        tcd.clean_mem()
    torch.cuda.synchronize(dev)


def install(pipe, plan, version, shape=None):
    """Install the planned parts on a built pipeline (check 6: warmup). Returns
    the frozenset of parts actually active. Always call it — also with an empty
    plan — so hooks left by a previous pipeline in this process are cleared."""
    _forget()
    if not plan:
        return frozenset()
    shape = shape or _DEFAULT_SHAPE
    device = pipe.device
    active = set()
    for part in PARTS:
        if part not in plan:
            continue
        if part == "fp8_conv_tcd" and getattr(pipe, "TCDecoder", None) is None:
            continue
        try:
            undo, release = _install_part(part, pipe, version, device, _guard(part))
            _state.undo[part] = undo
            if release is not None:
                _state.release.append(release)
            _state.warming = True
            try:
                _warmup(part, pipe, shape)
            finally:
                _state.warming = False
            active.add(part)
        except Exception as e:
            undo = _state.undo.pop(part, None)
            if undo is not None:
                undo()
            _warn(f"[FlashVSR] accel: {part} disabled: warmup failed ({type(e).__name__}: {e}); "
                  f"using the standard path.")
    if active:
        log(f"[FlashVSR] accel: enabled {', '.join(p for p in PARTS if p in active)}.", message_type='info')
    return frozenset(active)


def active_parts():
    return frozenset(_state.undo)


def demoted_parts():
    return frozenset(_state.demoted)


def manifest_value(parts):
    """Value for the --resume manifest's "accel" key: versioned part names, or
    None when nothing is active (the key is then omitted, keeping the hash of
    non-accelerated runs unchanged)."""
    if not parts:
        return None
    return [f"{p}@{PART_VERSIONS[p]}" for p in PARTS if p in parts]


def failure_hint():
    """Extra text for a failed-run error when a part was demoted during it."""
    if not _state.demoted:
        return ""
    envs = sorted({f"{_ENV[p]}=0" for p in _state.demoted})
    return (f" Acceleration part(s) {', '.join(sorted(_state.demoted))} failed and were switched off for "
            f"this run. Rerunning with the same settings retries them; to turn them off set "
            f"{' '.join(envs)} (tiled tiny-long: --resume reuses completed tiles only when the "
            f"acceleration settings match).")

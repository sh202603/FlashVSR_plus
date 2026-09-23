"""DiT acceleration parts (FP8 linears, fused RMSNorm+RoPE / AdaLN), installed by
vsrlib.accel on top of the vendored flashvsr-sm89-ops kernels (vsrlib/sm89_ops)."""
import torch
import torch.nn as nn
import triton

from src.models import wan_video_dit
from .sm89_ops.fp8_linear import FP8Linear
from .sm89_ops.fp8_ffn import FP8FFN
from .sm89_ops.fused_adaln import fused_ln_modulate, fused_gate_add
from .sm89_ops.fused_rms_rope import _rms_rope_kernel, _freqs_cos_sin


def probe_scaled_mm(device):
    """Raise unless cuBLASLt provides tensorwise-scaled FP8 GEMM here."""
    a = torch.zeros(32, 64, dtype=torch.float8_e4m3fn, device=device)
    b = torch.zeros(32, 64, dtype=torch.float8_e4m3fn, device=device).t()
    one = torch.ones((), dtype=torch.float32, device=device)
    torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)


class _FP8Linear(FP8Linear):
    """FP8Linear that takes part in the pipeline's CPU offload (onload/offload)
    and reports runtime failures through `guard`."""

    def __init__(self, linear, device, guard):
        super().__init__(linear)
        self._device = device
        self._forward = guard(super().forward)

    def offload(self):
        self.to("cpu")

    def onload(self):
        self.to(self._device)

    def forward(self, x):
        return self._forward(x)


class _FP8FFN(FP8FFN):
    def __init__(self, lin0, lin2, guard):
        super().__init__(lin0, lin2)
        self._forward = guard(super().forward)

    def forward(self, x):
        return self._forward(x)


def install_fp8_dit(dit, device, guard):
    """Swap each block's self_attn.{q,k,v,o}, cross_attn.{q,o} and FFN for FP8
    versions. cross_attn.k/v stay bf16: they only build the text KV cache in
    init_cross_kv, which the pipeline may re-run. The bf16 originals are parked
    on the CPU for the undo. Returns an undo callable."""
    swapped = []  # (parent, name, original)

    def swap(parent, name, new):
        orig = getattr(parent, name)
        setattr(parent, name, new)
        orig.to("cpu")  # one module at a time: no bf16+fp8 peak for the whole DiT
        swapped.append((parent, name, orig))

    def undo():
        for parent, name, orig in reversed(swapped):
            orig.to(device)
            setattr(parent, name, orig)
        swapped.clear()

    try:
        for blk in dit.blocks:
            for attn, names in ((blk.self_attn, "qkvo"), (blk.cross_attn, "qo")):
                for n in names:
                    swap(attn, n, _FP8Linear(getattr(attn, n), device, guard))
            ffn = blk.ffn
            if not (isinstance(ffn, nn.Sequential) and len(ffn) == 3 and isinstance(ffn[1], nn.GELU)
                    and ffn[1].approximate == "tanh"):
                raise RuntimeError("unexpected DiT FFN layout")
            swap(blk, "ffn", _FP8FFN(_FP8Linear(ffn[0], device, guard), _FP8Linear(ffn[2], device, guard), guard))
    except Exception:
        undo()
        raise
    return undo


def _unwrap(m):
    # enable_vram_management wraps norms in AutoWrappedModule
    return getattr(m, "module", m)


def make_fused_hooks(guard):
    """(FUSED_ROPE_FN, FUSED_ADALN) implementations for wan_video_dit."""
    cache = {"freqs": None}

    def rope(x, norm, freqs, num_heads):
        m = _unwrap(norm)
        # freqs is rebuilt once per chunk and shared by all 30 blocks. Holding a
        # reference keeps the identity check exact (a freed tensor's address could
        # be reused by the next chunk's freqs).
        if cache["freqs"] is not freqs:
            cos, sin = _freqs_cos_sin(freqs)
            cache.update(freqs=freqs, cos=cos, sin=sin)
        B, L, D = x.shape
        head_dim = D // num_heads
        x2 = x.contiguous().view(B * L, D)
        out = torch.empty_like(x2)
        _rms_rope_kernel[(B * L,)](
            x2, m.weight, cache["cos"], cache["sin"], out, L,
            D=D, HALF=head_dim // 2, HEAD_DIM=head_dim, eps=m.eps,
            ROPE=True, BLOCK=triton.next_power_of_2(D), num_warps=8, num_stages=1,
        )
        return out.view(B, L, D)

    def _check_mod(x, p):
        if p.dim() != 3 or p.shape[0] != x.shape[0] or p.shape[1] != 1 or p.shape[2] != x.shape[2]:
            raise RuntimeError(f"fused AdaLN expects [B, 1, D] modulation, got {tuple(p.shape)}")

    def ln_modulate(x, norm, shift, scale):
        _check_mod(x, shift)
        _check_mod(x, scale)
        # the kernel takes (scale, shift): reverse of modulate(x, shift, scale)
        return fused_ln_modulate(x.contiguous(), scale.contiguous(), shift.contiguous(), eps=_unwrap(norm).eps)

    def gate_add(x, gate, residual):
        _check_mod(x, gate)
        return fused_gate_add(x.contiguous(), gate.contiguous(), residual.contiguous())

    return guard(rope), (guard(ln_modulate), guard(gate_add))


def install_fused_dit(guard):
    wan_video_dit.FUSED_ROPE_FN, wan_video_dit.FUSED_ADALN = make_fused_hooks(guard)

    def undo():
        wan_video_dit.FUSED_ROPE_FN = None
        wan_video_dit.FUSED_ADALN = None
    return undo


@torch.no_grad()
def warmup_dit(dit, tokens):
    """Run block 0 once at a realistic token count: JIT-compiles the Triton
    kernels and exercises every installed DiT hook/module."""
    blk = dit.blocks[0]
    dev = blk.modulation.device
    D = blk.dim
    x = torch.zeros(1, tokens, D, dtype=torch.bfloat16, device=dev)
    for n in "qkvo":
        getattr(blk.self_attn, n)(x)
    blk.cross_attn.q(x)
    blk.cross_attn.o(x)
    blk.ffn(x)
    if wan_video_dit.FUSED_ROPE_FN is not None:
        half = D // blk.num_heads // 2
        freqs = torch.polar(torch.ones(tokens, 1, half, dtype=torch.float64, device=dev),
                            torch.zeros(tokens, 1, half, dtype=torch.float64, device=dev))
        wan_video_dit.FUSED_ROPE_FN(x, blk.self_attn.norm_q, freqs, blk.num_heads)
    if wan_video_dit.FUSED_ADALN is not None:
        p = torch.zeros(1, 1, D, dtype=torch.bfloat16, device=dev)
        wan_video_dit.FUSED_ADALN[0](x, blk.norm1, p, p)
        wan_video_dit.FUSED_ADALN[1](x, p, x)
    torch.cuda.synchronize(dev)

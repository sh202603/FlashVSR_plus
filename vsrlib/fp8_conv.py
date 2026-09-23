"""cuDNN graph-API FP8 convolutions for the TCDecoder and the LQ projector.

Installed by vsrlib.accel (FLASHVSR_FP8_CONV); never imported on the stock path.

Quantization: FP8 e4m3 with per-output-channel weight scales and static
power-of-two activation scales from a calibration table (fp8_conv_scales.py).
The e4m3 normal range is 2^-6..448, and several TCDecoder inputs sit mostly
below 2^-6 (amax ~0.3), so a fixed activation scale of 1.0 would push them into
the subnormals. Power-of-two scales keep the rescaling itself exact. Every conv
folds (weight scale x input scale / output scale), bias and ReLU into its
epilogue; ReLU commutes with the positive rescale.

TCDecoder: the stock decode runs one frame at a time through all 27 layers.
Here each layer processes a tile of latent frames at once (MemBlock's `past` is
"the same layer's input one frame earlier", so the result is the same), with the
cross-tile state kept in the decoder's own `mem` list so that `clean_mem()` —
which every pipeline calls at the start of `__call__` — resets it. Only conv
inputs are FP8: the residual stream through each stage's MemBlocks stays bf16
(quantizing it too cost ~3 dB of decoder PSNR), and the residual add happens in
the last conv's epilogue. The tail after the last MemBlock is frame-independent
and runs in sub-tiles to bound VRAM.
"""
import glob
import math
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

if sys.platform == "win32":
    # cudnn-frontend's cudart shim only probes Linux library names; point it at
    # the cudart DLL shipped with the torch wheel (already loaded by torch).
    _cudart = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_*.dll"))
    if _cudart:
        os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", os.path.basename(_cudart[0]))

import cudnn  # noqa: E402  (after the shim env var)

from .fp8_conv_scales import AMAX

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
# Activation scale headroom: calibrated amax maps to (448/32, 448/16], so inputs
# up to 16x hotter than anything seen in calibration still fit.
_HEADROOM = 16.0
_LATENT_TILE = 2            # latent frames per TCDecoder step (= a steady tiny-long chunk)
_TAIL_BYTES = 1 << 28       # cap for one full-resolution FP8 tensor in the decoder tail


def pow2_scale(amax):
    return 2.0 ** math.ceil(math.log2(max(amax, 1e-12) * _HEADROOM / FP8_MAX))


def _quantize(x, scale, memory_format):
    # torch's float->e4m3fn cast turns overflow into NaN, so saturate first.
    return (x / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8).contiguous(memory_format=memory_format)


def has_table(version):
    return str(version) in AMAX


class _Conv:
    """One FP8 conv + fused epilogue, with graphs built lazily per input shape.

    epilogue: acc * alpha[c] (+ bias[c]) (+ identity) (relu) -> FP8, or -> bf16
    for `final`. alpha folds the per-channel weight scale, the input scale and
    the output scale (1 for bf16 outputs); bias is pre-divided into output units.
    The identity (residual) input is a bf16 tensor and requires `final`.
    """

    def __init__(self, ctx, weight, bias, s_in, s_out, relu=False, identity=False,
                 final=False, padding=None, stride=None):
        nd = weight.dim() - 2
        assert final or not identity, "the residual stream is bf16: identity needs a bf16 output"
        self.ctx, self.nd = ctx, nd
        self.mf = torch.channels_last if nd == 2 else torch.channels_last_3d
        self.relu, self.identity, self.final = relu, identity, final
        self.padding = list(padding) if padding is not None else [weight.shape[-1] // 2] * nd
        self.stride = list(stride) if stride is not None else [1] * nd
        w = weight.detach().float()
        s_w = (w.abs().amax(dim=tuple(range(1, w.dim()))) / FP8_MAX).clamp_(min=1e-12)
        bshape = (1, w.shape[0]) + (1,) * nd
        self.w8 = (w / s_w.view(-1, *([1] * (w.dim() - 1)))).to(FP8).contiguous(memory_format=self.mf)
        self.alpha = (s_w * (s_in / s_out)).view(bshape).contiguous()
        self.bias = None if bias is None else (bias.detach().float() / s_out).view(bshape).contiguous()
        self.graphs = {}

    def _build(self, x):
        E4M3, FLOAT = cudnn.data_type.FP8_E4M3, cudnn.data_type.FLOAT
        g = cudnn.pygraph(io_data_type=E4M3, intermediate_data_type=FLOAT,
                          compute_data_type=FLOAT, handle=self.ctx.handle)
        X = g.tensor_like(x)
        W = g.tensor_like(self.w8)
        z = g.conv_fprop(image=X, weight=W, padding=self.padding, stride=self.stride,
                         dilation=[1] * self.nd)
        A = g.tensor_like(self.alpha)
        z = g.mul(a=z, b=A)
        io = {"X": X, "W": W, "A": A}
        if self.bias is not None:
            B = g.tensor_like(self.bias)
            z = g.bias(input=z, bias=B)
            io["B"] = B
        out_shape = self.out_shape(x.shape)
        if self.identity:
            ID = g.tensor_like(torch.empty(out_shape, dtype=torch.bfloat16, device=x.device)
                               .contiguous(memory_format=self.mf))
            z = g.add(a=z, b=ID)
            io["ID"] = ID
        if self.relu:
            z = g.relu(input=z)
        z.set_output(True).set_data_type(cudnn.data_type.BFLOAT16 if self.final else E4M3)
        g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        io["Y"] = z
        self.ctx.reserve_workspace(g.get_workspace_size())
        return g, io

    def out_shape(self, in_shape):
        n, _, *sp = in_shape
        k = self.w8.shape[2:]
        osp = [(s + 2 * p - kk) // st + 1 for s, p, kk, st in zip(sp, self.padding, k, self.stride)]
        return (n, self.w8.shape[0], *osp)

    def __call__(self, x, identity=None, out=None):
        key = tuple(x.shape)
        if key not in self.graphs:
            self.graphs[key] = self._build(x)
        g, io = self.graphs[key]
        if out is None:
            out = torch.empty(self.out_shape(x.shape), dtype=torch.bfloat16 if self.final else FP8,
                              device=x.device).contiguous(memory_format=self.mf)
        feed = {io["X"]: x, io["W"]: self.w8, io["A"]: self.alpha, io["Y"]: out}
        if "B" in io:
            feed[io["B"]] = self.bias
        if "ID" in io:
            feed[io["ID"]] = identity
        g.execute(feed, self.ctx.workspace, handle=self.ctx.handle)
        return out


class _Ctx:
    """cuDNN handle bound to torch's current stream, plus a shared workspace."""

    def __init__(self, device):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.handle = cudnn.create_handle()
            cudnn.set_stream(handle=self.handle,
                             stream=torch.cuda.current_stream(self.device).cuda_stream)
        self.workspace = torch.empty(8, dtype=torch.uint8, device=self.device)

    def reserve_workspace(self, size):
        if self.workspace.numel() < size:
            self.workspace = torch.empty(size, dtype=torch.uint8, device=self.device)

    def release(self):
        handle, self.handle = self.handle, None
        self.workspace = None
        if handle is not None:
            try:
                cudnn.destroy_handle(handle)
            except Exception:
                pass


def probe(device):
    """Build and run one small FP8 conv graph; raises if cuDNN can't provide it
    (cuDNN < 9.17, no FP8 kernels for this GPU, broken frontend install)."""
    ctx = _Ctx(device)
    try:
        conv = _Conv(ctx, torch.randn(16, 16, 3, 3, device=device), torch.zeros(16, device=device),
                     s_in=1.0, s_out=1.0, relu=True)
        x = torch.zeros(1, 16, 8, 8, dtype=FP8, device=device).contiguous(memory_format=torch.channels_last)
        conv(x)
        torch.cuda.synchronize(device)
    finally:
        ctx.release()


def _park(modules):
    """Swap each module's weight/bias for an empty placeholder while the FP8
    copies are in use, keeping the bf16 originals on the CPU for the undo. The
    pipeline moves these modules with .to(device) on every call, so parking the
    originals on the CPU alone would not keep the VRAM free — the placeholder does.
    Returns a restore callable."""
    saved = []
    for m in modules:
        for name in ("weight", "bias"):
            p = getattr(m, name, None)
            if isinstance(p, torch.nn.Parameter):
                saved.append((m, name, p.data.to("cpu"), p.requires_grad))
                setattr(m, name, torch.nn.Parameter(torch.empty(0, dtype=p.dtype, device=p.device),
                                                    requires_grad=False))

    def restore():
        for m, name, data, rg in saved:
            dev = getattr(m, name).device  # wherever the pipeline has moved the module since
            setattr(m, name, torch.nn.Parameter(data.to(dev), requires_grad=rg))
        saved.clear()
    return restore


# ----------------------------------------------------------------------------
# LQ projector
# ----------------------------------------------------------------------------

class FP8CausalConv3d:
    """Replacement forward for one CausalConv3d of the LQ projector.

    The causal-cache concat and the replicate padding stay in bf16 exactly as in
    CausalConv3d.forward; only the conv itself runs in FP8 (bf16 output), so
    both LQ projector variants (Buffer_* for -v 10, Causal_* for -v 11) keep
    their own cache handling untouched.
    """

    def __init__(self, ctx, conv3d, amax_in):
        self.m = conv3d
        self.s_in = pow2_scale(amax_in)
        self.conv = _Conv(ctx, conv3d.weight, conv3d.bias, s_in=self.s_in, s_out=1.0, final=True,
                          padding=[0, 0, 0], stride=list(conv3d.stride))

    def __call__(self, x, cache_x=None):
        m = self.m
        padding = list(m._padding)
        if cache_x is not None and m._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode='replicate')
        y = self.conv(_quantize(x, self.s_in, torch.channels_last_3d))
        return y.contiguous()


def lq_convs(lq_proj):
    """[(name, CausalConv3d)] of an LQ projector, unwrapping VRAM-management wrappers."""
    out = []
    for name in ("conv1", "conv2"):
        m = getattr(lq_proj, name)
        out.append((name, getattr(m, "module", m)))
    return out


def install_lq(lq_proj, version, device, guard):
    """Swap the LQ projector's conv forwards for FP8 ones (runtime errors go
    through `guard`). Returns (undo, object with .release())."""
    table = AMAX[str(version)]["lq"]
    ctx = _Ctx(device)
    convs = [m for _, m in lq_convs(lq_proj)]
    try:
        runners = [FP8CausalConv3d(ctx, m, table[name]) for name, m in lq_convs(lq_proj)]
    except Exception:
        ctx.release()
        raise
    for m, r in zip(convs, runners):
        m.forward = guard(r)
    restore = _park(convs)

    def undo():
        restore()
        for m in convs:
            if "forward" in m.__dict__:
                del m.forward
        ctx.release()
    return undo, ctx


# ----------------------------------------------------------------------------
# TCDecoder
# ----------------------------------------------------------------------------

def _i8(x):
    return x.view(torch.int8)


def _nhwc(x):
    """A channels-last (n, C, h, w) tensor as its underlying NHWC memory."""
    assert x.is_contiguous(memory_format=torch.channels_last)
    return x.permute(0, 2, 3, 1)


@triton.jit
def _quant_cat_kernel(x_ptr, cat_ptr, n_elem, frame_elems, C: tl.constexpr, inv_s, BLOCK: tl.constexpr):
    # x: NHWC bf16; cat: NHWC FP8 with 2C channels. Frame t's quantized values go
    # to cat[t, :C] (its input) and cat[t+1, C:] (the next frame's `past`).
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_s
    q = tl.minimum(tl.maximum(v, -448.0), 448.0).to(tl.float8e4nv)
    pix = offs // C
    c = offs % C
    tl.store(cat_ptr + pix * (2 * C) + c, q, mask=mask)
    tl.store(cat_ptr + (pix + frame_elems // C) * (2 * C) + C + c, q,
             mask=mask & (offs + frame_elems < n_elem))


@triton.jit
def _quant_up2_kernel(x_ptr, out_ptr, n_elem, H, W, C: tl.constexpr, inv_s, BLOCK: tl.constexpr):
    # x: NHWC bf16 -> out: NHWC FP8 at 2H x 2W (nearest)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_s
    q = tl.minimum(tl.maximum(v, -448.0), 448.0).to(tl.float8e4nv)
    c = offs % C
    pix = offs // C
    xx = pix % W
    yy = (pix // W) % H
    t = pix // (W * H)
    base = ((t * (2 * H) + 2 * yy) * (2 * W) + 2 * xx) * C + c
    row = 2 * W * C
    tl.store(out_ptr + base, q, mask=mask)
    tl.store(out_ptr + base + C, q, mask=mask)
    tl.store(out_ptr + base + row, q, mask=mask)
    tl.store(out_ptr + base + row + C, q, mask=mask)


_QBLOCK = 2048


def _quant_cat(x, scale, past):
    """cat([x, past], dim=1) quantized to FP8 for a MemBlock's first conv. The
    `past` of frame 0 is the given FP8 frame (zeros when None), of frame t the
    input of frame t-1 — the stock per-frame recurrence."""
    n, c, h, w = x.shape
    cat = torch.empty((n, 2 * c, h, w), dtype=FP8, device=x.device).contiguous(memory_format=torch.channels_last)
    ne = x.numel()
    _quant_cat_kernel[(triton.cdiv(ne, _QBLOCK),)](_nhwc(x), _nhwc(cat), ne, c * h * w, C=c,
                                                  inv_s=1.0 / scale, BLOCK=_QBLOCK)
    if past is None:
        _i8(cat)[:1, c:].zero_()
    else:
        _i8(cat)[:1, c:].copy_(_i8(past))
    return cat


def _quant_up2(x, scale):
    """Nearest 2x upsample of a bf16 channels-last tensor, quantized to FP8."""
    n, c, h, w = x.shape
    out = torch.empty((n, c, 2 * h, 2 * w), dtype=FP8, device=x.device).contiguous(memory_format=torch.channels_last)
    ne = x.numel()
    _quant_up2_kernel[(triton.cdiv(ne, _QBLOCK),)](_nhwc(x), _nhwc(out), ne, h, w, C=c,
                                                  inv_s=1.0 / scale, BLOCK=_QBLOCK)
    return out


def _tgrow_reorder(x, stride):
    """TGrow's (n, s*C, h, w) -> (n*s, C, h, w) channel-to-time split, channels-last."""
    if stride == 1:
        return x
    n, sc, h, w = x.shape
    c = sc // stride
    y = _i8(x).permute(0, 2, 3, 1).reshape(n, h, w, stride, c).permute(0, 3, 1, 2, 4).reshape(n * stride, h, w, c)
    return y.permute(0, 3, 1, 2).view(FP8)


class FP8TCDecoder:
    """FP8 execution of TAEHV.decoder (the FlashVSR TCDecoder), see module docstring."""

    # Stage layout of the [512, 256, 128, 128] decoder built by build_tcdecoder:
    # (MemBlock indices, index of the Upsample, TGrow, conv that follows)
    _STAGES = (((5, 6, 7), 8, 9, 10), ((11, 12, 13), 14, 15, 16), ((17, 18, 19), 20, 21, 22))

    def __init__(self, tcd, version, device):
        from src.models.TCDecoder import MemBlock, TGrow, Clamp, IdentityConv2d
        dec = tcd.decoder
        expect = [Clamp, torch.nn.Conv2d, torch.nn.ReLU, IdentityConv2d, torch.nn.ReLU,
                  MemBlock, MemBlock, MemBlock, torch.nn.Upsample, TGrow, torch.nn.Conv2d,
                  MemBlock, MemBlock, MemBlock, torch.nn.Upsample, TGrow, torch.nn.Conv2d,
                  MemBlock, MemBlock, MemBlock, torch.nn.Upsample, TGrow, torch.nn.Conv2d,
                  torch.nn.ReLU, IdentityConv2d, torch.nn.ReLU, torch.nn.Conv2d]
        if len(dec) != len(expect) or not all(isinstance(m, t) for m, t in zip(dec, expect)):
            raise RuntimeError("unexpected TCDecoder layout")
        for i in (8, 14, 20):
            if dec[i].mode != "nearest" or float(dec[i].scale_factor) != 2.0:
                raise RuntimeError("unexpected TCDecoder upsample")
        for mbs, *_ in self._STAGES:
            if not all(isinstance(dec[i].skip, torch.nn.Identity) for i in mbs):
                raise RuntimeError("unexpected MemBlock skip")
        a = AMAX[str(version)]["tcd"]
        s = {k: pow2_scale(v) for k, v in a.items()}
        # One input scale per stage for everything quantized from its bf16
        # residual stream (MemBlock first-conv inputs, the stage-end upsample).
        self.stage = stage = [max(s[f"decoder.{i}"] for i in mbs) for mbs, *_ in self._STAGES]
        self.ctx = ctx = _Ctx(device)
        self.tgrow_stride = {tg: dec[tg].stride for _, _, tg, _ in self._STAGES}

        def c(i, s_in, s_out=1.0, **kw):
            m = dec[i] if isinstance(i, int) else i
            return _Conv(ctx, m.weight, m.bias, s_in=s_in, s_out=s_out, **kw)

        self.s_entry = s["decoder.1"]
        L = {}
        L[1] = c(1, s["decoder.1"], s["decoder.3"], relu=True)
        L[3] = c(3, s["decoder.3"], relu=True, final=True)            # -> bf16 stage-0 stream
        for k, (mbs, _up, tg, nxt) in enumerate(self._STAGES):
            for i in mbs:
                cv = dec[i].conv
                L[(i, 0)] = c(cv[0], stage[k], s[f"decoder.{i}.conv.2"], relu=True)
                L[(i, 2)] = c(cv[2], s[f"decoder.{i}.conv.2"], s[f"decoder.{i}.conv.4"], relu=True)
                L[(i, 4)] = c(cv[4], s[f"decoder.{i}.conv.4"], relu=True, identity=True, final=True)
            L[tg] = c(dec[tg].conv, stage[k], s[f"decoder.{nxt}"])
            if k < 2:
                L[nxt] = c(nxt, s[f"decoder.{nxt}"], final=True)       # -> bf16 next-stage stream
        # tail: conv22 (+ReLU 23), IdentityConv 24 (+ReLU 25), final conv 26
        L[22] = c(22, s["decoder.22"], s["decoder.24"], relu=True)
        L[24] = c(24, s["decoder.24"], s["decoder.26"], relu=True)
        L[26] = c(26, s["decoder.26"], final=True)
        self.L = L

    def release(self):
        self.L = {}
        self.ctx.release()

    def _memblock(self, i, k, x, n_valid, mem):
        ch = x.shape[1]
        cat = _quant_cat(x, self.stage[k], mem[i])
        # State for the next tile: this layer's (quantized) input at the last
        # *valid* frame — frames past n_valid are padding.
        mem[i] = _i8(cat)[n_valid - 1:n_valid, :ch].contiguous(memory_format=torch.channels_last).view(FP8)
        h1 = self.L[(i, 0)](cat)
        h2 = self.L[(i, 2)](h1)
        return self.L[(i, 4)](h2, identity=x)

    def _tail(self, x, out):
        """Stage-3 upsample, TGrow and the frame-independent convs, in sub-tiles."""
        n, ch, h, w = x.shape
        stride = self.tgrow_stride[21]
        per_frame = self.L[21].w8.shape[0] * 4 * h * w  # TGrow output bytes per input frame
        j = 1
        while j * 2 <= n and n % (j * 2) == 0 and (j * 2) * per_frame <= _TAIL_BYTES:
            j *= 2
        for a in range(0, n, j):
            y = _tgrow_reorder(self.L[21](_quant_up2(x[a:a + j], self.stage[2])), stride)
            y = self.L[22](y.contiguous(memory_format=torch.channels_last))
            y = self.L[24](y)
            self.L[26](y, out=out[a * stride:(a + j) * stride])

    def _run_tile(self, xt, n_valid, mem, out):
        # decoder.0 Clamp in bf16 exactly as the stock layer, then quantize once.
        x = _quantize(torch.tanh(xt / 3) * 3, self.s_entry, torch.channels_last)
        x = self.L[1](x)
        x = self.L[3](x)
        nv = n_valid
        for k, (mbs, _up, tg, nxt) in enumerate(self._STAGES):
            for i in mbs:
                x = self._memblock(i, k, x, nv, mem)
            if k == 2:
                self._tail(x, out)
                return
            stride = self.tgrow_stride[tg]
            x = _tgrow_reorder(self.L[tg](_quant_up2(x, self.stage[k])), stride)
            x = self.L[nxt](x.contiguous(memory_format=torch.channels_last))
            nv *= stride

    def decode(self, x, mem):
        """x: (1, T, C, h, w) latents (+cond channels); returns (1, 4T, 3, 8h, 8w) bf16."""
        if x.shape[0] != 1:
            raise RuntimeError("FP8 TCDecoder supports batch size 1 only")
        T = x.shape[1]
        t_up = 1
        for tg in self.tgrow_stride.values():
            t_up *= tg
        H, W = x.shape[3] * 8, x.shape[4] * 8
        out = torch.empty((T * t_up, 3, H, W), dtype=torch.bfloat16, device=x.device)
        k = _LATENT_TILE
        for a in range(0, T, k):
            xt = x[0, a:a + k]
            n_valid = xt.shape[0]
            if n_valid < k:
                # Pad with copies of the last frame: later frames never affect
                # earlier ones, and mem is taken from the last valid frame.
                xt = torch.cat([xt, xt[-1:].expand(k - n_valid, *xt.shape[1:])], dim=0)
            ot = torch.empty((k * t_up, 3, H, W), dtype=torch.bfloat16, device=x.device).contiguous(memory_format=torch.channels_last)
            self._run_tile(xt, n_valid, mem, ot)
            out[a * t_up:(a + n_valid) * t_up].copy_(ot[:n_valid * t_up])
        return out.unsqueeze(0)


def install_tcd(tcd, version, device, guard):
    """Route TAEHV.decode_video through FP8TCDecoder (runtime errors go through
    `guard`). Returns (undo, runner)."""
    runner = FP8TCDecoder(tcd, version, device)

    def decode_video(x, parallel=True, show_progress_bar=False, cond=None):
        trim_flag = tcd.mem[-8] is None  # same check as the stock method
        if cond is not None:
            x = torch.cat([tcd.pixel_shuffle(cond), x], dim=2)
        x = runner.decode(x, tcd.mem)
        if trim_flag:
            return x[:, tcd.frames_to_trim:]
        return x

    tcd.decode_video = guard(decode_video)
    restore = _park([m for m in tcd.decoder.modules() if isinstance(m, torch.nn.Conv2d)])

    def undo():
        restore()
        if "decode_video" in tcd.__dict__:
            del tcd.decode_video
        tcd.clean_mem()  # drop FP8 state the stock path can't read
        runner.release()
    return undo, runner

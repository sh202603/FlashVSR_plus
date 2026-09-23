"""Triton/FP8 kernels vendored from flashvsr-sm89-ops.

Source: https://github.com/aireet/flashvsr-sm89-ops at commit 6dab0b6
(flashvsr_sm89_ops/{fp8_quant,fp8_linear,fp8_ffn,fused_adaln,fused_rms_rope}.py),
licensed under the Apache License 2.0 (see LICENSE in this directory).

Modifications: absolute `flashvsr_sm89_ops.*` imports changed to relative ones.
The upstream package __init__, the TCDecoder channels_last/compile helpers and the
LCSA block-sparse attention are not included. Wiring into FlashVSR_plus lives in
vsrlib/accel_dit.py.
"""

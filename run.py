#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

parser = argparse.ArgumentParser(description="FlashVSR+: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.")
parser.add_argument("-i", "--input", type=str, help="Path to video file or folder of images")
parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor, default=4")
parser.add_argument("-v", "--version", type=str, default="10", choices=["10", "11"], help="Model version, default=10")
parser.add_argument("-m", "--mode", type=str, default="tiny", choices=["tiny", "tiny-long", "full"], help="The type of pipeline to use, default=tiny")
parser.add_argument("--tiled-vae", action="store_true", help="Enable tile decoding")
parser.add_argument("--tiled-dit", action="store_true", help="Enable tile inference")
parser.add_argument("--tile-size", type=int, default=256, help="Chunk size of tile inference, default=256")
parser.add_argument("--overlap", type=int, default=24, help="Overlap size of tile inference, default=24")
parser.add_argument("--unload-dit", action="store_true", help="Unload DiT before decoding")
parser.add_argument("--color-fix", action="store_true", help="Correct output video color")
parser.add_argument("--seed", type=int, default=0, help="Random Seed, default=0")
parser.add_argument("-t", "--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type for processing, default=bf16")
parser.add_argument("-d", "--device", type=str, default="auto", help="Device to run FlashVSR")
parser.add_argument("-f", "--fps", type=int, default=30, help="Output FPS (for image sequences only), default=30")
parser.add_argument("-q", "--quality", type=int, default=6, help="Output video quality, default=6")
parser.add_argument("-a", "--attention", default="sage", choices=["sage", "block"], help="Attention mode, default=sage")
parser.add_argument("--output-height", type=int, default=None, help="Downscale the final stitched video to this height (tiled tiny-long only), default=None (native)")
parser.add_argument("--temp-quality", type=int, default=8, choices=range(1, 11), metavar="[1-10]", help="Quality of per-tile temp videos in tiled tiny-long mode, default=8")
parser.add_argument("--kv-ratio", type=int, default=3, help="KV cache length of sparse attention; lower saves VRAM at some quality cost, default=3")
parser.add_argument("--pad-align", action="store_true", help="Pad the upscaled frame to the next multiple of 128 instead of center-cropping, then crop the output back to exactly scale*input size — preserves frame edges on non-tiled runs (tiled-DiT already preserves them)")
parser.add_argument("--resume", action="store_true", help="Resume an interrupted tiled tiny-long run: keep _temp at startup and reuse completed tile videos from a previous run with identical parameters")
parser.add_argument("--accel", action="store_true", help="Speed up inference with FP8 convolutions/linears and fused DiT kernels (RTX 40 series or newer, bf16; falls back to the standard path automatically). Same as FLASHVSR_ACCEL=1; the output differs slightly from a run without it")
parser.add_argument("output_folder", type=str, help="Path to save output video")
args = parser.parse_args()

import os

# vsrlib.common is torch-free and, as a side effect of this import, sets the
# CUDA allocator env var (unless user-provided) before the first torch import
# below — the same ordering run.py guaranteed when it owned that block.
from vsrlib.common import log, root, temp

from tqdm import tqdm
log("[FlashVSR] Preparing dependencies...", message_type="finish")

# Everything from here down to cli_entry() is re-exported on purpose: external
# pipelines (lada-ex / jasna workers) import this module with a dummy sys.argv
# and reuse its functions and globals as run.<name>. Removing any of these
# names (even seemingly unused imports) breaks that contract.
import re
import math
import uuid
import json
import hashlib
import datetime
import itertools
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from huggingface_hub import snapshot_download
from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj, Causal_LQ4x_Proj

from vsrlib.media_io import (
    model_downlod, is_ffmpeg_available, natural_key, list_images_natural,
    is_video, save_video, merge_video_with_audio, prepare_tensors, probe_input,
    count_video_frames,
)
from vsrlib.preprocess import (
    tensor2video, largest_8n1_leq, next_8n5, compute_scaled_and_target_dims,
    compute_scaled_and_padded_dims, tensor_upscale_then_center_crop,
    get_input_params_from_dims, stream_tile_frames, get_input_params,
    input_tensor_generator, prepare_input_tensor,
)
from vsrlib.tiling import (
    calculate_tile_coords, create_feather_mask_numpy, create_feather_mask,
    stitch_video_tiles,
)
from vsrlib.resume import (
    _input_fingerprint, build_resume_manifest, resume_manifest_hash,
    prepare_resume_dir, scan_completed_tiles,
)
from vsrlib.engine import devices, init_pipeline, main

def cli_entry():
    """Console entry point (flashvsr-cli). Uses the module-level `args`, which
    argparse populated from sys.argv at import time — the same flow as
    `python run.py ...`, so both invocations behave identically."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[args.dtype]
    except:
        dtype = torch.bfloat16

    if args.accel:
        # vsrlib.accel reads the environment (so external workers can opt in too)
        os.environ["FLASHVSR_ACCEL"] = "1"

    if args.attention == "sage":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True

    if os.path.exists(temp):
        # --resume must never destroy rescuable tiles, even when the rest of the
        # command turns out not to match (the manifest check handles that later).
        if args.resume:
            log("[FlashVSR] --resume: keeping previous _temp contents.", message_type='info')
        else:
            shutil.rmtree(temp)
    os.makedirs(temp, exist_ok=True)
    name = os.path.basename(args.input.rstrip('/'))
    # Create the output folder up front: tiny-long writes to it via imageio directly
    # (no save_video), and a missing folder must not surface only after hours of tiles.
    os.makedirs(args.output_folder, exist_ok=True)
    final = os.path.join(args.output_folder, f"FlashVSR_{args.mode}_{name.split('.')[0]}_{args.seed}.mp4")
    result, fps = main(args.input, args.version, args.mode, args.scale, args.color_fix, args.tiled_vae, args.tiled_dit,args.tile_size,
        args.overlap, args.unload_dit, dtype, kv_ratio=args.kv_ratio, seed=args.seed, device=args.device, quality=args.quality, output=final,
        output_height=args.output_height, temp_quality=args.temp_quality, pad_align=args.pad_align, resume=args.resume)
    if args.mode != "tiny-long":
        # Cropped-back dims are arbitrary even numbers; macro_block_size=2 stops
        # imageio's default 16px auto-resize (same precedent as stitch_video_tiles).
        save_video(result, final, fps=fps, quality=args.quality, macro_block_size=2 if args.pad_align else None)

    merge_video_with_audio(final, args.input)
    log("[FlashVSR] Done.", message_type='finish')

if __name__ == "__main__":
    cli_entry()

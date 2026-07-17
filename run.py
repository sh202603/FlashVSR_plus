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
parser.add_argument("output_folder", type=str, help="Path to save output video")
args = parser.parse_args()

# Reduce CUDA allocator fragmentation for long tiled runs. Must be set before the
# first CUDA allocation, and never overrides a user-provided allocator config.
import os
if sys.platform.startswith("linux") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

from tqdm import tqdm
log("[FlashVSR] Preparing dependencies...", message_type="finish")

import os
import re
import math
import uuid
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

root = os.path.dirname(os.path.abspath(__file__))
temp = os.path.join(root, "_temp")
devices = get_device_list()

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(root, "models", model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def is_ffmpeg_available():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        log("[FlashVSR] FFmpeg not found!", message_type="warning")
        log("Please install FFmpeg and ensure it is in your system's PATH.")
        log("- Windows: Download from https://www.ffmpeg.org/download.html and add the 'bin' directory to PATH.")
        log("- macOS (via Homebrew): brew install ffmpeg")
        log("- Linux (Ubuntu/Debian): sudo apt-get install ffmpeg")
        return False
    return True

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def save_video(frames, save_path, fps=30, quality=5, macro_block_size=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    # macro_block_size is only forwarded when explicitly requested so the default
    # writer call stays identical for existing callers.
    writer_kwargs = {} if macro_block_size is None else {"macro_block_size": macro_block_size}
    w = imageio.get_writer(save_path, fps=fps, quality=quality, **writer_kwargs)
    for frame_np in tqdm(frames_np, desc=f"[FlashVSR] Saving video"):
        w.append_data(frame_np)
    w.close()

def merge_video_with_audio(video_path, audio_source_path):
    temp = video_path+"temp.mp4"
    
    if os.path.isdir(audio_source_path):
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
        return
    
    if not is_ffmpeg_available():
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
        return
    
    try:
        probe = ffmpeg.probe(audio_source_path)
        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
        if not audio_streams:
            log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
            return
        log("[FlashVSR] Copying audio tracks...")
        os.rename(video_path, temp)
        input_video = ffmpeg.input(temp)['v']
        input_audio = ffmpeg.input(audio_source_path)['a']
        output_ffmpeg = ffmpeg.output(
            input_video, input_audio, video_path,
            vcodec='copy', acodec='copy'
        ).run(overwrite_output=True, quiet=True)
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
    except ffmpeg.Error as e:
        print("[ERROR] FFmpeg error during merge:", e.stderr.decode() if e.stderr else "Unknown error")
        log(f"[FlashVSR] Audio merge failed. A silent video has been saved to '{video_path}'.", message_type='warning')
        
    finally:
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError as e:
                log(f"[FlashVSR] Could not remove temporary file '{temp}': {e}", message_type='error')
    
def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")
        
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def compute_scaled_and_padded_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    """Ceil-to-multiple counterpart of compute_scaled_and_target_dims (--pad-align).
    Returns (sW, sH, tW, tH, pad_left, pad_top); tW >= sW and tH >= sH always,
    so the frame is padded up instead of center-cropped and no edge is lost."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple
    return sW, sH, tW, tH, (tW - sW) // 2, (tH - sH) // 2

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int, pad_left=None, pad_top=None) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW

    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)

    if pad_left is not None or pad_top is not None:
        # pad-align path: pad up to (tH, tW) instead of cropping, preserving edges.
        pl = pad_left or 0
        pt = pad_top or 0
        pr = max(0, tW - sW - pl)
        pb = max(0, tH - sH - pt)
        if pl or pr or pt or pb:
            # reflect needs pad < dim; fall back to replicate for tiny inputs.
            mode = 'reflect' if (max(pl, pr) < sW and max(pt, pb) < sH) else 'replicate'
            upscaled_tensor = F.pad(upscaled_tensor, (pl, pr, pt, pb), mode=mode)
        return upscaled_tensor.squeeze(0)

    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
            
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        
        frames = []
        for p in paths0:
            with Image.open(p).convert('RGB') as img:
                img_np = np.array(img).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img_np).to(dtype))
                
        vid = torch.stack(frames, 0)
        fps = 30
        return vid, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        meta = {}
        try:
            meta = rdr.get_meta_data()
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
        except Exception:
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
            
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        
        total = meta.get('nframes', rdr.count_frames())
        if total is None or total <= 0 :
             total = len([_ for _ in rdr])
             rdr = imageio.get_reader(path)
            
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        frames = []
        try:
            for frame_data in rdr:
                frame_np = frame_data.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame_np).to(dtype))
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        vid = torch.stack(frames, 0)
        return vid, fps
    
    raise ValueError(f"Unsupported input: {path}")

def probe_input(path):
    """Read only the metadata of a video/image-folder input: no frame data is kept.
    Returns (kind, source, frame_count, height, width, fps); fps is None for folders."""
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as img0:
            w0, h0 = img0.size
        return "images", paths0, len(paths0), h0, w0, None

    if is_video(path):
        rdr = imageio.get_reader(path)
        try:
            try:
                meta = rdr.get_meta_data()
            except Exception:
                meta = {}
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape

            fps_val = meta.get('fps', 30)
            fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

            total = meta.get('nframes', None)
            if not isinstance(total, (int, np.integer)) or total <= 0:
                total = rdr.count_frames()
            if total is None or total <= 0 or (isinstance(total, float) and not math.isfinite(total)):
                total = sum(1 for _ in rdr)
        finally:
            rdr.close()
        if total <= 0:
            raise RuntimeError(f"Cannot read frames from {path}")
        return "video", path, int(total), h0, w0, fps

    raise ValueError(f"Unsupported input: {path}")

def get_input_params_from_dims(N0, h0, w0, scale, pad_align=False):
    """Same math as padding to 8n+5 in main() followed by get_input_params(), but
    computed from metadata alone so no frames need to be resident."""
    multiple = 128
    if pad_align:
        sW, sH, tW, tH, _, _ = compute_scaled_and_padded_dims(w0, h0, scale=scale, multiple=multiple)
    else:
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    F = largest_8n1_leq(next_8n5(N0) + 4)
    if F == 0:
        raise RuntimeError(f"Not enough frames. Got {N0}.")
    return tH, tW, F

def stream_tile_frames(kind, source, x1, y1, x2, y2, N0, F, device, scale=4, tW=None, tH=None, dtype=torch.bfloat16, pad_align=False):
    """Streaming counterpart of input_tensor_generator for tiny-long: reads frames
    from disk one at a time, crops the tile region, upscales, and yields F CPU
    tensors of shape (C, tH, tW) in [-1, 1]. Indices >= N0 repeat the last frame,
    which matches padding the input to 8n+5 with its final frame."""
    pad_left = pad_top = None
    if pad_align:
        # Centered padding offsets up to the ceil-aligned (tH, tW).
        pad_left = max(0, (tW - (x2 - x1) * scale) // 2)
        pad_top = max(0, (tH - (y2 - y1) * scale) // 2)
    reader = imageio.get_reader(source) if kind == "video" else None
    frame_iter = iter(reader) if reader is not None else None
    last_out = None
    try:
        for i in range(F):
            if i < N0:
                frame_np = None
                if kind == "video":
                    try:
                        frame_np = np.asarray(next(frame_iter))
                    except StopIteration:
                        log(f"[FlashVSR] Input ended early at frame {i}/{N0}; repeating the last frame.", message_type='warning')
                        N0 = i
                else:
                    with Image.open(source[i]) as img:
                        frame_np = np.array(img.convert('RGB'))
                if frame_np is not None:
                    crop_np = frame_np[y1:y2, x1:x2, :3].astype(np.float32) / 255.0
                    frame_t = torch.from_numpy(crop_np).to(dtype).to(device)
                    tensor_chw = tensor_upscale_then_center_crop(frame_t, scale=scale, tW=tW, tH=tH, pad_left=pad_left, pad_top=pad_top)
                    last_out = (tensor_chw * 2.0 - 1.0).to('cpu').to(dtype)
                    del frame_t, tensor_chw
            if last_out is None:
                raise RuntimeError(f"Cannot read any frames from {source}")
            yield last_out
    finally:
        if reader is not None:
            reader.close()

def get_input_params(image_tensor, scale, pad_align=False):
    N0, h0, w0, _ = image_tensor.shape

    multiple = 128
    if pad_align:
        sW, sH, tW, tH, _, _ = compute_scaled_and_padded_dims(w0, h0, scale=scale, multiple=multiple)
    else:
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)

    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")

    return tH, tW, F

def input_tensor_generator(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16, pad_align=False):
    """
    一个生成器函数，逐帧处理并 yield 准备好的帧张量，以节省内存。
    产出的每个张量形状为 (C, H, W)。
    """
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, F = get_input_params(image_tensor, scale, pad_align=pad_align)
    pad_left = pad_top = None
    if pad_align:
        _, _, _, _, pad_left, pad_top = compute_scaled_and_padded_dims(w0, h0, scale=scale)

    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH, pad_left=pad_left, pad_top=pad_top)
        tensor_out = tensor_chw * 2.0 - 1.0
        del tensor_chw
        yield tensor_out.to('cpu').to(dtype)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16, pad_align=False):
    N0, h0, w0, _ = image_tensor.shape

    multiple = 128
    pad_left = pad_top = None
    if pad_align:
        sW, sH, tW, tH, pad_left, pad_top = compute_scaled_and_padded_dims(w0, h0, scale=scale, multiple=multiple)
    else:
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)

    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")

    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH, pad_left=pad_left, pad_top=pad_top)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)
        
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask_numpy(size, overlap):
    H, W = size
    mask = np.ones((H, W, 1), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    
    mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
    mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
    
    mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
    mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
    
    return mask

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def stitch_video_tiles(
    tile_paths,
    tile_coords,
    final_dims,
    scale,
    overlap,
    output_path,
    fps,
    quality,
    cleanup=True,
    chunk_size=40,
    frame_count=None,
    output_height=None,
    ram_budget_gb=8.0,
):
    if not tile_paths:
        log("No tile videos found to stitch.", message_type='error')
        return

    final_W, final_H = final_dims

    readers = [imageio.get_reader(p) for p in tile_paths]

    try:
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0 or (isinstance(num_frames, float) and not math.isfinite(num_frames)):
            probe_rdr = imageio.get_reader(tile_paths[0])
            num_frames = sum(1 for _ in probe_rdr)
            probe_rdr.close()
        num_frames = int(num_frames)
        # Tile videos contain 8n+5 padding frames beyond the source length; stop at
        # frame_count so the padding never reaches the final video.
        total_frames = num_frames if frame_count is None else min(num_frames, frame_count)

        # The feather masks are time-invariant, so the weight canvas is a single 2-D
        # plane computed once, not a per-chunk 4-D canvas (which cost chunk_size×
        # final_H×final_W×3 float32 — ~16GB at 8K — all over again every chunk).
        masks = []
        weight_canvas = np.zeros((final_H, final_W, 1), dtype=np.float32)
        for (x1_orig, y1_orig, x2_orig, y2_orig) in tile_coords:
            tile_H, tile_W = (y2_orig - y1_orig) * scale, (x2_orig - x1_orig) * scale
            mask = create_feather_mask_numpy((tile_H, tile_W), overlap * scale)
            masks.append(mask)
            out_y1, out_x1 = y1_orig * scale, x1_orig * scale
            weight_canvas[out_y1:out_y1 + tile_H, out_x1:out_x1 + tile_W, :] += mask
        weight_canvas[weight_canvas == 0] = 1.0

        bytes_per_frame = final_H * final_W * 3 * 4
        budget_frames = max(1, int(ram_budget_gb * (1024 ** 3)) // bytes_per_frame)
        chunk_size = max(1, min(chunk_size, budget_frames))

        out_size = None
        if output_height is not None:
            if output_height < final_H:
                out_h = output_height - (output_height % 2)
                out_w = int(round(final_W * out_h / final_H / 2)) * 2
                out_size = (out_h, out_w)
                log(f"[FlashVSR] Stitching at {final_W}x{final_H}, downscaling output to {out_w}x{out_h}", message_type='info')
            else:
                log(f"[FlashVSR] --output-height {output_height} >= native height {final_H}, keeping native resolution.", message_type='warning')

        # One persistent iterator per tile: imageio's iter_data() restarts from
        # frame 0 on every call, so re-creating it per chunk re-decodes the whole
        # video each time (O(n²) — hours on multi-minute inputs).
        iters = [reader.iter_data() for reader in readers]

        with imageio.get_writer(output_path, fps=fps, quality=quality, macro_block_size=2) as writer:
            for start_frame in tqdm(range(0, total_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                current_chunk_size = min(chunk_size, total_frames - start_frame)

                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)

                for i, frame_iter in enumerate(iters):
                    tile_chunk_frames = list(itertools.islice(frame_iter, current_chunk_size))
                    if len(tile_chunk_frames) != current_chunk_size:
                        raise RuntimeError(
                            f"Tile video {i+1} ended early ({start_frame + len(tile_chunk_frames)}/{total_frames} frames) — "
                            f"the tiled run is incomplete, refusing to write a broken output."
                        )
                    tile_chunk_np = np.stack(tile_chunk_frames, axis=0).astype(np.float32) / 255.0

                    tile_H, tile_W = tile_chunk_np.shape[1:3]
                    if (tile_H, tile_W) != masks[i].shape[:2]:
                        raise RuntimeError(
                            f"Tile video {i+1} is {tile_W}x{tile_H}, expected {masks[i].shape[1]}x{masks[i].shape[0]} — "
                            f"tile_size/scale mismatch."
                        )

                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    chunk_canvas[:, out_y1:out_y1 + tile_H, out_x1:out_x1 + tile_W, :] += tile_chunk_np * masks[i][np.newaxis]

                chunk_canvas /= weight_canvas[np.newaxis]

                for frame_idx_in_chunk in range(current_chunk_size):
                    frame = chunk_canvas[frame_idx_in_chunk]
                    if out_size is not None:
                        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
                        frame_t = F.interpolate(frame_t, size=out_size, mode="area")
                        frame = frame_t.squeeze(0).permute(1, 2, 0).numpy()
                    frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)

    finally:
        log("Closing all tile reader instances...")
        for reader in readers:
            try:
                reader.close()
            except Exception:
                pass

    if cleanup:
        log("Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                log(f"Could not remove temporary file '{path}': {e}", message_type='warning')

def init_pipeline(version, mode, device, dtype):
    if version == "10":
        model = "FlashVSR"
    else:
        model = "FlashVSR-v1.1"
    model_downlod(model_name="JunhaoZhuang/" + model)
    model_path = os.path.join(root, "models", model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist! Please save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist! Please save it to "{model_path}"')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist! Please save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist! Please save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist! Please save it to "{model_path}"')
    prompt_path = os.path.join(root, "models", "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
        
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe

def main(input, version, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, dtype, sparse_ratio=2, kv_ratio=3, local_range=11, seed=0, device="auto", quality=6, output=None, output_height=None, temp_quality=8, pad_align=False):
    _device = device
    if device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
    if _device == "auto" or _device not in devices:
        raise RuntimeError("No devices found to run FlashVSR!")
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
        
    if tiled_dit and (tile_overlap > tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    if tiled_dit and tile_overlap <= 0:
        raise ValueError('The "tile_overlap" must be positive!')
    if tiled_dit and (tile_size * scale) % 128 != 0:
        raise ValueError(f'"tile_size" x scale must be a multiple of 128, otherwise tiles get center-cropped and stitching misaligns (for scale {scale} use e.g. {", ".join(str(t) for t in (128, 160, 192, 224, 256) if (t * scale) % 128 == 0)}).')

    if mode == "tiny-long":
        # tiny-long streams frames from disk per tile instead of preloading the whole
        # video: a 10-minute 1080p input would need >200GB of RAM as bf16 tensors.
        kind, source, N0, h0, w0, src_fps = probe_input(input)
        _fps = src_fps if kind == "video" else args.fps
        frame_count = N0
        frames = None
        log("[FlashVSR] Preparing frames...", message_type="finish")
    else:
        _frames, fps = prepare_tensors(input, dtype=dtype)
        _fps = fps if is_video(input) else args.fps

        add = next_8n5(_frames.shape[0]) - _frames.shape[0]
        padding_frames = _frames[-1:, :, :, :].repeat(add, 1, 1, 1)
        frames = torch.cat([_frames, padding_frames], dim=0)
        frame_count = _frames.shape[0]
        N0, h0, w0 = frame_count, frames.shape[1], frames.shape[2]
        del _frames
        clean_vram()

        log("[FlashVSR] Preparing frames...", message_type="finish")

    if tiled_dit and tile_size > min(h0, w0):
        raise ValueError(f'"tile_size" ({tile_size}) must not exceed the smaller input dimension ({min(h0, w0)}); use a smaller tile or disable --tiled-dit.')

    if pad_align and tiled_dit:
        log("[FlashVSR] --pad-align has no effect on the tiled-DiT path (tiles already preserve the full frame).", message_type='info')

    if tiled_dit:
        if mode == "tiny-long":
            H, W = h0, w0
            local_temp = os.path.join(temp, str(uuid.uuid4()))
            os.makedirs(local_temp, exist_ok=True)
        else:
            N, H, W, C = frames.shape
            num_aligned_frames = largest_8n1_leq(N + 4) - 4
            final_output_canvas = torch.zeros(
                (num_aligned_frames, H * scale, W * scale, C),
                dtype=dtype,
                device="cpu"
            )
            weight_sum_canvas = torch.zeros_like(final_output_canvas)

        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []
        temp_videos = []

        if _device.startswith("cuda"):
            free_b, total_b = torch.cuda.mem_get_info()
            # Linear fit of measured tiny-long footprints (RTX 5080, Linux,
            # expandable_segments): ~8.7 GiB reserved at 768^2, ~12.3 GiB at 1024^2
            # (weights + area-proportional KV cache + transients), plus ~1 GiB CUDA
            # context that lives outside the torch allocator.
            est_gb = 5.1 + 8.2 * ((tile_size * scale) ** 2) / float(1024 ** 2)
            log(f"[FlashVSR] {len(tile_coords)} tiles; estimated peak VRAM ~{est_gb:.1f} GiB per tile (free now: {free_b/2**30:.1f} GiB)", message_type='info')
            if est_gb * (2 ** 30) > free_b * 0.95:
                log('[FlashVSR] Estimated peak is close to or above free VRAM — consider a smaller --tile-size (192 needs ~11 GiB).', message_type='warning')
        if mode == "tiny-long" and _fps:
            est_disk_gb = len(tile_coords) * (frame_count / max(_fps, 1)) * 0.6 * ((tile_size * scale) / 768.0) ** 2 / 1024.0
            free_disk_gb = shutil.disk_usage(temp).free / 2 ** 30
            log(f"[FlashVSR] Temp tile videos: ~{est_disk_gb:.1f} GiB estimated, all kept until stitching finishes (free disk: {free_disk_gb:.1f} GiB)", message_type='info')
            if est_disk_gb > free_disk_gb:
                log("[FlashVSR] Free disk space may be insufficient for temp tile videos!", message_type='warning')

        pipe = init_pipeline(version, mode, _device, dtype)
        if _device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            temp_name = None
            input_tile = None
            if mode == "tiny-long":
                temp_name = os.path.join(local_temp, f"{i+1:05d}.mp4")
                th, tw, F = get_input_params_from_dims(N0, y2 - y1, x2 - x1, scale)
                LQ_tile = stream_tile_frames(kind, source, x1, y1, x2, y2, N0, F, _device, scale=scale, tW=tw, tH=th, dtype=dtype)
            else:
                input_tile = frames[:, y1:y2, x1:x2, :]
                LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                LQ_tile = LQ_tile.to(_device)

            if i == 0:
                log(f"[FlashVSR] Processing {frame_count} frames...", message_type='info')
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: ({x1},{y1}) to ({x2},{y2})", message_type='info')

            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, fps=_fps, quality=temp_quality, output_path=temp_name, tiled_dit=True
            )

            if _device.startswith("cuda"):
                log(f"[FlashVSR] Tile {i+1}/{len(tile_coords)} peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB allocated, {torch.cuda.max_memory_reserved()/2**30:.2f} GiB reserved", message_type='info')
                torch.cuda.reset_peak_memory_stats()

            temp_videos.append(temp_name)
            if mode == "tiny-long":
                if output_tile_gpu is not True:
                    raise RuntimeError(f"[FlashVSR] Tile {i+1}/{len(tile_coords)} failed — see the error above.")
                del LQ_tile, input_tile
                clean_vram()
                continue
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
        
        if mode == "tiny-long":
            stitch_video_tiles(tile_paths=temp_videos, tile_coords=tile_coords, final_dims=(W * scale, H * scale),
                scale=scale, overlap=tile_overlap, output_path=output, fps=_fps, quality=quality, cleanup=True,
                frame_count=frame_count, output_height=output_height
            )
            shutil.rmtree(local_temp)
            del pipe
            clean_vram()
            return None, _fps
        else:
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output = final_output_canvas / weight_sum_canvas
    else:
        pad_crop_rect = None
        if pad_align:
            sW, sH, ptW, ptH, pl, pt = compute_scaled_and_padded_dims(w0, h0, scale=scale)
            pad_crop_rect = (pt, pl, sH, sW)
            log(f"[FlashVSR] --pad-align: processing at {ptW}x{ptH} (padded), output cropped back to {sW}x{sH}.", message_type='info')
        if mode == "tiny-long":
            th, tw, F = get_input_params_from_dims(N0, h0, w0, scale, pad_align=pad_align)
            LQ = stream_tile_frames(kind, source, 0, 0, w0, h0, N0, F, _device, scale=scale, tW=tw, tH=th, dtype=dtype, pad_align=pad_align)
        else:
            LQ, th, tw, F = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype, pad_align=pad_align)
            LQ = LQ.to(_device)

        pipe = init_pipeline(version, mode, _device, dtype)
        log(f"[FlashVSR] Processing {frame_count} frames...", message_type='info')
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit, fps=_fps, quality=quality, output_path=output, tiled_dit=True,
            crop_rect=pad_crop_rect if mode == "tiny-long" else None
        )

        if mode == "tiny-long":
            if video is not True:
                raise RuntimeError("[FlashVSR] Pipeline failed — see the error above.")
            del pipe, LQ
            clean_vram()
            return video, _fps

        log("[FlashVSR] Preparing frames...")
        final_output = tensor2video(video).to("cpu")
        if pad_crop_rect is not None:
            _t, _l, _oh, _ow = pad_crop_rect
            final_output = final_output[:, _t:_t + _oh, _l:_l + _ow, :]
        del pipe, video, LQ
        clean_vram()

    return final_output[:frame_count, :, :, :], fps

if __name__ == "__main__":
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[args.dtype]
    except:
        dtype = torch.bfloat16

    if args.attention == "sage":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True

    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp, exist_ok=True)
    name = os.path.basename(args.input.rstrip('/'))
    # Create the output folder up front: tiny-long writes to it via imageio directly
    # (no save_video), and a missing folder must not surface only after hours of tiles.
    os.makedirs(args.output_folder, exist_ok=True)
    final = os.path.join(args.output_folder, f"FlashVSR_{args.mode}_{name.split('.')[0]}_{args.seed}.mp4")
    result, fps = main(args.input, args.version, args.mode, args.scale, args.color_fix, args.tiled_vae, args.tiled_dit,args.tile_size,
        args.overlap, args.unload_dit, dtype, kv_ratio=args.kv_ratio, seed=args.seed, device=args.device, quality=args.quality, output=final,
        output_height=args.output_height, temp_quality=args.temp_quality, pad_align=args.pad_align)
    if args.mode != "tiny-long":
        # Cropped-back dims are arbitrary even numbers; macro_block_size=2 stops
        # imageio's default 16px auto-resize (same precedent as stitch_video_tiles).
        save_video(result, final, fps=fps, quality=args.quality, macro_block_size=2 if args.pad_align else None)

    merge_video_with_audio(final, args.input)
    log("[FlashVSR] Done.", message_type='finish')
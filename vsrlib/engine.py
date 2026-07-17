import os
import json
import shutil

import torch

from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj, Causal_LQ4x_Proj

from .common import log, root, temp
from .media_io import model_downlod, is_video, probe_input, prepare_tensors, count_video_frames
from .preprocess import (
    tensor2video, largest_8n1_leq, next_8n5, compute_scaled_and_padded_dims,
    get_input_params_from_dims, stream_tile_frames, prepare_input_tensor,
)
from .tiling import calculate_tile_coords, create_feather_mask, stitch_video_tiles
from .resume import build_resume_manifest, resume_manifest_hash, prepare_resume_dir, scan_completed_tiles

devices = get_device_list()

def _cli_fps():
    # The CLI args live in run.py (parsed at its import); imported lazily to
    # avoid a circular import. Every caller of main() imports run.py first (the
    # CLI itself and the lada-ex/jasna workers), so run is initialized by now.
    import run
    return run.args.fps

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

def main(input, version, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, dtype, sparse_ratio=2, kv_ratio=3, local_range=11, seed=0, device="auto", quality=6, output=None, output_height=None, temp_quality=8, pad_align=False, resume=False):
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
    if resume and not (tiled_dit and mode == "tiny-long"):
        log("[FlashVSR] --resume only affects --tiled-dit with -m tiny-long; continuing as a normal run.", message_type='warning')

    if mode == "tiny-long":
        # tiny-long streams frames from disk per tile instead of preloading the whole
        # video: a 10-minute 1080p input would need >200GB of RAM as bf16 tensors.
        kind, source, N0, h0, w0, src_fps = probe_input(input)
        _fps = src_fps if kind == "video" else _cli_fps()
        frame_count = N0
        frames = None
        log("[FlashVSR] Preparing frames...", message_type="finish")
    else:
        _frames, fps = prepare_tensors(input, dtype=dtype)
        _fps = fps if is_video(input) else _cli_fps()

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

        completed_tiles = set()
        if mode == "tiny-long":
            manifest = build_resume_manifest(
                kind, input, source, N0, h0, w0, scale, tile_size, tile_overlap, seed,
                version, mode, str(dtype), kv_ratio, local_range, color_fix, sparse_ratio,
                temp_quality, attention="block" if wan_video_dit.USE_BLOCK_ATTN else "sage",
                fps=_fps, num_tiles=len(tile_coords))
            local_temp = os.path.join(temp, "tiles_" + resume_manifest_hash(manifest))
            if prepare_resume_dir(local_temp, manifest, resume):
                completed_tiles = scan_completed_tiles(local_temp, len(tile_coords))
                log(f"[FlashVSR] Resume: {len(completed_tiles)}/{len(tile_coords)} tiles already complete, {len(tile_coords) - len(completed_tiles)} to compute.", message_type='info')

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
            est_disk_gb = (len(tile_coords) - len(completed_tiles)) * (frame_count / max(_fps, 1)) * 0.6 * ((tile_size * scale) / 768.0) ** 2 / 1024.0
            free_disk_gb = shutil.disk_usage(temp).free / 2 ** 30
            log(f"[FlashVSR] Temp tile videos: ~{est_disk_gb:.1f} GiB estimated, all kept until stitching finishes (free disk: {free_disk_gb:.1f} GiB)", message_type='info')
            if est_disk_gb > free_disk_gb:
                log("[FlashVSR] Free disk space may be insufficient for temp tile videos!", message_type='warning')

        if mode == "tiny-long" and len(completed_tiles) == len(tile_coords):
            log("[FlashVSR] All tiles already complete — skipping model load, going straight to stitching.", message_type='info')
            pipe = None
        else:
            pipe = init_pipeline(version, mode, _device, dtype)
            if _device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            if mode == "tiny-long" and i in completed_tiles:
                temp_videos.append(os.path.join(local_temp, f"{i+1:05d}.mp4"))
                log(f"[FlashVSR] Skipping tile {i+1}/{len(tile_coords)}: already complete (resume).", message_type='info')
                continue
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
                # Written on every run, not just --resume ones: the run that crashes
                # is usually the one started without the flag. Records the measured
                # frame count — the pipeline emits fewer frames than its padded input
                # (an internal detail), so the count cannot be derived here.
                n_written = count_video_frames(temp_name)
                if n_written is None:
                    log(f"[FlashVSR] Could not read back tile {i+1} — resume marker skipped.", message_type='warning')
                else:
                    try:
                        with open(temp_name + ".done", "w") as f:
                            json.dump({"frames": n_written}, f)
                    except OSError as e:
                        log(f"[FlashVSR] Could not write resume marker for tile {i+1}: {e}", message_type='warning')
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

import os
import math
import itertools

import torch
import imageio
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from .common import log

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

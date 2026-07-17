import torch
import imageio
import numpy as np
import torch.nn.functional as F

from PIL import Image
from einops import rearrange

from .common import log
from src.models.utils import clean_vram

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

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

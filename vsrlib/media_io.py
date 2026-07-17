import os
import re
import math
import shutil

import torch
import imageio
import ffmpeg
import numpy as np

from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

from .common import log, root

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

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

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

def count_video_frames(path):
    """Frame count of a video file, or None if unreadable. Same fallback chain as
    stitch_video_tiles: count_frames() metadata first, full decode scan if bogus."""
    try:
        rdr = imageio.get_reader(path)
    except Exception:
        return None
    try:
        n = rdr.count_frames()
        if n is None or n <= 0 or (isinstance(n, float) and not math.isfinite(n)):
            rdr.close()
            rdr = imageio.get_reader(path)
            n = sum(1 for _ in rdr)
        return int(n)
    except Exception:
        return None
    finally:
        rdr.close()

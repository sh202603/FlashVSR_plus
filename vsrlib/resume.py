import os
import json
import shutil
import hashlib
import datetime

from .common import log
from .media_io import count_video_frames

def _input_fingerprint(kind, input_path, source):
    # st_mtime_ns (int) instead of st_mtime: floats don't survive a JSON round-trip
    # bit-exactly, which would spuriously invalidate the manifest on every resume.
    if kind == "images":
        return {"type": "images", "path": os.path.abspath(input_path),
                "files": [[os.path.basename(p), os.path.getsize(p)] for p in source]}
    st = os.stat(input_path)
    return {"type": "video", "path": os.path.abspath(input_path),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}

def build_resume_manifest(kind, input_path, source, frame_count, height, width, scale,
                          tile_size, tile_overlap, seed, version, mode, dtype, kv_ratio,
                          local_range, color_fix, sparse_ratio, temp_quality, attention,
                          fps=None, num_tiles=None):
    """"params" holds everything that affects tile pixels, coords, ordering or count —
    two runs with equal params produce interchangeable tile videos. Stitch-only
    settings (output path/quality/height, fps) must stay out of "params" so changing
    them doesn't discard resumable tiles; "info" is for human inspection only."""
    return {
        "format": 1,
        "params": {
            "input": _input_fingerprint(kind, input_path, source),
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "scale": scale,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "seed": seed,
            "version": version,
            "mode": mode,
            "dtype": dtype,
            "kv_ratio": kv_ratio,
            "local_range": local_range,
            "color_fix": color_fix,
            "sparse_ratio": sparse_ratio,
            "temp_quality": temp_quality,
            "attention": attention,
        },
        "info": {
            "fps": fps,
            "num_tiles": num_tiles,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }

def resume_manifest_hash(manifest):
    blob = json.dumps(manifest["params"], sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def prepare_resume_dir(local_temp, manifest, resume):
    """Ensure the deterministic tile dir exists. Returns True when `resume` is set and
    the on-disk manifest's params match exactly (completed tiles may be reused);
    otherwise the dir is recreated from scratch with a fresh manifest."""
    manifest_path = os.path.join(local_temp, "manifest.json")
    if resume and os.path.isdir(local_temp):
        try:
            with open(manifest_path) as f:
                on_disk = json.load(f)
        except (OSError, ValueError):
            on_disk = None
        if isinstance(on_disk, dict) and on_disk.get("params") == manifest["params"]:
            return True
        log("[FlashVSR] --resume: previous tiles were made with different parameters — discarding them.", message_type='warning')
    if os.path.exists(local_temp):
        shutil.rmtree(local_temp)
    os.makedirs(local_temp, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return False

def scan_completed_tiles(local_temp, num_tiles):
    """Indices of tiles whose mp4 is fully written: .done marker present and the
    on-disk frame count matches the count recorded in the marker at completion time.
    The pipeline closes its writer in a finally block, so a crashed tile leaves a
    valid-but-truncated mp4 — mere existence is not completion — and how many frames
    it emits for a given input length is an internal detail (measured, not derived
    here). Any leftover that fails the checks is deleted so the tile re-encodes."""
    completed = set()
    for i in range(num_tiles):
        mp4 = os.path.join(local_temp, f"{i+1:05d}.mp4")
        marker = mp4 + ".done"
        if os.path.exists(mp4) and os.path.exists(marker):
            try:
                with open(marker) as f:
                    recorded = json.load(f).get("frames")
            except (OSError, ValueError):
                recorded = None
            n = count_video_frames(mp4)
            if isinstance(recorded, int) and n == recorded:
                completed.add(i)
                continue
            log(f"[FlashVSR] Tile {i+1}: {n} frames on disk, marker records {recorded} — will re-encode.", message_type='warning')
        for path in (mp4, marker):
            if os.path.exists(path):
                os.remove(path)
    return completed

# ⚡ FlashVSR+

**Optimized inference pipeline based on [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) project**

**Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

**Modified:** lihaoyun6  

**Fork maintainer:** sh202603  

<a href='http://zhuang2002.github.io/FlashVSR'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a> &nbsp;
<a href="https://huggingface.co/datasets/JunhaoZhuang/VSR-120K"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a> &nbsp;
<a href="https://arxiv.org/abs/2510.12747"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

**Your star means a lot for us to develop this project!** :star:

<img src="./teaser.jpg" />

---
### 🤔 What's New?

- Replaced `Block-Sparse-Attention` with `Sparse_SageAttention` to avoid building complex cuda kernels.  
- With the new `tile_dit` method, you can even output 1080P video on 8GB of VRAM.   
- Support copying audio tracks to output files (powered by FFmpeg). 
- Introduced Blackwell GPU support for FlashVSR.  

#### 🍴 What's new in this fork

- Streaming tiled-DiT for `tiny-long` mode: frames are read from disk per tile and the output mp4 is stitched chunk-by-chunk, so long/1080p inputs run on 16GB VRAM with flat host-RAM usage (see the *Low VRAM* section below).  
- Long clips now work correctly in `tiny-long` mode: RoPE frequency tables grow dynamically with clip length, and causal KV caches are carried across chunks (noise stays CPU-resident).  
- `--pad-align`: preserves frame edges on non-tiled runs instead of center-cropping (also available in the web UI).  
- `--resume`: crash recovery for tiled `tiny-long` runs — completed tile videos from an interrupted run are detected and reused on re-run.  
- New low-VRAM CLI knobs: `--output-height`, `--temp-quality`, `--kv-ratio`.  
- uv packaging: `uv sync` sets up the whole environment and installs the `flashvsr-cli` console command.  
- `--accel` (web UI: *"Acceleration"* checkbox): ~1.6× faster inference with FP8 convolutions/linears and fused DiT kernels, and 1–2 GiB less VRAM, on RTX 40 series or newer with FlashVSR v1.1 (`-v 11`). Other GPUs and `-v 10` fall back to the standard path automatically (see *Acceleration* below).  

---
### 🚀 Getting Started

Follow these steps to set up and run **FlashVSR** on your local machine:

> ⚠️ **Note:** This project is primarily designed and optimized for **4× video super-resolution**.  
> We **strongly recommend** using the **4× SR setting** to achieve better results and stability. ✅

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sh202603/FlashVSR_plus
cd FlashVSR_plus
````

#### 2️⃣ Set Up the Python Environment

Recommended: [uv](https://docs.astral.sh/uv/). A single command creates `.venv`, installs all dependencies (torch/torchvision come from the PyTorch **cu130** index, configured in `pyproject.toml`) and installs the project in editable mode, which provides the `flashvsr-cli` command:

```bash
uv sync
```

> `pyproject.toml` is the canonical dependency list; `requirements.txt` mirrors it for plain-pip users.

Alternatively, set up manually with conda + pip:

```bash
conda create -n flashvsr
conda activate flashvsr

# for CUDA 12.8
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

# for CUDA 13.0
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

#### 3️⃣ Download Model Weights

- When you run FlashVSR+ for the first time, it will automatically download all required models from HuggingFace.  

- You can also manually download all files from [FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR) and put them in the following location:  

```
./models/FlashVSR/
│
├── LQ_proj_in.ckpt                                   
├── TCDecoder.ckpt                                    
├── Wan2.1_VAE.pth                                    
├── diffusion_pytorch_model_streaming_dmd.safetensors 
└── README.md
```  

- With `-v 11` the pipeline uses [FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) weights instead, auto-downloaded into `./models/FlashVSR-v1.1/` the same way (default `-v 10` → `./models/FlashVSR/`).  

#### 4️⃣ Run Inference

CLI example:

```bash
python run.py -i ./inputs/example0.mp4 -s 4 ./

# after `uv sync` (or `uv pip install -e .`), the venv also provides a console command:
flashvsr-cli -i ./inputs/example0.mp4 -s 4 ./
```

- `--pad-align` pads the upscaled frame to the next multiple of 128 instead of center-cropping (the default loses up to 127 output pixels per dimension on non-tiled runs), then crops the output back to exactly `scale × input` size. Tiled-DiT runs already preserve the full frame, so the flag is a no-op there. The web UI exposes the same option as the *"Preserve full frame"* checkbox.

Or use gradio web ui:  

```bash
python webui.py
```

#### 💾 Low VRAM (16GB) and long 1080p inputs

In `tiny-long` mode the CLI streams frames from disk per tile and stitches tile videos chunk-by-chunk, so host RAM and VRAM stay flat regardless of clip length:

```bash
flashvsr-cli -i input.mp4 -m tiny-long --tiled-dit --tile-size 192 --overlap 24 --output-height 2160 -v 11 ./
```

- `--tile-size 192` keeps the per-tile GPU footprint around **10 GiB** (measured on an RTX 5080 under Linux). The default 256 needs ~13.5 GiB and is only ~5% faster overall (fewer tiles, but per-tile time scales with tile area), so on 16GB cards there is no reason not to use 192. Per-tile peak VRAM is logged so you can tune this. `tile_size × scale` must be a multiple of 128.
- `--output-height` downscales the stitched result after blending (the model is 4x-fixed, so a 1080p input otherwise produces a 7680×4320 file).
- `--kv-ratio` (default 3) sets the KV-cache length of the sparse attention; lowering it saves additional VRAM at some quality cost.
- Temp tile videos are kept until stitching finishes (`--temp-quality 8` ≈ 0.6 MB/s per tile); the run logs a disk-space estimate at startup.
- If a long run crashes or is killed, re-run the same command with `--resume`: completed tiles are verified and skipped, and if all tiles were done the run goes straight to stitching. Changing any parameter that affects tile content (input, seed, tile size, …) starts a fresh tile set instead; stale tiles are cleaned up by the next run without `--resume`.
- Throughput reality: a 1080p input is split into 84 tiles at tile 192 — measured pace extrapolates to roughly **a day (~23h) per 10 minutes of video** on an RTX 5080. For a ~4× faster, lower-fidelity pass, downscale the input to 540p first and let the 4x model produce 2160p directly.
- Constant-frame-rate input is recommended; VFR sources may end with a few duplicated tail frames.

#### ⚡ Acceleration (`--accel`)

```bash
flashvsr-cli -i input.mp4 -m tiny-long -v 11 --accel ./
```

`--accel` (or the environment variable `FLASHVSR_ACCEL=1`) swaps in three groups of faster implementations:

| Part | What runs faster | Environment variable |
|---|---|---|
| FP8 convolutions | TCDecoder and LQ projector convolutions in FP8 (cuDNN graph API) | `FLASHVSR_FP8_CONV` |
| FP8 DiT | DiT linears and FFN in FP8 | `FLASHVSR_FP8_DIT` |
| Fused DiT | RMSNorm+RoPE and AdaLN as fused Triton kernels | `FLASHVSR_FUSED_DIT` |

Setting a part's variable to `1` enables just that part; `0` removes it even under `--accel`. The DiT parts come from [flashvsr-sm89-ops](https://github.com/aireet/flashvsr-sm89-ops) (vendored in `vsrlib/sm89_ops/`, Apache-2.0).

Measured on an RTX 5060 Ti 16 GB (`tiny-long`, `-v 11`, 90 frames):

| | 256² → 512² (scale 2) | 256² → 1024² (scale 4) |
|---|---|---|
| standard | 5.42 s, peak 5.48 GiB | 21.76 s, peak 11.58 GiB |
| `--accel` | 3.32 s (1.63×), peak 4.35 GiB | 13.08 s (1.66×), peak 9.47 GiB |

- **Requirements:** FlashVSR v1.1 weights (`-v 11`), an FP8-capable NVIDIA GPU (sm89+: RTX 40 series or newer) and `--dtype bf16` (the default). The FP8 convolutions also need `nvidia-cudnn-frontend` (a regular dependency) with cuDNN ≥ 9.17 (bundled with the cu130 torch wheel).
- **FlashVSR v1.0 (`-v 10`) is not supported:** the FP8 calibration and the quality checks were done on v1.1 only, so with `-v 10` every part is skipped (one-line warning) and the standard path runs.
- **Automatic fallback:** at startup each part is checked (GPU, dtype, libraries, a trial build and a warmup run). A part that can't run is skipped with a one-line warning and the standard code path runs instead; if every part is skipped, the output is bit-identical to a run without `--accel`. A part that fails in the middle of a run is switched off for the rest of that process.
- **Output:** the result differs slightly from a standard run (FP8 rounding; the sparse attention amplifies tiny numeric differences). In our checks the flow-warping error stayed within 1.2× of the standard run, mostly driven by the FP8 TCDecoder; if you see flicker, try `FLASHVSR_FP8_CONV=0`.
- **`--resume`:** the active parts are part of the tile-set fingerprint, so tiles made with different acceleration settings are never mixed. Resuming with the same settings reuses completed tiles.
- The web UI has the same option as the *"Acceleration"* checkbox. `full` mode decodes with the Wan VAE, so only the LQ projector's convolutions run in FP8 there.

---

### 🤗 Feedback & Support

We welcome feedback and issues. Thank you for trying **FlashVSR+**

---

### 📄 Acknowledgments

We gratefully acknowledge the following open-source projects:

* **FlashVSR** — [https://github.com/OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
* **DiffSynth Studio** — [https://github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Sparse_SageAttention** — [https://github.com/jt-zhang/Sparse\_SageAttention_API](https://github.com/jt-zhang/Sparse_SageAttention_API)
* **taehv** — [https://github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)
* **flashvsr-sm89-ops** — [https://github.com/aireet/flashvsr-sm89-ops](https://github.com/aireet/flashvsr-sm89-ops) (FP8 linear and fused DiT kernels used by `--accel`, vendored in `vsrlib/sm89_ops/`)

---

### 📞 Contact

* **Junhao Zhuang**
  Email: [zhuangjh23@mails.tsinghua.edu.cn](mailto:zhuangjh23@mails.tsinghua.edu.cn)

---

### 📜 Citation

```bibtex
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}
```

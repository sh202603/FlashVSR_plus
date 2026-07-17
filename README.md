# ⚡ FlashVSR+

**Optimized inference pipeline based on [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) project**

**Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

**Modified:** lihaoyun6  

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

---
### 🚀 Getting Started

Follow these steps to set up and run **FlashVSR** on your local machine:

> ⚠️ **Note:** This project is primarily designed and optimized for **4× video super-resolution**.  
> We **strongly recommend** using the **4× SR setting** to achieve better results and stability. ✅

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lihaoyun6/FlashVSR_plus
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

#### 4️⃣ Run Inference

CLI example:

```bash
python run.py -i ./inputs/example0.mp4 -s 4 ./

# after `uv sync` (or `uv pip install -e .`), the venv also provides a console command:
flashvsr-cli -i ./inputs/example0.mp4 -s 4 ./
```

- `--pad-align` pads the upscaled frame to the next multiple of 128 instead of center-cropping (the default loses up to 127 output pixels per dimension on non-tiled runs), then crops the output back to exactly `scale × input` size. Tiled-DiT runs already preserve the full frame, so the flag is a no-op there.

Or use gradio web ui:  

```bash
python webui.py
```

#### 💾 Low VRAM (16GB) and long 1080p inputs

In `tiny-long` mode the CLI streams frames from disk per tile and stitches tile videos chunk-by-chunk, so host RAM and VRAM stay flat regardless of clip length:

```bash
python run.py -i input.mp4 -m tiny-long --tiled-dit --tile-size 192 --overlap 24 --output-height 2160 ./
```

- `--tile-size 192` keeps the per-tile GPU footprint around **10 GiB** (measured on an RTX 5080 under Linux). The default 256 needs ~13.5 GiB and is only ~5% faster overall (fewer tiles, but per-tile time scales with tile area), so on 16GB cards there is no reason not to use 192. Per-tile peak VRAM is logged so you can tune this. `tile_size × scale` must be a multiple of 128.
- `--output-height` downscales the stitched result after blending (the model is 4x-fixed, so a 1080p input otherwise produces a 7680×4320 file).
- Temp tile videos are kept until stitching finishes (`--temp-quality 8` ≈ 0.6 MB/s per tile); the run logs a disk-space estimate at startup.
- Throughput reality: a 1080p input is split into 84 tiles at tile 192 — measured pace extrapolates to roughly **a day (~23h) per 10 minutes of video** on an RTX 5080. For a ~4× faster, lower-fidelity pass, downscale the input to 540p first and let the 4x model produce 2160p directly.
- Constant-frame-rate input is recommended; VFR sources may end with a few duplicated tail frames.

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

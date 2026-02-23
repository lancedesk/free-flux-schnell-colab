# Free FLUX.1 schnell on Google Colab (ComfyUI)

Run [Black Forest Labs' FLUX.1 schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) — the fastest FLUX model — for free on Google Colab T4 GPU via [ComfyUI](https://github.com/comfyanonymous/ComfyUI). No local GPU required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lancedesk/free-stable-diffusion-colab/blob/main/flux-schnell-colab/flux_schnell_colab.ipynb)

---

## FLUX.1 schnell vs dev vs Stable Diffusion

| Feature | SD 1.5 | FLUX.1 dev | FLUX.1 schnell |
|---------|--------|------------|----------------|
| Parameters | 860M | 12B | **12B** |
| Architecture | UNet | DiT | **DiT** |
| Steps needed | 20–30 | 20–25 | **4–8** |
| Time/image (T4) | ~30–45 sec | ~60–90 sec | **~15–25 sec** |
| Image quality | Good (fine-tuned) | Excellent | **Very good** |
| Prompt adherence | Moderate | Excellent | **Very good** |
| Text in images | Poor | Good | **Good** |
| License | Open | Non-commercial | **Apache 2.0** |

> **schnell** = "fast" in German. Distilled from FLUX.1 pro for 1–4 step generation. Slightly less detailed than dev but **3–4× faster** with a **fully open Apache 2.0 license**.

## Features

- **FLUX.1 schnell fp8** — 12B DiT model, Apache 2.0 license, no Hugging Face token required
- **4-step generation** — ~15–25 sec/image on T4 (faster than SD 1.5 + hires fix)
- **ComfyUI** — node-based interface with native FLUX support
- **ComfyUI-Manager** — install extra nodes in-browser
- **Dual text encoders** — T5-XXL fp8 (CPU-offloaded) + CLIP-L
- **Cloudflare Tunnel** — reliable public URL, replaces Gradio's unstable `--share`
- **Pre-built workflow** — guidance-free schnell workflow loads automatically

## Quick Start

1. Click the **Open in Colab** badge above, or go to [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → select `flux_schnell_colab.ipynb`
2. **Runtime → Change runtime type** → Hardware accelerator: **T4 GPU** → **Save**
3. **Runtime → Run all** — ~6 minutes total
4. **Click the `trycloudflare.com` URL** that appears in Cell 3 output → ComfyUI opens!
5. Click **Queue Prompt** — first image generates in ~30–40 sec (CUDA compilation), then ~15–25 sec

## Pre-configured Defaults

| Setting | Value | Notes |
|---------|-------|-------|
| **Model** | **FLUX.1 schnell fp8** | 12B DiT, Apache 2.0 |
| **Sampler** | **euler** | Recommended for FLUX |
| **Scheduler** | **simple** | Optimal for FLUX |
| **Steps** | **4** | Distilled sweet spot; 6–8 max |
| **Guidance** | **None** | schnell is guidance-free (CFG=0 baked in) |
| **Resolution** | **1024 × 1024** | Native FLUX resolution |
| T5-XXL | fp8, CPU-offloaded | Keeps GPU free for the UNet |
| VRAM mode | `--lowvram` | Streams model layers to GPU as needed |

> **Why no guidance node?** FLUX.1 schnell was distilled with `cfg=0` baked in. Adding `FluxGuidance` degrades output quality — the workflow correctly uses `BasicGuider` directly.

## schnell vs dev — which to use?

| Use schnell when… | Use dev when… |
|-------------------|---------------|
| Prototyping / trying many prompts quickly | Final high-quality renders |
| Need commercial license (Apache 2.0) | Personal / research use only |
| Want ~3–4× faster generation | Want absolute maximum detail |
| Batch generating variations | Quality over quantity |

> Both use the **same ComfyUI setup and same model files** (T5-XXL, CLIP-L, VAE). If you run both notebooks, Cell 2 detects existing files and skips re-downloading.

## VRAM Breakdown (T4 — 15 GB)

| Component | Size | Location |
|-----------|------|----------|
| FLUX.1 schnell fp8 | ~12 GB | GPU (streamed via `--lowvram`) |
| T5-XXL fp8 | ~4.9 GB | CPU (offloaded) |
| CLIP-L | ~246 MB | GPU |
| VAE (ae) | ~335 MB | GPU |

## Resolution Guide

Use `EmptySD3LatentImage` in the workflow to change resolution:

| Format | Width | Height |
|--------|-------|--------|
| Square | 1024 | 1024 |
| Landscape | 1360 | 768 |
| Portrait | 768 | 1360 |
| Wide | 1536 | 640 |

For maximum speed, use **768×768** (~8–12 sec/image at 4 steps).

## Prompting Tips

- schnell has strong prompt adherence despite fewer steps — be specific
- No negative prompt needed
- Camera details improve realism: `85mm lens, f/2.8, Fujifilm XT3, film grain`
- Text rendering works: `a chalkboard sign that says "Daily Special"`
- Style keywords: `cinematic, moody, editorial lighting, golden hour, overcast`
- For batch prototyping: set `batch_size: 4` in `EmptySD3LatentImage` — 4 images in ~60 sec

## Step Count Guide

| Steps | Result |
|-------|--------|
| 1 | Very rough sketch, extremely fast |
| 4 | **Recommended** — good quality, ~15-25 sec |
| 6–8 | Marginal improvement, slower |
| 10+ | No improvement over 8 for schnell |

## API Usage

```python
import requests, json, time
from PIL import Image
from io import BytesIO

BASE = "https://your-url.trycloudflare.com"

wf = json.load(open('/content/ComfyUI/user/default/workflows/flux_schnell_default.json'))
wf["4"]["inputs"]["text"] = "your prompt here"
wf["7"]["inputs"]["noise_seed"] = 99999
# wf["5"]["inputs"]["batch_size"] = 4  # generate 4 at once

prompt_id = requests.post(f"{BASE}/prompt", json={"prompt": wf}).json()["prompt_id"]

while True:
    history = requests.get(f"{BASE}/history/{prompt_id}").json()
    if prompt_id in history:
        break
    time.sleep(1)

for node_id, output in history[prompt_id]["outputs"].items():
    for img in output.get("images", []):
        data = requests.get(f"{BASE}/view", params={
            "filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]
        }).content
        Image.open(BytesIO(data)).save(f"schnell_{img['filename']}")
        print(f"Saved: schnell_{img['filename']}")
```

## How It Works

| Cell | What it does | Time |
|------|-------------|------|
| Cell 1 | Clone ComfyUI + ComfyUI-Manager, install PyTorch cu121 | ~3-4 min |
| Cell 2 | Download FLUX.1 schnell fp8, T5-XXL fp8, CLIP-L, VAE (~18 GB) | ~3-4 min |
| Cell 3 | Write guidance-free workflow, start Cloudflare tunnel, launch ComfyUI | ~30 sec |

**All models Apache 2.0 / no token required:**
- FLUX.1 schnell fp8 → [Kijai/flux-fp8](https://huggingface.co/Kijai/flux-fp8)
- T5-XXL fp8 + CLIP-L → [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders)
- VAE → [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of memory | `--lowvram` is already set. Reduce to 768×768 or lower batch size. |
| Blurry / low detail at 4 steps | Try 6–8 steps. Going above 8 won't help for schnell. |
| Workflow not loading | Click folder icon in ComfyUI → `flux_schnell_default` |
| Black / corrupted images | Re-run Cell 2 to re-download `ae.safetensors` |
| `UNETLoader` node missing | Re-run Cell 1 — ComfyUI may need updating |
| Tunnel URL missing | Look in Cell 3 output for `trycloudflare.com` |
| T5 memory warnings | Normal — T5 runs on CPU in fp8 |

## Known Limitations

- Free Colab sessions last ~12 hours, then reset — **save your images!**
- The public URL changes every session
- schnell quality is slightly below dev at 4 steps (a deliberate speed/quality trade-off)

## License

**FLUX.1 schnell** is released under the [Apache 2.0 License](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md) — free for commercial use, modification, and redistribution.

## Resources

- [FLUX.1 schnell on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [ComfyUI FLUX Examples](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
- [FLUX.1 dev notebook](../flux-dev-colab/) — higher quality, non-commercial
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion)

---

**Star the repo if this saved you time!**


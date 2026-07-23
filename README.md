# AI vs. Human Image Classifier using Diffusion Model Features

A binary image classifier that detects whether an image was AI-generated or 
created by a human — built by extracting features from pretrained diffusion 
model components (VAE encoders and diffusion U-Nets) rather than training a 
vision model from scratch.

## Why diffusion models as feature extractors?

The hypothesis: AI-generated images carry subtle artifacts from the 
generation process itself. Diffusion models (VAEs, U-Nets) are trained to 
understand and reconstruct the image manifold, so their internal 
representations might encode signals that separate "real" from "generated" 
images better than a generic, from-scratch CNN would.

## Overview

Four pretrained extractors are compared, each feeding into the same 
downstream classifier:

| Extractor | Source Model | Feature Type |
|---|---|---|
| VAE (SD) | `stabilityai/sd-vae-ft-mse` | Encoder latent (sampled from latent distribution) |
| VAE (Tiny) | `stabilityai/sd-vae-ft-ema` | Encoder latent (sampled from latent distribution) |
| DDPM | `google/ddpm-cifar10-32` | U-Net noise prediction @ timestep=10 |
| DDIM | `google/ddpm-cifar10-32` | U-Net noise prediction @ timestep=20 |

## Pipeline

1. **Data loading** — Custom PyTorch `Dataset` reads image paths and binary 
   labels (AI vs. Human) from a CSV file, with resize (256) → center crop 
   (224) → tensor transforms applied.
2. **Feature extraction** — Each pretrained model runs in `eval()` mode under 
   `torch.no_grad()` to produce a flattened feature vector per image.
3. **Classification** — A `sklearn.linear_model.LogisticRegression` model is 
   trained independently on each extractor's features.
4. **Evaluation** — Accuracy, Precision, Recall, F1, and AUC are computed for 
   each extractor, along with confusion matrices, ROC curves, and sample 
   predictions. Results are compiled into a final comparison table and bar 
   chart.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **DDIM** | **0.745** | 0.743 | 0.792 | **0.767** | 0.778 |
| DDPM | 0.715 | 0.706 | 0.792 | 0.747 | **0.785** |
| VAE (Tiny) | 0.565 | 0.602 | 0.528 | 0.563 | 0.603 |
| VAE (SD) | 0.560 | 0.594 | 0.538 | 0.564 | 0.604 |

**Key finding:** Diffusion U-Net (DDPM/DDIM) noise-prediction features 
substantially outperform VAE latent features for distinguishing AI-generated 
from human images. This suggests the denoising network captures more 
discriminative artifacts of the generation process than the VAE's 
compression-oriented latent space — DDIM was the best overall performer 
(F1 = 0.767).

## Tech Stack

- **PyTorch** / **torchvision**
- **Hugging Face `diffusers`** (AutoencoderKL, DDPMPipeline, DDIMPipeline)
- **Hugging Face `transformers`**
- **scikit-learn** (LogisticRegression, classification metrics)
- **pandas**, **numpy**
- **matplotlib** (visualizations)


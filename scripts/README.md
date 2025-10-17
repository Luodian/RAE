# RAE Evaluation Scripts

Scripts to reproduce rFID and PSNR evaluation results for the RAE (Regularized Autoencoder) model.

## Overview

This directory contains scripts to:
1. Download ImageNet-1K validation set (50k images)
2. Reconstruct images using RAE
3. Prepare reference set with center cropping
4. Evaluate rFID using ADM suite
5. Evaluate PSNR

## Quick Start

### Run Full Evaluation Pipeline

```bash
cd /opt/tiger/RAE
bash scripts/run_full_evaluation.sh
```

This will execute all steps automatically.

## Step-by-Step Guide

### 1. Download ImageNet Validation Set

Downloads 50k validation images from HuggingFace:

```bash
bash scripts/01_download_imagenet_val.sh
```

**Output:** `/opt/tiger/RAE/data/imagenet_val/` (organized in ImageFolder format)

### 2. Reconstruct Validation Set

Reconstructs all 50k images using the RAE model:

```bash
bash scripts/02_reconstruct_val_set.sh
```

**Configuration:**
- Model: SigLIP2 pretrained RAE
- Resolution: 256x256
- Precision: bfloat16
- GPUs: 8
- Batch size: 8 per GPU

**Output:**
- Reconstructed images: `/opt/tiger/RAE/evaluations/reconstructions/`
- NPZ file: `RAE-pretrained-bs8-bf16.npz`

### 3. Prepare Reference Set

Prepares the reference set by applying center cropping to original images:

```bash
bash scripts/03_prepare_reference_set.sh
```

This follows the same center cropping procedure as the reconstruction pipeline.

**Output:** `/opt/tiger/RAE/evaluations/reference_set_256.npz`

### 4. Setup ADM Evaluation Suite

Downloads and sets up the ADM evaluation suite for rFID computation:

```bash
bash scripts/04_setup_adm_evaluation.sh
```

**Output:** `/opt/tiger/RAE/evaluations/adm_suite/`

### 5. Evaluate rFID

Computes rFID-50K score using the standard FID evaluation protocol:

```bash
bash scripts/05_evaluate_rfid.sh
```

This compares:
- Reference: Original validation images (center-cropped)
- Generated: RAE reconstructions

**Note:** Update `RECONSTRUCTION_NPZ` path in the script if your reconstruction NPZ has a different name.

### 6. Evaluate PSNR

Computes Peak Signal-to-Noise Ratio between original and reconstructed images:

```bash
bash scripts/06_evaluate_psnr.sh
```

**Output:**
- Console output with mean/median/std PSNR
- Detailed results: `/opt/tiger/RAE/evaluations/reconstructions/psnr_results.txt`

## Configuration

### Changing Image Resolution

To evaluate at different resolutions (e.g., 512):

1. Update `IMAGE_SIZE` in:
   - `02_reconstruct_val_set.sh`
   - `03_prepare_reference_set.sh`

2. Update output paths to reflect resolution (e.g., `reference_set_512.npz`)

### Using Custom Model Checkpoints

To evaluate a fine-tuned model:

1. Update `CONFIG_PATH` in `02_reconstruct_val_set.sh` to point to your model config

2. Ensure your config file specifies the correct checkpoint path

### Adjusting Batch Size

For memory constraints, modify `BATCH_SIZE` in:
- `02_reconstruct_val_set.sh` (reconstruction)
- `05_evaluate_rfid.sh` (FID computation)

## Directory Structure

```
/opt/tiger/RAE/
├── scripts/
│   ├── 01_download_imagenet_val.sh      # Download validation set
│   ├── 02_reconstruct_val_set.sh        # Reconstruct images
│   ├── 03_prepare_reference_set.py      # Reference prep (Python)
│   ├── 03_prepare_reference_set.sh      # Reference prep (Shell)
│   ├── 04_setup_adm_evaluation.sh       # Setup ADM suite
│   ├── 05_evaluate_rfid.sh              # Compute rFID
│   ├── 06_evaluate_psnr.py              # PSNR calculation (Python)
│   ├── 06_evaluate_psnr.sh              # PSNR evaluation (Shell)
│   ├── run_full_evaluation.sh           # Master script
│   └── README.md                        # This file
├── data/
│   └── imagenet_val/                    # Downloaded validation set
└── evaluations/
    ├── reference_set_256.npz            # Reference images
    ├── reconstructions/                 # Reconstructed images
    │   ├── RAE-pretrained-bs8-bf16.npz
    │   └── psnr_results.txt
    └── adm_suite/                       # ADM evaluation code
```

## Requirements

All dependencies are managed through the RAE virtual environment at `/opt/tiger/RAE/.venv`.

Required packages (automatically installed):
- datasets (HuggingFace)
- torch, torchvision
- PIL, numpy
- blobfile, scipy (for ADM suite)
- pytorch-fid (fallback for FID computation)

## Troubleshooting

### NCCL/CUDA Errors

The scripts include `NCCL_NVLS_ENABLE=0` to avoid compatibility issues. If you still encounter NCCL errors:

```bash
# Add NCCL debugging
NCCL_DEBUG=INFO bash scripts/02_reconstruct_val_set.sh
```

### Out of Memory

Reduce batch size in reconstruction:
- Edit `02_reconstruct_val_set.sh`
- Lower `BATCH_SIZE` (e.g., from 8 to 4 or 2)

### Different NPZ File Names

The reconstruction script automatically generates NPZ file names based on configuration. Update the paths in:
- `05_evaluate_rfid.sh`
- `06_evaluate_psnr.sh`

Check the actual NPZ filename in `/opt/tiger/RAE/evaluations/reconstructions/` and update the `RECONSTRUCTION_NPZ` variable accordingly.

## Citation

If you use these evaluation scripts, please cite the RAE paper and the ADM evaluation suite:

```bibtex
@article{rae2024,
  title={Regularized Autoencoders for Image Representation Learning},
  author={...},
  journal={...},
  year={2024}
}

@article{dhariwal2021diffusion,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Dhariwal, Prafulla and Nichol, Alexander},
  journal={NeurIPS},
  year={2021}
}
```

## References

- [ADM Evaluation Suite](https://github.com/openai/guided-diffusion)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- [ImageNet-1K Dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

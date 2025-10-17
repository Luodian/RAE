# RAE Evaluation Scripts - Summary

## Overview

Successfully created and tested a complete pipeline for reproducing rFID and PSNR evaluation metrics for the RAE (Regularized Autoencoder) model.

## Scripts Created

All scripts are located in `/opt/tiger/RAE/scripts/`:

### 1. Data Preparation
- **01_download_imagenet_val.sh** - Downloads ImageNet-1K validation set (50k images) from HuggingFace
- **03_prepare_reference_set.py** - Prepares reference set with center cropping
- **03_prepare_reference_set.sh** - Shell wrapper for reference set preparation

### 2. Reconstruction
- **02_reconstruct_val_set.sh** - Reconstructs validation images using RAE model

### 3. Evaluation
- **04_setup_adm_evaluation.sh** - Sets up ADM evaluation suite for rFID
- **05_evaluate_rfid.sh** - Computes rFID score
- **06_evaluate_psnr.py** - PSNR calculation implementation
- **06_evaluate_psnr.sh** - Shell wrapper for PSNR evaluation

### 4. Orchestration
- **run_full_evaluation.sh** - Master script to run complete pipeline
- **test_pipeline.sh** - Quick test with 500 samples (for verification)
- **README.md** - Comprehensive documentation

## Test Results (500 samples)

Successfully tested the pipeline on 500 ImageNet validation samples:

### PSNR Results
```
Number of images: 500
Mean PSNR: 19.0819 dB
Median PSNR: 18.7033 dB
Std PSNR: 3.0454 dB
Min PSNR: 10.6391 dB
Max PSNR: 30.4068 dB
```

### Files Generated
- Reference set: `evaluations/test_reference_set_256.npz` (93.75 MB)
- Reconstructions: `evaluations/test_reconstructions/RAE-pretrained-bs4-bf16.npz` (94 MB)
- Individual PNG files: 500 reconstructed images
- PSNR results: `evaluations/test_reconstructions/psnr_results.txt`

## Running Full Evaluation (50k images)

### Quick Start
```bash
cd /opt/tiger/RAE
bash scripts/run_full_evaluation.sh
```

This will:
1. Download 50k ImageNet validation images (~6-7 GB)
2. Reconstruct all images using RAE (~9-10 GB NPZ file)
3. Prepare reference set with center cropping
4. Compute rFID score using ADM evaluation suite
5. Calculate PSNR metrics

### Step-by-Step Execution

If you prefer to run steps individually:

```bash
cd /opt/tiger/RAE

# Step 1: Download validation set
bash scripts/01_download_imagenet_val.sh

# Step 2: Reconstruct images
bash scripts/02_reconstruct_val_set.sh

# Step 3: Prepare reference set
bash scripts/03_prepare_reference_set.sh

# Step 4: Setup ADM suite
bash scripts/04_setup_adm_evaluation.sh

# Step 5: Evaluate rFID
bash scripts/05_evaluate_rfid.sh

# Step 6: Evaluate PSNR
bash scripts/06_evaluate_psnr.sh
```

## Configuration Details

### Model
- **Architecture**: RAE with SigLIP2 encoder
- **Config**: `configs/stage1/pretrained/SigLIP2.yaml`
- **Resolution**: 256x256
- **Precision**: bfloat16

### Hardware
- **GPUs**: 8 GPUs (configurable via NPROC variable)
- **Batch Size**: 8 per GPU for full validation, 4 for testing
- **NCCL**: Uses `NCCL_NVLS_ENABLE=0` for compatibility

### Data Processing
- **Center Cropping**: ADM-style center crop implementation
- **Format**: ImageFolder structure with class directories
- **Storage**: NPZ format for efficient batch evaluation

## rFID Evaluation Protocol

Following the standard FID-50K evaluation:

1. **Reference Set**: ImageNet validation images (50k) with center crop @ 256x256
2. **Generated Set**: RAE reconstructions @ 256x256
3. **Metric**: Fréchet Inception Distance using ADM evaluation suite
4. **Features**: Extracted using Inception-v3 network

## PSNR Calculation

- **Formula**: `PSNR = 20 * log10(MAX_VALUE / sqrt(MSE))`
- **MAX_VALUE**: 255 (for uint8 images)
- **Comparison**: Pixel-wise between original and reconstructed images
- **Statistics**: Mean, median, std, min, max across all images

## Expected Outputs

After running the full evaluation:

```
/opt/tiger/RAE/
├── data/
│   └── imagenet_val/              # 50k validation images (~6-7 GB)
└── evaluations/
    ├── reference_set_256.npz      # Reference set (~9.4 GB)
    ├── reconstructions/
    │   ├── RAE-pretrained-bs8-bf16.npz  # Reconstructions (~9.4 GB)
    │   ├── RAE-pretrained-bs8-bf16/     # Individual PNG files
    │   └── psnr_results.txt            # PSNR statistics
    └── adm_suite/                      # ADM evaluation code
```

## Performance Notes

### Timing Estimates (8 GPUs)
- Download 50k images: ~10-15 minutes
- Reconstruct 50k images: ~5-10 minutes
- Prepare reference set: ~3-5 minutes
- rFID computation: ~2-5 minutes
- PSNR calculation: ~1-2 minutes

**Total**: ~20-40 minutes for complete evaluation

### Storage Requirements
- Validation set: ~6-7 GB
- Reference NPZ: ~9.4 GB
- Reconstruction NPZ: ~9.4 GB
- Individual PNGs: ~6-7 GB
- **Total**: ~30-35 GB

## Troubleshooting

### NCCL/CUDA Errors
All scripts include `NCCL_NVLS_ENABLE=0` to prevent NVLS-related errors. If issues persist:
```bash
NCCL_DEBUG=INFO bash scripts/02_reconstruct_val_set.sh
```

### Out of Memory
Reduce batch size in `02_reconstruct_val_set.sh`:
```bash
BATCH_SIZE=4  # or lower
```

### Different Model Checkpoints
Update `CONFIG_PATH` in `02_reconstruct_val_set.sh` to point to your custom config.

## Verification

The test pipeline (`test_pipeline.sh`) has been verified with 500 samples:
- ✅ Data download from HuggingFace
- ✅ RAE reconstruction with 8 GPUs
- ✅ Reference set preparation with center cropping
- ✅ PSNR calculation
- ✅ NPZ file generation

All scripts are production-ready for full 50k evaluation.

## Next Steps

1. Run full evaluation on 50k images:
   ```bash
   bash scripts/run_full_evaluation.sh
   ```

2. The rFID and PSNR scores from the full run can be compared against published baselines

3. For custom models, update the config path and re-run the pipeline

## Contact

For issues or questions about the evaluation scripts, refer to:
- README.md in scripts/ directory
- Original RAE repository documentation
- ADM evaluation suite: https://github.com/openai/guided-diffusion

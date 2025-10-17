#!/usr/bin/env python3
"""
Evaluate PSNR (Peak Signal-to-Noise Ratio) between original and reconstructed images.
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, max_value: float = 255.0) -> float:
    """
    Calculate PSNR between two images.

    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        max_value: Maximum possible pixel value (default: 255 for uint8)

    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 20 * math.log10(max_value / math.sqrt(mse))
    return psnr


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSNR between original and reconstructed images')
    parser.add_argument('--reference-npz', type=str, required=True,
                        help='Path to reference set NPZ file (original images)')
    parser.add_argument('--reconstruction-npz', type=str, required=True,
                        help='Path to reconstruction NPZ file')
    parser.add_argument('--max-value', type=float, default=255.0,
                        help='Maximum pixel value (default: 255)')
    args = parser.parse_args()

    # Load NPZ files
    print(f"Loading reference set from {args.reference_npz}...")
    ref_data = np.load(args.reference_npz)
    ref_key = list(ref_data.keys())[0]
    reference_images = ref_data[ref_key]

    print(f"Loading reconstructions from {args.reconstruction_npz}...")
    recon_data = np.load(args.reconstruction_npz)
    recon_key = list(recon_data.keys())[0]
    reconstructed_images = recon_data[recon_key]

    print(f"Reference shape: {reference_images.shape}")
    print(f"Reconstruction shape: {reconstructed_images.shape}")

    # Verify shapes match
    if reference_images.shape != reconstructed_images.shape:
        print("Warning: Shape mismatch!")
        min_samples = min(len(reference_images), len(reconstructed_images))
        print(f"Using first {min_samples} samples for comparison")
        reference_images = reference_images[:min_samples]
        reconstructed_images = reconstructed_images[:min_samples]

    num_images = len(reference_images)
    print(f"\nEvaluating PSNR on {num_images} images...")

    # Calculate PSNR for each image
    psnr_values = []
    for i in tqdm(range(num_images), desc="Computing PSNR"):
        psnr = calculate_psnr(reference_images[i], reconstructed_images[i], args.max_value)
        psnr_values.append(psnr)

    psnr_values = np.array(psnr_values)

    # Print statistics
    print("\n" + "=" * 60)
    print("PSNR Evaluation Results")
    print("=" * 60)
    print(f"Number of images: {num_images}")
    print(f"Mean PSNR: {np.mean(psnr_values):.4f} dB")
    print(f"Median PSNR: {np.median(psnr_values):.4f} dB")
    print(f"Std PSNR: {np.std(psnr_values):.4f} dB")
    print(f"Min PSNR: {np.min(psnr_values):.4f} dB")
    print(f"Max PSNR: {np.max(psnr_values):.4f} dB")
    print("=" * 60)

    # Save detailed results
    output_path = Path(args.reconstruction_npz).parent / "psnr_results.txt"
    with open(output_path, 'w') as f:
        f.write("PSNR Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Reference: {args.reference_npz}\n")
        f.write(f"Reconstruction: {args.reconstruction_npz}\n")
        f.write(f"Number of images: {num_images}\n")
        f.write(f"Mean PSNR: {np.mean(psnr_values):.4f} dB\n")
        f.write(f"Median PSNR: {np.median(psnr_values):.4f} dB\n")
        f.write(f"Std PSNR: {np.std(psnr_values):.4f} dB\n")
        f.write(f"Min PSNR: {np.min(psnr_values):.4f} dB\n")
        f.write(f"Max PSNR: {np.max(psnr_values):.4f} dB\n")
        f.write("=" * 60 + "\n")

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

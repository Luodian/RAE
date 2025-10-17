#!/usr/bin/env python3
"""
Prepare reference set for rFID evaluation.
Pack the validation set using center_crop_arr to crop them into desired resolution.
"""
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main():
    parser = argparse.ArgumentParser(description='Prepare reference set for rFID evaluation')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageNet validation set')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path for reference NPZ file')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Target image size for center crop')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    print(f"Total images: {len(dataset)}")

    # Process all images
    all_images = []
    print("Processing images...")
    for idx in tqdm(range(len(dataset)), desc="Center cropping"):
        image, _ = dataset[idx]
        # Convert PIL to numpy array
        arr = np.array(image)
        all_images.append(arr)

    # Stack into single array
    all_images = np.stack(all_images, axis=0)
    print(f"Reference array shape: {all_images.shape}")

    # Save as NPZ
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, arr_0=all_images)
    print(f"Saved reference set to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

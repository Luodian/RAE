# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Distributed Stage-1 reconstructions on torch-xla devices.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch_xla.amp import autocast
from tqdm import tqdm


from sample_ddp import create_npz_from_sample_folder  # noqa: E402
from stage1 import RAE  # noqa: E402
from utils.train_utils import initialize_cache, set_random_seed  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import parse_configs  # noqa: E402
from tqdm import tqdm  # noqa: E402



def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    return component.replace(os.sep, "-")


def run_sampling(args: argparse.Namespace) -> None:
    initialize_cache(is_sample=False)
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    set_random_seed(args.global_seed, rank)
    if rank == 0:
        print(f"Starting rank={rank}, seed={args.global_seed}, world_size={world_size}, device={device}.")

    autocast_kwargs = dict(device=device, dtype=torch.bfloat16, enabled=args.precision == "bf16")


    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = IndexedImageFolder(args.data_path, transform=transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::max(1, world_size)]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
        f"{args.image_size}",
    ]
    sample_folder_dir = os.path.join(args.sample_dir, "-".join(folder_components))
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving reconstructed samples at {sample_folder_dir}")
    xm.rendezvous("mkdir")

    loader = DataLoader(
        subset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    local_total = len(rank_indices)
    total_batches = max(1, math.ceil(local_total / max(1, args.per_proc_batch_size)))
    parallel_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    iterator = tqdm(
        parallel_loader,
        desc="Stage1 recon",
        total=total_batches,
        disable=rank != 0,
    )

    for images, indices in iterator:
        if images.numel() == 0:
            continue
        with autocast(**autocast_kwargs):
            latents = rae.encode(images)
            recon = rae.decode(latents)
        recon = recon.clamp(0, 1)
        recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        indices_cpu = xm._maybe_convert_to_cpu(indices)
        indices_list = indices_cpu.tolist() if hasattr(indices_cpu, "tolist") else list(indices_cpu)
        for sample, idx in zip(recon_np, indices_list):
            Image.fromarray(sample).save(f"{sample_folder_dir}/{idx:06d}.png")
        xm.mark_step()

    xm.rendezvous("sampling_done")
    if rank == 0 and args.save_npz:
        create_npz_from_sample_folder(sample_folder_dir, requested)
        print("Done.")
    xm.rendezvous("done")

def _mp_worker(rank: int, args: argparse.Namespace) -> None:
    del rank
    run_sampling(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images.")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store reconstructed samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4, help="Number of images processed per device step.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to reconstruct (defaults to full dataset).")
    parser.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision mode.")
    parser.add_argument("--save-npz", action="store_true", help="Create an npz archive after sampling.")
    parsed_args = parser.parse_args()
    xmp.spawn(_mp_worker, args=(parsed_args,), start_method="fork")
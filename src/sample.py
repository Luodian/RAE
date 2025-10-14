# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Single-process sampling entrypoint for torch-xla devices.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from torchvision.utils import save_image
from omegaconf import OmegaConf

from stage1 import RAE  # noqa: E402
from stage2.models import Stage2ModelProtocol  # noqa: E402
from utils.train_utils import initialize_cache, set_random_seed  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import parse_configs  # noqa: E402
from utils.sample_utils import manual_sample, make_timesteps  # noqa: E402
from tqdm import tqdm  # noqa: E402

def parse_guidance_value(cfg: Dict[str, Any], key: str, default: float) -> float:
    if key in cfg:
        return cfg[key]
    dashed = key.replace("_", "-")
    return cfg.get(dashed, default)


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    initialize_cache(is_sample=True)

    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = int(xm.get_ordinal())
    set_random_seed(args.seed)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        _,
    ) = parse_configs(args.config)
    if rae_config is None or model_config is None:
        raise ValueError("Config must include stage_1 and stage_2 sections.")

    def to_dict(cfg_section: Optional[OmegaConf]) -> Dict[str, Any]:
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[return-value]

    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    misc_cfg = to_dict(misc_config)

    latent_size = tuple(int(dim) for dim in misc_cfg.get("latent_size", (768, 16, 16)))
    shift_dim = misc_cfg.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc_cfg.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        print(f"Using time_dist_shift={time_dist_shift:.4f}.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    rae.eval()
    model.eval()

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    if sampler_mode != "ODE":
        raise ValueError(f"Manual sampling only supports ODE mode, but got {sampler_mode}.")
    sampler_params = dict(sampler_cfg.get("params", {}))
    num_steps = int(sampler_params.get("num_steps", sampler_params.get("steps", 50)))

    num_classes = int(misc_cfg.get("num_classes", 1000))
    null_label = int(misc_cfg.get("null_label", num_classes))

    def parse_labels(arg: Optional[str]) -> list[int]:
        if arg is None:
            return [207, 360]
        labels = [label.strip() for label in arg.split(",") if label.strip()]
        if not labels:
            raise ValueError("At least one class label must be provided.")
        return [int(label) for label in labels]

    class_labels = parse_labels(args.class_labels)

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    guidance_method = guidance_cfg.get("method", "cfg")
    t_min = parse_guidance_value(guidance_cfg, "t_min", 0.0)
    t_max = parse_guidance_value(guidance_cfg, "t_max", 1.0)
    schedule = make_timesteps(num_steps, 1/1000, 1.0, time_dist_shift) 

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Guidance model config required for autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    n = len(class_labels)
    z = torch.randn(n, *latent_size, device=device)
    y = torch.tensor(class_labels, device=device, dtype=torch.long)

    sample_fn = model.forward
    model_kwargs: Optional[Dict[str, Any]] = None

    if guidance_scale > 1.0:
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward not initialized.")
            sample_fn = model.forward_with_autoguidance
            model_kwargs = {
                "cfg_scale": guidance_scale,
                "cfg_interval": (t_min, t_max),
                "additional_model_forward": guid_model_forward,
            }
        elif guidance_method == "cfg":
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((n,), null_label, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], dim=0)
            sample_fn = model.forward_with_cfg
            model_kwargs = {
                "cfg_scale": guidance_scale,
                "cfg_interval": (t_min, t_max),
            }
        else:
            raise ValueError(f"Unsupported guidance method '{guidance_method}'.")

    start_time = time()
    latents = manual_sample(
        sample_fn,
        z,
        y,
        schedule,
        device=device,
        model_kwargs=model_kwargs,
    )
    if guidance_scale > 1.0 and guidance_method == "cfg":
        latents, _ = latents.chunk(2, dim=0)

    samples = rae.decode(latents).clamp_(0, 1)
    samples_cpu = xm._maybe_convert_to_cpu(samples)

    if rank == 0:
        samples_cpu = samples_cpu.clamp(0, 1).float()
        save_image(samples_cpu, args.output, nrow=4, normalize=True, value_range=(0, 1))
        print(f"Sampling took {time() - start_time:.2f} seconds. Saved to {args.output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images from a stage-2 model on torch-xla devices.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--class-labels", type=str, default=None, help="Comma-separated class labels to sample.")
    parser.add_argument("--output", type=str, default="sample.png", help="Output image path.")
    parsed_args = parser.parse_args()
    main(parsed_args)

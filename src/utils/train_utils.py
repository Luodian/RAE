from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import os
import torch_xla
from typing import Optional
import torch
import random
import numpy as np
from torch_xla.amp import autocast
from contextlib import _GeneratorContextManager, nullcontext
def initialize_cache(is_sample: bool = True):
    CACHE_DIR = '/home/bytetriper/.cache/xla_compile/torch_sample' if is_sample else '/home/bytetriper/.cache/xla_compile/torch_train'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not torch_xla._XLAC._xla_computation_cache_is_initialized(): # only initialize once
        # TODO: add a lock to prevent multiple processes from initializing the cache
        xr.initialize_cache(CACHE_DIR, readonly=False)

def get_autocast_context(enabled: bool = True) -> autocast:
    return autocast(enabled=enabled, device = xm.xla_device())

def set_random_seed(seed: Optional[int] = None, rank: Optional[int] = None):
    if seed is None:
        seed = 114514
    if rank is None:
        # try get rank via xla
        rank = int(xm.get_ordinal())
    seed = seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    xm.set_rng_state(seed)


def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config


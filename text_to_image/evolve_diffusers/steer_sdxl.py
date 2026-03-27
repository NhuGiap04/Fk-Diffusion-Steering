from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import torch
from diffusers.image_processor import PipelineImageInput
from .rewards import get_reward_function
from .pipeline_sdxl import (
    BaseSDXL,
    get_scheduler_sigmas_for_timesteps,
    latent_to_decode,
    rescale_noise_cfg,
    XLA_AVAILABLE,
)

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm



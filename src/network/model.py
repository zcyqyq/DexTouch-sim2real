import torch

from src.network.graspness_sample import GraspnessSample
from src.network.diffusion_sample import DiffusionSample

def get_model(config: dict):
    """
        get model by config
    """
    if config.type in ['graspness_isa', 'graspness_diffusion', 'graspness_cvae']:
        return GraspnessSample(config)
    elif config.type in ['glob_diff']:
        return DiffusionSample(config)
    raise NotImplementedError()
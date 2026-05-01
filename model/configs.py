import os
import torch

from dataclasses import dataclass
from helpers import get_default_device, get_most_recent_version

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Flowers"  # "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    working_dir = os.getcwd()
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join(working_dir, "Log", "Inference")
    root_checkpoint_dir = os.path.join("Log", "checkpoints")

    # Current log and checkpoint directory.
    recent_version = str(get_most_recent_version(root_checkpoint_dir))
    log_dir = os.path.join(root_log_dir, recent_version)
    checkpoint_dir = os.path.join(root_checkpoint_dir, f'version_{recent_version}/ckpt.tar')

@dataclass
class TrainingConfig:
    TIMESTEPS = 200
    IMG_SHAPE = (3, 64, 64)
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    LR = 5e-5
    NUM_WORKERS = 0

    SCHEDULE_TYPE = 'cosine'  # 'linear', 'cosine'

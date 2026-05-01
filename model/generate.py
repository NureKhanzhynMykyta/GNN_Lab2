import torch
import os
from datetime import datetime

from configs import BaseConfig, TrainingConfig
from train import eps_model, dd, reverse_diffusion

checkpoint = torch.load(BaseConfig.checkpoint_dir, map_location=BaseConfig.DEVICE)
eps_model.load_state_dict(checkpoint['model'])
eps_model.to(BaseConfig.DEVICE)

log_dir = "inference_results"
os.makedirs(log_dir, exist_ok=True)
 

def sample(generate_video=True, num_images=256, timesteps=1000, nrow=32):
    # generated_video = True for generating video of the entire reverse diffusion proces or False to for saving only the final generated image.
    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
    save_path = os.path.join(log_dir, filename)
    
    reverse_diffusion(
        eps_model,
        dd,
        num_images=num_images,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=timesteps,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=nrow
    )
    
    print(save_path)


if __name__ == '__main__':
    sample()
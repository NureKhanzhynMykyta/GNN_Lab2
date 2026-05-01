import os
import gc
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as TF
from IPython.display import display
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from torch.cuda import amp
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from configs import BaseConfig, TrainingConfig
from dataloader import get_dataloader, inverse_transform
from helpers import get, frames2vid, setup_log_directory
from diffusion import DenoiseDiffusion


def train_one_epoch(model, dd, loader, optimizer, scaler, loss_fn, epoch=800,
                   base_config=BaseConfig(), training_config=TrainingConfig()):

    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s, _ in loader:
            tq.update(1)
            x0s = x0s.to(base_config.DEVICE)

            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = dd.q_sample(x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss


@torch.inference_mode()
def reverse_diffusion(model, dd, timesteps=1000, img_shape=(3, 64, 64),
                      num_images=5, nrow=8, device=BaseConfig.DEVICE, **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    save_path = BaseConfig.working_dir + "/sample.png"

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps-1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(dd.beta, ts)
        one_by_sqrt_alpha_t               = get(dd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(dd.sqrt_one_minus_alpha_cumulative, ts)

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process.
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None

    else: # Display and save the image at the final timestep of the reverse process.
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        display(pil_image)
        return None


class ModelConfig:
    N_CH = 32
    BASE_CH_MULT = (1, 2, 4)
    APPLY_ATTENTION = (False, False, False)
    N_BLOCKS = 1

eps_model = UNet(
    input_channels = TrainingConfig.IMG_SHAPE[0],
    n_channels = ModelConfig.N_CH,
    ch_mults = ModelConfig.BASE_CH_MULT,
    is_attn = ModelConfig.APPLY_ATTENTION,
    n_blocks = ModelConfig.N_BLOCKS
)

eps_model.to(BaseConfig.DEVICE)

dd = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=TrainingConfig.TIMESTEPS,
        device=BaseConfig.DEVICE,
        schedule_type=TrainingConfig.SCHEDULE_TYPE
    )

if __name__ == '__main__':
    optimizer = torch.optim.AdamW(eps_model.parameters(), lr=TrainingConfig.LR)
    dataloader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=TrainingConfig.BATCH_SIZE,
        device=BaseConfig.DEVICE,
        pin_memory=True,
        num_workers=TrainingConfig.NUM_WORKERS,
    )
    loss_fn = nn.MSELoss()
    scaler = amp.GradScaler()

    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    print(f"Generating Forward Diffusion ({TrainingConfig.SCHEDULE_TYPE})...")
    x0_sim, _ = next(iter(dataloader))
    x0_sim = x0_sim[:1].to(BaseConfig.DEVICE)
    viz_steps = [0, 50, 100, 150, 199]
    plt.figure(figsize=(15, 3))
    for i, s in enumerate(viz_steps):
        t_sim = torch.tensor([s], device=BaseConfig.DEVICE)
        xt_sim, _ = dd.q_sample(x0_sim, t_sim)

        plt.subplot(1, 5, i + 1)
        img_to_show = inverse_transform(xt_sim).cpu().squeeze(0).permute(1, 2, 0).numpy().astype('uint8')
        plt.imshow(img_to_show)

        plt.axis('off')
        plt.title(f"Step {s}")
    plt.savefig(os.path.join(log_dir, f"forward_diffusion_{TrainingConfig.SCHEDULE_TYPE}.png"))
    plt.close()

    # print("Generating Forward Diffusion visualization...")
    # x0_sim, _ = next(iter(dataloader))
    # x0_sim = x0_sim[:1].to(BaseConfig.DEVICE)
    # viz_steps = [0, 25, 50, 75, 99]
    # plt.figure(figsize=(15, 3))
    # for i, s in enumerate(viz_steps):
    #     t_sim = torch.tensor([s], device=BaseConfig.DEVICE)
    #     xt_sim, _ = dd.q_sample(x0_sim, t_sim)
    #
    #     plt.subplot(1, 5, i + 1)
    #     img_to_show = inverse_transform(xt_sim).cpu().squeeze()
    #     plt.imshow(img_to_show, cmap='gray')
    #
    #     plt.axis('off')
    #     plt.title(f"Step {s}")
    # plt.savefig(os.path.join(log_dir, "forward_diffusion_process.png"))
    # plt.close()

    loss_history = []

    for epoch in range(1, TrainingConfig.NUM_EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        epoch_loss = train_one_epoch(eps_model, dd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)
        loss_history.append(epoch_loss)

        if epoch == TrainingConfig.NUM_EPOCHS or epoch % 5 == 0:
            save_path = os.path.join(log_dir, f"epoch_{epoch}_grid.png")
            reverse_diffusion(eps_model, dd, timesteps=TrainingConfig.TIMESTEPS,
                              num_images=16, nrow=4, save_path=save_path,
                              img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

            checkpoint_dict = {
                "epoch": epoch,
                "model": eps_model.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Convergence")
    plt.savefig(os.path.join(log_dir, "loss_graph.png"))
    print(f"Training finished")

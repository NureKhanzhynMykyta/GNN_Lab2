from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from model.configs import BaseConfig
from model.dataloader import get_dataloader, inverse_transform


loader = get_dataloader(
    dataset_name=BaseConfig.DATASET,
    batch_size=128,
    device=BaseConfig.DEVICE,
)

plt.figure(figsize=(12, 6), facecolor='white')

for b_image, _ in loader:
    b_image = inverse_transform(b_image).cpu()
    grid_img = make_grid(b_image / 255.0, nrow=16, padding=True, pad_value=1, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    
plt.savefig("data.png")

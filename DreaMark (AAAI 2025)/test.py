import os
import hashlib
import torch
import torchvision
from torchvision.transforms import functional
import random
import numpy as np
import torchvision.utils
import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler

from nerf.utils_img import gaussian_noise

_img2mse = lambda x, y : torch.mean((x - y) ** 2)
_mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def decode_latents(latents):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents).sample
    return image

if __name__ == "__main__":
    img = torchvision.io.read_image("exp-test/2024-08-14-a-DSLR-photo-of-peacock.-16-1000-scale-7.5-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-finetune-tet-256/0.png") / 255
    img2 = torchvision.io.read_image("exp-test/2024-08-14-a-DSLR-photo-of-peacock.-16-1000-scale-7.5-lr-0.001-albedo-le-10-render-512-cube-sd-2.1-5000-finetune-tet-256/brightness.png") / 255
    img_mse = _img2mse(img, img2)
    psnr = _mse2psnr(img_mse)
    # torchvision.utils.save_image(diff, "diff.png")
    # print(diff)
    print(psnr)
    exit()
    seed_everything(42)
    img = torchvision.io.read_image("000000003398.jpg")
    img = img/255
    print(img.max(), img.min())
    img = gaussian_noise(img, 0.3)
    torchvision.utils.save_image(img, "tmp.png")
    exit()
    
    model_key = "stabilityai/stable-diffusion-2-1-base"
    _dtype = torch.float16
    device = torch.device(f"cuda:5")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_key, local_files_only=True, torch_dtype=_dtype
    ).to(device)
    latents = pipe("a DSLR photo of a peacock", output_type="latent").images
    target = torch.randn_like(latents)

    ori_image = decode_latents(latents).detach()
    latents.requires_grad = True
    opt = torch.optim.Adam([latents], lr=1e-3)
    progress = tqdm.tqdm(range(1000))
    for it in progress:
        opt.zero_grad()
        image = decode_latents(latents)
        loss = (image - ori_image).abs().mean() + (target - latents).abs().mean()
        loss.backward()
        opt.step()
        progress.set_postfix_str(
            f"loss: {loss.item()}, latent_diff: {(target - latents).abs().mean().item()}, image_diff: {(image - ori_image).abs().mean().item()}"
        )
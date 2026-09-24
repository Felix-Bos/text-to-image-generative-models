"""
Compare visuellement DDPM et Flow Matching sur les mêmes prompts texte,
à partir de checkpoints entraînés des deux approches.

Usage (via la CLI unifiée, voir src/cli.py) :
    python -m src.cli compare \\
        --ddpm-checkpoint checkpoints/diffusion/latest.pt \\
        --flow-checkpoint checkpoints/flow_matching/latest.pt \\
        --prompt "a smiling woman with bangs" --prompt "a man with a beard"

Les deux checkpoints doivent avoir été entraînés avec la même architecture
U-Net (base_channels, channel_multipliers) pour pouvoir être chargés — c'est
le cas par défaut puisque les deux train.py utilisent la même config.
"""

from __future__ import annotations

import torch

from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion
from src.backend.diffusion.sample import generate as sample_ddpm
from src.backend.flow_matching.sample import generate as sample_flow_matching
from src.backend.shared.device import get_device
from src.backend.shared.preview import save_preview_grid
from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
from src.backend.shared.unet import UNet

DEFAULT_PROMPTS = [
    "a young woman smiling with bangs",
    "a man with a beard and no smile",
    "an elderly person wearing eyeglasses",
    "a smiling young man with no beard",
]


def _load_unet(checkpoint_path: str, device: torch.device) -> UNet:
    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    unet.eval()
    return unet


def compare(
    ddpm_checkpoint: str,
    flow_checkpoint: str,
    prompts: list[str],
    ddpm_timesteps: int = 1000,
    flow_num_steps: int = 50,
    out_path: str = "report/comparison_ddpm_vs_flow_matching.png",
    device: str | torch.device | None = None,
) -> None:
    device = torch.device(device) if device is not None else get_device()
    print(f"Device: {device}")

    text_encoder = FrozenCLIPTextEncoder(device=device)
    text_embed = text_encoder(prompts)

    print(f"\nGénération DDPM ({ddpm_timesteps} étapes)...")
    ddpm_unet = _load_unet(ddpm_checkpoint, device)
    diffusion = GaussianDiffusion(timesteps=ddpm_timesteps, device=device)
    ddpm_images = sample_ddpm(ddpm_unet, diffusion, text_embed, device=device)

    print(f"\nGénération Flow Matching ({flow_num_steps} étapes)...")
    flow_unet = _load_unet(flow_checkpoint, device)
    flow_images = sample_flow_matching(flow_unet, text_embed, num_steps=flow_num_steps, device=device)

    # On sauvegarde deux grilles séparées (une par approche) plutôt qu'une
    # combinée : plus simple à lire, et évite de dupliquer save_preview_grid
    # pour gérer 2 lignes.
    ddpm_out = out_path.replace(".png", "_ddpm.png")
    flow_out = out_path.replace(".png", "_flow_matching.png")

    save_preview_grid(ddpm_images, prompts, ddpm_out, title=f"DDPM ({ddpm_timesteps} steps)")
    save_preview_grid(flow_images, prompts, flow_out, title=f"Flow Matching ({flow_num_steps} steps)")

    print(f"\nComparaison sauvegardée dans:\n  {ddpm_out}\n  {flow_out}")

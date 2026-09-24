"""
Training loop pour le Flow Matching text-to-image.

Miroir de src/backend/diffusion/train.py : même dataset, même U-Net (partagé
via shared/unet.py), même text encoder CLIP gelé. Seule la partie spécifique
au flow matching change :

    1. On prend un batch d'images + captions réelles.
    2. On tire un t aléatoire CONTINU dans [0, 1] par image du batch
       (contre un t entier dans [0, T) pour le DDPM).
    3. On interpole x_t = (1-t)*x_0 + t*noise (ConditionalFlowMatching.sample_xt)
       -> x_t, velocity_cible = noise - x_0.
    4. On encode les captions avec CLIP (gelé) -> text_embed.
    5. Le U-Net prédit la vélocité : v_pred = UNet(x_t, t, text_embed).
    6. loss = MSE(v_pred, velocity_cible) ; backward ; optimizer.step().

Le classifier-free guidance, le système de preview/resume et le logging
TensorBoard sont identiques au DDPM (voir diffusion/train.py pour les
commentaires détaillés).

Ce module n'a pas de CLI propre : il est appelé par src/cli.py (`python -m
src.cli train flow_matching ...`).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.backend.flow_matching.flow_matching import ConditionalFlowMatching
from src.backend.flow_matching.sample import generate as sample_flow_matching
from src.backend.shared.dataset import CelebACaptionDataset
from src.backend.shared.device import get_device
from src.backend.shared.preview import save_preview_grid
from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
from src.backend.shared.unet import UNet

# Mêmes captions que diffusion/train.py, pour pouvoir comparer les deux
# approches sur exactement les mêmes prompts (voir compare.py).
PREVIEW_CAPTIONS = [
    "a young woman smiling with bangs",
    "a man with a beard and no smile",
    "an elderly person wearing eyeglasses",
    "a smiling young man with no beard",
]


def train(
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-4,
    image_size: int = 64,
    cfg_dropout_prob: float = 0.1,
    limit_batches: int | None = None,
    checkpoint_dir: str = "checkpoints/flow_matching",
    checkpoint_every_steps: int = 500,
    preview_every_steps: int = 500,
    preview_num_steps: int = 50,
    log_every_steps: int = 50,
    resume: str | None = None,
    device: str | torch.device | None = None,
) -> None:
    """Entraîne le U-Net du modèle Flow Matching sur CelebA-Dialog.

    Args mêmes rôles que src/backend/diffusion/train.py (pas de `timesteps`
    ici : t est continu, pas besoin de discrétiser pour l'entraînement).
    `preview_num_steps` est le nombre de pas du solveur ODE utilisé pour les
    previews (indépendant de l'entraînement, qui n'utilise pas de solveur).
    """
    device = torch.device(device) if device is not None else get_device()
    print(f"Device: {device}")

    dataset = CelebACaptionDataset(split="train", image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

    flow = ConditionalFlowMatching(device=device)
    text_encoder = FrozenCLIPTextEncoder(device=device)
    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"U-Net: {n_params:,} paramètres entraînables")

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    preview_path = checkpoint_path / "previews"

    writer = SummaryWriter(log_dir=str(checkpoint_path / "tensorboard"))
    writer.add_text(
        "config",
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, image_size={image_size}, "
        f"cfg_dropout_prob={cfg_dropout_prob}, params={n_params:,}",
    )

    global_step = 0
    start_epoch = 0
    if resume is not None:
        global_step, start_epoch = _load_checkpoint(unet, optimizer, resume)
        print(f"Reprise depuis {resume} : step={global_step}, epoch={start_epoch + 1}")

    preview_text_embed = text_encoder(PREVIEW_CAPTIONS)

    unet.train()

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            running_loss = 0.0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, (images, captions) in enumerate(progress):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break

                images = images.to(device)

                captions = [
                    "" if torch.rand(1).item() < cfg_dropout_prob else c for c in captions
                ]

                # t continu dans [0, 1], contrairement au t entier discret du DDPM.
                t = torch.rand(images.shape[0], device=device)
                x_t, _noise, velocity = flow.sample_xt(images, t)

                text_embed = text_encoder(captions)
                v_pred = unet(x_t, t, text_embed)

                loss = torch.nn.functional.mse_loss(v_pred, velocity)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % log_every_steps == 0:
                    avg_loss = running_loss / log_every_steps
                    progress.set_postfix(loss=f"{avg_loss:.4f}")
                    writer.add_scalar("loss/train", avg_loss, global_step)
                    running_loss = 0.0

                if global_step % checkpoint_every_steps == 0:
                    _save_checkpoint(unet, optimizer, global_step, epoch, checkpoint_path / "latest.pt")

                if global_step % preview_every_steps == 0:
                    _run_preview(
                        unet,
                        preview_text_embed,
                        preview_path,
                        global_step,
                        preview_num_steps,
                        device,
                        writer,
                    )
                    unet.train()  # _run_preview met le modèle en eval()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{epochs} terminée en {epoch_time / 60:.1f} min")
            writer.add_scalar("time/epoch_minutes", epoch_time / 60, epoch)
            _save_checkpoint(unet, optimizer, global_step, epoch, checkpoint_path / f"epoch_{epoch + 1}.pt")

        print("Entraînement terminé.")
    finally:
        writer.close()


@torch.no_grad()
def _run_preview(
    unet: torch.nn.Module,
    text_embed: torch.Tensor,
    preview_path: Path,
    step: int,
    num_steps: int,
    device: torch.device,
    writer: SummaryWriter,
) -> None:
    images = sample_flow_matching(
        unet, text_embed, num_steps=num_steps, device=device, show_progress=False
    )
    save_preview_grid(
        images,
        PREVIEW_CAPTIONS,
        preview_path / f"step_{step:07d}.png",
        title=f"Flow Matching — step {step}",
    )
    images_01 = (images.clamp(-1, 1) + 1) / 2
    writer.add_images("preview/samples", images_01, global_step=step)


def _save_checkpoint(
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(
    unet: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=next(unet.parameters()).device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint.get("epoch", 0)

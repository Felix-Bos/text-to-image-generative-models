"""
API FastAPI servant la génération d'images text-to-image (DDPM et Flow
Matching) à un frontend web (voir src/frontend/web/).

Lancement :
    uvicorn src.frontend.api.main:app --reload --port 8000

Les modèles (CLIP + U-Net par approche) sont chargés paresseusement, au
premier appel de /generate pour l'approche demandée, pas au démarrage du
serveur — pas besoin d'avoir entraîné les deux approches pour démarrer l'API,
et le démarrage du serveur reste instantané.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.backend.shared.device import get_device
from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
from src.backend.shared.unet import UNet

app = FastAPI(title="Text-to-Image (DDPM vs Flow Matching)", version="0.1.0")

# Autorise le frontend React (servi séparément, ex: Vite sur localhost:5173)
# à appeler cette API depuis le navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_CHECKPOINTS = {
    "diffusion": "checkpoints/diffusion/latest.pt",
    "flow_matching": "checkpoints/flow_matching/latest.pt",
}


class _ModelCache:
    """Cache paresseux : évite de recharger CLIP/U-Net à chaque requête."""

    def __init__(self) -> None:
        self.device = get_device()
        self.text_encoder: FrozenCLIPTextEncoder | None = None
        self.unets: dict[str, UNet] = {}
        self.unet_checkpoint_paths: dict[str, str] = {}

    def get_text_encoder(self) -> FrozenCLIPTextEncoder:
        if self.text_encoder is None:
            self.text_encoder = FrozenCLIPTextEncoder(device=self.device)
        return self.text_encoder

    def get_unet(self, approach: str, checkpoint_path: str) -> UNet:
        # Recharge seulement si le checkpoint demandé diffère de celui en cache.
        if self.unets.get(approach) is None or self.unet_checkpoint_paths.get(approach) != checkpoint_path:
            if not Path(checkpoint_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Checkpoint introuvable: {checkpoint_path}. "
                        f"Entraînez d'abord le modèle ({approach}) via la CLI "
                        f"(voir README.md, section CLI)."
                    ),
                )
            unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            unet.load_state_dict(checkpoint["unet_state_dict"])
            unet.eval()
            self.unets[approach] = unet
            self.unet_checkpoint_paths[approach] = checkpoint_path
        return self.unets[approach]


_cache = _ModelCache()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Description textuelle de l'image à générer")
    approach: str = Field(..., pattern="^(diffusion|flow_matching)$")
    checkpoint: str | None = Field(
        None, description="Chemin vers un checkpoint .pt spécifique (sinon checkpoints/<approach>/latest.pt)"
    )
    image_size: int = Field(64, ge=32, le=256)
    num_steps: int = Field(
        50, ge=1, le=1000, description="Pas de sampling (timesteps pour diffusion, pas ODE pour flow matching)"
    )
    seed: int | None = Field(None, description="Graine aléatoire, pour une génération reproductible")


class GenerateResponse(BaseModel):
    image_base64: str
    approach: str
    prompt: str
    num_steps: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(_cache.device)}


@app.get("/status")
def status() -> dict:
    """Indique quels checkpoints par défaut sont disponibles sur disque."""
    return {
        approach: Path(path).exists() for approach, path in DEFAULT_CHECKPOINTS.items()
    }


class TrainingStatusResponse(BaseModel):
    running: bool
    latest_step: int | None = None
    loss_points: list[tuple[int, float]] = []
    latest_preview_base64: str | None = None
    latest_preview_step: int | None = None


@app.get("/training-status", response_model=TrainingStatusResponse)
def training_status(approach: str, checkpoint_dir: str | None = None) -> TrainingStatusResponse:
    """Lit les logs TensorBoard et la dernière preview d'un entraînement en
    cours (ou terminé), pour alimenter la page de monitoring live du frontend.

    Ne nécessite pas que le training loop tourne dans ce process : lit
    directement les fichiers sur disque (logs TensorBoard + previews/*.png)
    écrits par `src.backend.<approach>.train`, donc fonctionne même si
    l'entraînement tourne dans un terminal séparé.
    """
    if approach not in DEFAULT_CHECKPOINTS:
        raise HTTPException(status_code=400, detail=f"approach invalide: {approach}")

    base_dir = Path(checkpoint_dir or f"checkpoints/{approach}")
    tensorboard_dir = base_dir / "tensorboard"
    previews_dir = base_dir / "previews"

    if not tensorboard_dir.exists():
        return TrainingStatusResponse(running=False)

    loss_points = _read_loss_scalars(tensorboard_dir)
    latest_step = loss_points[-1][0] if loss_points else None

    latest_preview_base64 = None
    latest_preview_step = None
    preview_files = sorted(previews_dir.glob("step_*.png")) if previews_dir.exists() else []
    if preview_files:
        latest_file = preview_files[-1]
        latest_preview_step = int(latest_file.stem.split("_")[1])
        latest_preview_base64 = base64.b64encode(latest_file.read_bytes()).decode("utf-8")

    return TrainingStatusResponse(
        running=True,
        latest_step=latest_step,
        loss_points=loss_points,
        latest_preview_base64=latest_preview_base64,
        latest_preview_step=latest_preview_step,
    )


def _read_loss_scalars(tensorboard_dir: Path, max_points: int = 300) -> list[tuple[int, float]]:
    """Lit le scalaire `loss/train` depuis TOUS les fichiers d'events du
    dossier (un entraînement repris via --resume écrit plusieurs fichiers),
    fusionnés et triés par step. Sous-échantillonne à `max_points` pour ne
    pas envoyer des dizaines de milliers de points au frontend.
    """
    all_points: list[tuple[int, float]] = []
    for event_file_dir in [tensorboard_dir]:
        try:
            ea = EventAccumulator(str(event_file_dir), size_guidance={"scalars": 0})
            ea.Reload()
            if "loss/train" in ea.Tags().get("scalars", []):
                for scalar_event in ea.Scalars("loss/train"):
                    all_points.append((scalar_event.step, scalar_event.value))
        except Exception:
            # Fichier d'events corrompu/en cours d'écriture : on ignore plutôt que 500.
            pass

    all_points.sort(key=lambda p: p[0])

    if len(all_points) > max_points:
        stride = len(all_points) // max_points
        all_points = all_points[::stride]

    return all_points


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    checkpoint_path = request.checkpoint or DEFAULT_CHECKPOINTS[request.approach]
    unet = _cache.get_unet(request.approach, checkpoint_path)
    text_encoder = _cache.get_text_encoder()

    if request.seed is not None:
        torch.manual_seed(request.seed)

    text_embed = text_encoder([request.prompt])

    if request.approach == "diffusion":
        from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion
        from src.backend.diffusion.sample import generate as sample_diffusion

        diffusion = GaussianDiffusion(timesteps=request.num_steps, device=_cache.device)
        images = sample_diffusion(
            unet,
            diffusion,
            text_embed,
            image_size=request.image_size,
            device=_cache.device,
            show_progress=False,
        )
    else:
        from src.backend.flow_matching.sample import generate as sample_flow_matching

        images = sample_flow_matching(
            unet,
            text_embed,
            image_size=request.image_size,
            num_steps=request.num_steps,
            device=_cache.device,
            show_progress=False,
        )

    image_base64 = _tensor_to_base64_png(images[0])

    return GenerateResponse(
        image_base64=image_base64,
        approach=request.approach,
        prompt=request.prompt,
        num_steps=request.num_steps,
    )


def _tensor_to_base64_png(image_tensor: torch.Tensor) -> str:
    """Convertit un tenseur image (C, H, W) dans [-1, 1] en PNG encodé base64."""
    image_01 = (image_tensor.clamp(-1, 1) + 1) / 2
    image_np = (image_01.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil_image = Image.fromarray(image_np)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

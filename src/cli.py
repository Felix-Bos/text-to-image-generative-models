"""
CLI unifiée du projet : entraînement, génération et comparaison, pour les
deux approches (DDPM / diffusion et flow matching).

Usage général :
    python -m src.cli <commande> <sous-commande> [options]

Commandes disponibles (voir aussi `README.md`) :

    train diffusion       Entraîne le DDPM.
    train flow_matching   Entraîne le Flow Matching.
    sample diffusion      Génère des images avec un checkpoint DDPM.
    sample flow_matching  Génère des images avec un checkpoint Flow Matching.
    compare                Génère les mêmes prompts avec les deux approches, côte à côte.

Exemples :
    python -m src.cli train diffusion --epochs 5 --batch-size 32
    python -m src.cli train flow_matching --epochs 5 --resume checkpoints/flow_matching/latest.pt
    python -m src.cli sample diffusion --checkpoint checkpoints/diffusion/latest.pt \\
        --prompt "a smiling woman with bangs" --out report/my_sample.png
    python -m src.cli compare --ddpm-checkpoint checkpoints/diffusion/latest.pt \\
        --flow-checkpoint checkpoints/flow_matching/latest.pt

Suivi TensorBoard (pendant ou après l'entraînement) :
    tensorboard --logdir checkpoints/diffusion/tensorboard
    tensorboard --logdir checkpoints/flow_matching/tensorboard
"""

from __future__ import annotations

import argparse

import torch


def _add_train_diffusion_parser(subparsers) -> None:
    p = subparsers.add_parser("diffusion", help="Entraîne le DDPM")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000, help="Nombre d'étapes T du schedule de bruit")
    p.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Limite le nombre de batches par epoch (smoke test rapide)",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/diffusion")
    p.add_argument("--checkpoint-every-steps", type=int, default=500)
    p.add_argument("--preview-every-steps", type=int, default=500)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--resume", type=str, default=None, help="Chemin vers un checkpoint .pt à reprendre")
    p.add_argument("--device", type=str, default=None, help="mps / cuda / cpu (auto-détecté sinon)")


def _add_train_flow_matching_parser(subparsers) -> None:
    p = subparsers.add_parser("flow_matching", help="Entraîne le Flow Matching")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Limite le nombre de batches par epoch (smoke test rapide)",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/flow_matching")
    p.add_argument("--checkpoint-every-steps", type=int, default=500)
    p.add_argument("--preview-every-steps", type=int, default=500)
    p.add_argument("--preview-num-steps", type=int, default=50, help="Pas du solveur ODE pour les previews")
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--resume", type=str, default=None, help="Chemin vers un checkpoint .pt à reprendre")
    p.add_argument("--device", type=str, default=None, help="mps / cuda / cpu (auto-détecté sinon)")


def _cmd_train(args: argparse.Namespace) -> None:
    if args.approach == "diffusion":
        from src.backend.diffusion.train import train as train_diffusion

        train_diffusion(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            timesteps=args.timesteps,
            cfg_dropout_prob=args.cfg_dropout_prob,
            limit_batches=args.limit_batches,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            preview_every_steps=args.preview_every_steps,
            log_every_steps=args.log_every_steps,
            resume=args.resume,
            device=args.device,
        )
    elif args.approach == "flow_matching":
        from src.backend.flow_matching.train import train as train_flow_matching

        train_flow_matching(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            cfg_dropout_prob=args.cfg_dropout_prob,
            limit_batches=args.limit_batches,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            preview_every_steps=args.preview_every_steps,
            preview_num_steps=args.preview_num_steps,
            log_every_steps=args.log_every_steps,
            resume=args.resume,
            device=args.device,
        )


def _add_sample_diffusion_parser(subparsers) -> None:
    p = subparsers.add_parser("diffusion", help="Génère des images avec un checkpoint DDPM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, action="append", required=True, help="Répétable")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--out", type=str, default="report/sample_diffusion.png")
    p.add_argument("--device", type=str, default=None)


def _add_sample_flow_matching_parser(subparsers) -> None:
    p = subparsers.add_parser("flow_matching", help="Génère des images avec un checkpoint Flow Matching")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, action="append", required=True, help="Répétable")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=50, help="Pas du solveur ODE")
    p.add_argument("--out", type=str, default="report/sample_flow_matching.png")
    p.add_argument("--device", type=str, default=None)


def _cmd_sample(args: argparse.Namespace) -> None:
    from src.backend.shared.device import get_device
    from src.backend.shared.preview import save_preview_grid
    from src.backend.shared.text_encoder import FrozenCLIPTextEncoder
    from src.backend.shared.unet import UNet

    device = torch.device(args.device) if args.device is not None else get_device()

    unet = UNet(base_channels=64, channel_multipliers=(1, 2, 4, 4)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    unet.eval()

    text_encoder = FrozenCLIPTextEncoder(device=device)
    text_embed = text_encoder(args.prompt)

    if args.approach == "diffusion":
        from src.backend.diffusion.gaussian_diffusion import GaussianDiffusion
        from src.backend.diffusion.sample import generate

        diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
        images = generate(unet, diffusion, text_embed, image_size=args.image_size, device=device)
        title = f"DDPM ({args.timesteps} steps)"
    else:
        from src.backend.flow_matching.sample import generate

        images = generate(
            unet, text_embed, image_size=args.image_size, num_steps=args.num_steps, device=device
        )
        title = f"Flow Matching ({args.num_steps} steps)"

    save_preview_grid(images, args.prompt, args.out, title=title)
    print(f"Images sauvegardées dans {args.out}")


def _add_compare_parser(subparsers) -> None:
    p = subparsers.add_parser("compare", help="Compare DDPM et Flow Matching sur les mêmes prompts")
    p.add_argument("--ddpm-checkpoint", type=str, required=True)
    p.add_argument("--flow-checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, action="append", default=None, help="Répétable")
    p.add_argument("--ddpm-timesteps", type=int, default=1000)
    p.add_argument("--flow-num-steps", type=int, default=50)
    p.add_argument("--out-path", type=str, default="report/comparison_ddpm_vs_flow_matching.png")
    p.add_argument("--device", type=str, default=None)


def _cmd_compare(args: argparse.Namespace) -> None:
    from src.backend.compare import DEFAULT_PROMPTS, compare

    compare(
        ddpm_checkpoint=args.ddpm_checkpoint,
        flow_checkpoint=args.flow_checkpoint,
        prompts=args.prompt or DEFAULT_PROMPTS,
        ddpm_timesteps=args.ddpm_timesteps,
        flow_num_steps=args.flow_num_steps,
        out_path=args.out_path,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Entraînement, génération et comparaison DDPM vs Flow Matching (CelebA-Dialog)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entraîne un modèle")
    train_subparsers = train_parser.add_subparsers(dest="approach", required=True)
    _add_train_diffusion_parser(train_subparsers)
    _add_train_flow_matching_parser(train_subparsers)

    sample_parser = subparsers.add_parser("sample", help="Génère des images à partir d'un checkpoint")
    sample_subparsers = sample_parser.add_subparsers(dest="approach", required=True)
    _add_sample_diffusion_parser(sample_subparsers)
    _add_sample_flow_matching_parser(sample_subparsers)

    _add_compare_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "sample":
        _cmd_sample(args)
    elif args.command == "compare":
        _cmd_compare(args)


if __name__ == "__main__":
    main()

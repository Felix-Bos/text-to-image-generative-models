"""
Dataset PyTorch pour CelebA-Dialog (images + captions texte).

Ce module ne fait qu'une chose : charger une paire (image, texte) prête à
l'emploi pour un modèle de diffusion text-to-image. Il ne connaît rien au
bruitage / au modèle — c'est la couche la plus basse du pipeline.

Structure attendue sous `data/` (voir data/README.txt) :
    data/img_align_celeba/000001.jpg, 000002.jpg, ...
    data/celeba_caption/captions.json      -> {"000001.jpg": {"overall_caption": "...", ...}, ...}
    data/Eval/list_eval_partition.txt      -> "000001.jpg 0"  (0=train, 1=val, 2=test)

202 599 images au total, mais seulement 202 578 ont une caption (21 images
non annotées) : on les exclut automatiquement.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Partition officielle CelebA : cf. data/Eval/list_eval_partition.txt
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


class CelebACaptionDataset(Dataset):
    """Associe chaque image CelebA à sa légende (`overall_caption`).

    Args:
        data_dir: racine du dossier `data/` (contient img_align_celeba/, celeba_caption/, Eval/).
        split: "train", "val" ou "test" (suit le split officiel CelebA).
        image_size: taille (carrée) à laquelle les images sont redimensionnées.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        split: str = "train",
        image_size: int = 64,
    ) -> None:
        if split not in SPLIT_TO_CODE:
            raise ValueError(f"split doit être 'train', 'val' ou 'test', reçu {split!r}")

        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "img_align_celeba"
        self.split = split
        self.image_size = image_size

        self.captions: dict[str, str] = self._load_captions()
        self.image_names: list[str] = self._load_split(split)

        # Transform standard pour un DDPM :
        #   - resize + crop carré pour ne pas déformer les visages
        #   - passage en tenseur [0, 1]
        #   - normalisation en [-1, 1] (le bruit gaussien ajouté pendant la
        #     diffusion est centré en 0 ; on veut que l'image le soit aussi)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_captions(self) -> dict[str, str]:
        captions_path = self.data_dir / "celeba_caption" / "captions.json"
        with open(captions_path, "r") as f:
            raw = json.load(f)
        # On ne garde que la légende globale (overall_caption), pas le détail par attribut.
        return {name: entry["overall_caption"] for name, entry in raw.items()}

    def _load_split(self, split: str) -> list[str]:
        partition_path = self.data_dir / "Eval" / "list_eval_partition.txt"
        target_code = SPLIT_TO_CODE[split]

        names: list[str] = []
        with open(partition_path, "r") as f:
            for line in f:
                name, code = line.split()
                if int(code) != target_code:
                    continue
                # Exclut les images sans caption (21 images non annotées au total).
                if name not in self.captions:
                    continue
                names.append(name)
        return names

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        image = Image.open(self.image_dir / name).convert("RGB")
        image = self.transform(image)
        caption = self.captions[name]
        return image, caption


if __name__ == "__main__":
    # Petit auto-test manuel : `python src/dataset.py`
    # Vérifie que le dataset se charge et qu'un échantillon a la bonne forme.
    ds = CelebACaptionDataset(split="train", image_size=64)
    print(f"Nombre d'images (train): {len(ds)}")

    image, caption = ds[0]
    print(f"Image shape: {tuple(image.shape)}, dtype: {image.dtype}")
    print(f"Valeurs min/max: {image.min().item():.3f} / {image.max().item():.3f}")
    print(f"Caption: {caption!r}")

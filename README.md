# Text-to-Image Generative Models — DDPM vs Flow Matching

Implémentation **from scratch** en PyTorch de deux approches de génération d'images conditionnées par texte — un DDPM (diffusion) et un modèle de Flow Matching — entraînées sur [CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog) (visages + descriptions en langage naturel), **entièrement en local sur Apple Silicon (MPS)**.

Les deux approches partagent la même architecture U-Net et le même encodeur de texte (CLIP, gelé) : seule la façon d'entraîner et de générer diffère (voir [Architecture](#architecture)).

## Sommaire

- [Installation](#installation)
- [Préparation des données](#préparation-des-données)
- [Architecture](#architecture)
- [Utilisation de la CLI](#utilisation-de-la-cli)
  - [Entraînement](#entraînement)
  - [Suivi avec TensorBoard](#suivi-avec-tensorboard)
  - [Génération d'images](#génération-dimages)
  - [Comparer les deux approches](#comparer-les-deux-approches)
- [Interface web](#interface-web)
- [Structure du projet](#structure-du-projet)

## Installation

Prérequis : Python 3.11+, un Mac Apple Silicon (M1/M2/M3...) pour l'accélération MPS (fonctionne aussi sur CPU/CUDA, juste plus lent), Node.js 18+ pour l'interface web.

```bash
pip install -r requirements.txt
```

## Préparation des données

Le projet utilise [CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog), qui étend CelebA avec des légendes en langage naturel par image. Le dossier `data/` (non versionné, voir `.gitignore`) doit contenir :

```
data/
├── img_align_celeba/              # 202 599 images de visages, 178×218 px
├── celeba_caption/captions.json   # légendes ({"overall_caption": "..."} par image)
├── Anno/list_attr_celeba.txt      # 40 attributs binaires par image
└── Eval/list_eval_partition.txt   # split officiel train/val/test
```

- Images + attributs : [CelebA officiel](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Légendes (`captions.json`) : dossier "text" du repo [CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog) (Google Drive)

Vérifier que tout est en place :

```bash
python -m src.backend.shared.dataset
```

## Architecture

```
src/backend/
├── shared/           # commun aux deux approches
│   ├── dataset.py         CelebACaptionDataset : image + caption
│   ├── device.py          sélection mps > cuda > cpu
│   ├── text_encoder.py    CLIP (openai/clip-vit-base-patch32), gelé
│   ├── time_embedding.py  timestep t -> vecteur sinusoïdal + MLP
│   ├── unet_blocks.py     ResBlock (injection du temps) + CrossAttentionBlock (injection du texte)
│   ├── unet.py            U-Net complet (~23M paramètres), PARTAGÉ entre les deux approches
│   └── preview.py         sauvegarde d'une grille d'images générées
├── diffusion/         # DDPM (Ho et al., 2020)
│   ├── gaussian_diffusion.py   bruitage (q_sample) et débruitage (p_sample)
│   ├── train.py                loss = MSE(bruit prédit, bruit réel)
│   └── sample.py                génération par T pas de débruitage itératif
├── flow_matching/     # Conditional Flow Matching (Lipman et al., 2023)
│   ├── flow_matching.py        interpolation linéaire image <-> bruit, vélocité cible
│   ├── train.py                loss = MSE(vélocité prédite, vélocité réelle)
│   └── sample.py                génération par résolution d'ODE (Euler explicite)
├── compare.py         # génère les mêmes prompts avec les deux checkpoints
└── frontend/
    ├── api/main.py    # API FastAPI servant la génération au frontend web
    └── web/            # frontend React (Vite)

src/cli.py              # CLI unifiée (train / sample / compare)
```

**Ce qui est partagé** : le dataset, l'encodeur texte CLIP (gelé, jamais réentraîné) et le U-Net. Le U-Net ne « sait » pas s'il fait de la diffusion ou du flow matching — il apprend juste à transformer `(image bruitée x_t, t, texte)` en une sortie de même forme que `x_t` (un bruit pour le DDPM, une vélocité pour le flow matching).

**Ce qui diffère** : la façon dont `x_t` est construit à l'entraînement (bruitage par schedule discret vs interpolation linéaire continue), la cible de la loss, et la boucle de génération (débruitage stochastique T pas vs résolution d'ODE déterministe, généralement moins de pas).

## Utilisation de la CLI

Toutes les commandes passent par `python -m src.cli <commande> ...`. Lancer `python -m src.cli --help` (ou `... <commande> --help`) pour la liste complète des options à tout moment.

### Entraînement

**DDPM :**

```bash
python -m src.cli train diffusion \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-4 \
    --timesteps 1000
```

**Flow Matching :**

```bash
python -m src.cli train flow_matching \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-4
```

Options communes aux deux commandes :

| Option | Défaut | Description |
|---|---|---|
| `--epochs` | 1 | Nombre de passages complets sur le dataset |
| `--batch-size` | 32 | Taille de batch |
| `--lr` | 2e-4 | Learning rate (Adam) |
| `--image-size` | 64 | Résolution (carrée) des images |
| `--cfg-dropout-prob` | 0.1 | Probabilité de remplacer la caption par une chaîne vide pendant l'entraînement (classifier-free guidance) |
| `--limit-batches N` | — | N'entraîne que sur N batches par epoch — utile pour un smoke test rapide avant un run complet |
| `--checkpoint-dir` | `checkpoints/<approche>` | Dossier de sortie (poids, previews, logs TensorBoard) |
| `--checkpoint-every-steps` | 500 | Fréquence de sauvegarde des poids |
| `--preview-every-steps` | 500 | Fréquence de génération d'images de contrôle |
| `--resume PATH` | — | Reprend l'entraînement depuis un checkpoint `.pt` |
| `--device` | auto | Force `mps` / `cuda` / `cpu` (sinon auto-détecté, `mps` privilégié) |

Spécifique à `train diffusion` : `--timesteps` (nombre d'étapes T du schedule de bruit, 1000 par défaut).
Spécifique à `train flow_matching` : `--preview-num-steps` (pas du solveur ODE utilisé pour les previews, 50 par défaut).

**Reprendre un entraînement interrompu :**

```bash
python -m src.cli train diffusion --resume checkpoints/diffusion/latest.pt --epochs 10
```

**Smoke test rapide** (vérifie que tout tourne, en quelques secondes, avant un run long) :

```bash
python -m src.cli train diffusion --limit-batches 3 --batch-size 4 --epochs 1
```

À chaque run, le dossier `--checkpoint-dir` (par défaut `checkpoints/diffusion/` ou `checkpoints/flow_matching/`) contient :

```
checkpoints/diffusion/
├── latest.pt            # dernier checkpoint (poids U-Net + optimizer + step)
├── epoch_N.pt            # checkpoint de fin de chaque epoch
├── previews/              # grilles d'images .png générées périodiquement
└── tensorboard/            # logs TensorBoard
```

### Suivi avec TensorBoard

Pendant (ou après) l'entraînement, dans un autre terminal :

```bash
tensorboard --logdir checkpoints/diffusion/tensorboard
# ou pour comparer les deux runs côte à côte :
tensorboard --logdir checkpoints
```

Puis ouvrir [http://localhost:6006](http://localhost:6006). On y trouve :
- `loss/train` : la loss MSE au fil des steps
- `preview/samples` : les grilles d'images de contrôle générées périodiquement (onglet *Images*)
- `time/epoch_minutes` : temps par epoch
- `config` : hyperparamètres du run (onglet *Text*)

### Génération d'images

À partir d'un checkpoint entraîné :

```bash
python -m src.cli sample diffusion \
    --checkpoint checkpoints/diffusion/latest.pt \
    --prompt "a smiling woman with bangs" \
    --prompt "a man with a beard and glasses" \
    --out report/my_samples.png
```

```bash
python -m src.cli sample flow_matching \
    --checkpoint checkpoints/flow_matching/latest.pt \
    --prompt "a smiling woman with bangs" \
    --num-steps 50 \
    --out report/my_samples_fm.png
```

`--prompt` est répétable (une image générée par prompt, toutes dans la même grille).

### Comparer les deux approches

Génère les mêmes prompts avec les deux checkpoints, côte à côte :

```bash
python -m src.cli compare \
    --ddpm-checkpoint checkpoints/diffusion/latest.pt \
    --flow-checkpoint checkpoints/flow_matching/latest.pt \
    --prompt "a smiling woman with bangs" \
    --prompt "an elderly man with glasses"
```

Produit `report/comparison_ddpm_vs_flow_matching_ddpm.png` et `..._flow_matching.png`.

## Interface web

Une interface web permet de générer des images depuis le navigateur (formulaire prompt + choix de l'approche), en s'appuyant sur les checkpoints entraînés via la CLI.

### 1. Démarrer l'API (backend FastAPI)

```bash
uvicorn src.frontend.api.main:app --reload --port 8000
```

L'API charge les modèles paresseusement (au premier appel de `/generate` pour chaque approche) — pas besoin d'avoir entraîné les deux approches pour démarrer.

Endpoints :
- `GET /health` — vérifie que le serveur tourne et quel device est utilisé
- `GET /status` — indique si un checkpoint par défaut existe pour chaque approche
- `POST /generate` — génère une image (`prompt`, `approach`, `num_steps`, `image_size`, `checkpoint` optionnel)

### 2. Démarrer le frontend (React)

Dans un autre terminal :

```bash
cd src/frontend/web
npm install    # première fois seulement
npm run dev
```

Ouvrir [http://localhost:5173](http://localhost:5173). Par défaut le frontend appelle l'API sur `http://127.0.0.1:8000` ; pour changer l'URL, copier `.env.example` en `.env.local` et ajuster `VITE_API_BASE_URL`.

Le formulaire permet de choisir le prompt, l'approche (DDPM ou Flow Matching), le nombre d'étapes de sampling et la résolution. Si le checkpoint par défaut (`checkpoints/<approche>/latest.pt`) n'existe pas encore, un avertissement s'affiche invitant à entraîner le modèle via la CLI.

## Structure du projet

```
.
├── data/                      # dataset CelebA-Dialog (non versionné)
├── checkpoints/                # poids entraînés, previews, logs TensorBoard (non versionné)
├── report/                     # figures générées par les auto-tests des modules
├── requirements.txt
└── src/
    ├── cli.py                  # CLI unifiée
    └── backend/
        ├── shared/              # dataset, device, CLIP, time embedding, U-Net
        ├── diffusion/            # DDPM : bruitage, entraînement, sampling
        ├── flow_matching/         # Flow Matching : interpolation, entraînement, sampling
        ├── compare.py            # comparaison DDPM vs Flow Matching
        └── frontend/
            ├── api/               # backend FastAPI
            └── web/                # frontend React (Vite)
```

Chaque module de `src/backend/` a un bloc `if __name__ == "__main__":` qui sert d'auto-test : `python -m src.backend.shared.unet`, `python -m src.backend.diffusion.gaussian_diffusion`, etc. génèrent des figures de vérification dans `report/`.

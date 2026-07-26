# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`cnn-collab` is a minimal teaching project for practicing **Git/GitHub team collaboration**, built around a simple CNN that classifies MNIST digits. The code is intentionally kept small and simple so that students can extend it. Documentation and comments in this repo are written in Traditional Chinese (Taiwan). Match that language when editing docstrings/comments in these files, and default to Traditional Chinese in conversation with the user unless they write in English.

The README references a `docs/教學講義.md` (workshop handout with full exercise steps), but `/docs/` is excluded via `.gitignore` and is not present in this checkout — don't assume it exists.

## Commands

```bash
pip install -r requirements.txt   # torch, torchvision, matplotlib
python train.py                   # trains MinimalCNN on MNIST for 3 epochs, prints per-epoch train/val loss & acc
python model.py                   # sanity-check: runs a dummy batch through MinimalCNN and prints output shape
python sandbox.py                 # prints the current sandbox "clean baseline" marker string
```

There is no test suite, linter, or CI configuration in this repo — don't invent commands for these.

MNIST data downloads to `./data` on first `train.py` run (via `torchvision.datasets.MNIST(download=True)`), so the first run requires network access.

## Architecture

- `model.py` — `MinimalCNN`: 2 conv layers (1→16→32 channels, 3x3 kernel, padding 1) each followed by ReLU and 2x2 max-pool, flatten, then a single `nn.Linear(32*7*7, num_classes)` head. Input is `(B, 1, 28, 28)`, output is `(B, 10)` logits.
- `data.py` — `build_transforms()` returns the training preprocessing pipeline (currently just ToTensor + MNIST mean/std normalization); `get_dataloaders(batch_size, root)` downloads MNIST and returns `(train_loader, test_loader)`.
- `utils.py` — `accuracy(logits, labels)`: batch accuracy helper.
- `train.py` — training entrypoint: `train_one_epoch` / `evaluate` loops, Adam optimizer, `CrossEntropyLoss`, fixed 3 epochs, device auto-selected (`cuda` if available else `cpu`).
- `sandbox.py` — deliberately unrelated to the model code; it exists solely as a scratch file for git branch/reset exercises (see the "還原練習" comment header). Freely modifiable/breakable when doing git exercises; don't treat changes here as meaningful to the CNN project.

### The "功能註冊區" (feature registration) convention in `train.py`

`train.py` has an `ACTIVE_FEATURES` list near the top marked as the "功能註冊區" (feature registration zone). This is a deliberate collaboration exercise: each student/group implements a feature elsewhere in their own branch (e.g. data augmentation in `data.py`, a deeper model variant, an LR scheduler, eval tooling) and then appends a short string tag (e.g. `"augment"`, `"deep_model"`, `"lr_schedule"`, `"eval_tools"`) to `ACTIVE_FEATURES` to signal it's enabled — since everyone edits this same list, it's designed to produce merge conflicts that students practice resolving. When asked to add a feature, follow this pattern: implement the feature in the relevant module and append a tag to `ACTIVE_FEATURES` rather than wiring in ad hoc flags elsewhere.

## Repo hygiene note

`Miniconda3-latest-Linux-x86_64.sh` (~190MB installer) sits untracked at the repo root. It is not part of the project and should not be added to git.

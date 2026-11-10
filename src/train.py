"""Train the log-mel CNN (Phase 3). Run from the project root, after `python -m src.features logmel`:

    python -m src.train                  # full training with early stopping
    python -m src.train --epochs 1       # timing run: one epoch, to estimate the total cost

Defaults are the Phase 3 run: first 2.0 s of each note (126 frames), up to 18 epochs, 35 min wall-clock budget.
On the full 4 s (251 frames) one epoch took ~3.7 min on a laptop CPU, which would have left room for only ~9 epochs.

Only the "train" instruments minus the validation instruments are used for fitting. The validation instruments
(see `make_val_split`) choose the best epoch. The test instruments are not loaded at all in this script.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from src.dataset import ROOT, load_notes, with_validation
from src.features import load_logmel
from src.model import LogMelCNN, n_parameters

FAMILIES = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal"]
CHECKPOINT = ROOT / "models" / "cnn_logmel.pt"
HISTORY = ROOT / "reports" / "cnn_history.json"


def band_stats(X, chunk=1024):
    """Mean and std of each mel band over all notes and frames of X (notes, mels, frames). Float64 accumulation."""
    total = np.zeros(X.shape[1])
    total_sq = np.zeros(X.shape[1])
    for i in range(0, len(X), chunk):
        block = X[i : i + chunk].astype(np.float64)
        total += block.sum(axis=(0, 2))
        total_sq += (block**2).sum(axis=(0, 2))
    count = X.shape[0] * X.shape[2]
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(y, n_classes=len(FAMILIES)):
    """n_samples / (n_classes * count_c): the same formula as sklearn's class_weight="balanced"."""
    counts = np.bincount(y, minlength=n_classes)
    return len(y) / (n_classes * np.maximum(counts, 1))


def spec_augment(x, n_freq_masks=2, max_freq=16, n_time_masks=2, max_time=25):
    """SpecAugment (Park et al., 2019), masking only: zero out random bands of mel bins and of frames, per note.

    x is (batch, 1, mels, frames) and already normalised, so 0 means "the training average of that band".
    """
    batch, _, n_mels, n_frames = x.shape
    mask = torch.zeros(batch, 1, n_mels, n_frames, dtype=torch.bool)
    for size, max_width, axis in [(n_mels, max_freq, 2), (n_frames, max_time, 3)]:
        positions = torch.arange(size)
        for _ in range(n_freq_masks if axis == 2 else n_time_masks):
            width = torch.randint(0, max_width + 1, (batch,))
            start = (torch.rand(batch) * (size - width + 1)).long()
            band = (positions >= start[:, None]) & (positions < (start + width)[:, None])  # (batch, size)
            mask |= band[:, None, :, None] if axis == 2 else band[:, None, None, :]
    return x.masked_fill(mask, 0.0)


def balanced_accuracy(y_true, y_pred):
    """Mean recall over the families present in y_true (validation has only some families)."""
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def to_input(X, mean, std):
    """float16 log-mels (n, mels, frames) -> normalised float32 tensor (n, 1, mels, frames)."""
    x = (torch.from_numpy(X).float() - torch.from_numpy(mean)[:, None]) / torch.from_numpy(std)[:, None]
    return x[:, None]


@torch.no_grad()
def predict_logits(model, X, mean, std, batch_size=256):
    model.eval()
    return torch.cat([model(to_input(X[i : i + batch_size], mean, std)) for i in range(0, len(X), batch_size)]).numpy()


def load_checkpoint(path=CHECKPOINT):
    ckpt = torch.load(path, weights_only=False)
    model = LogMelCNN(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main(epochs, patience, batch_size, lr, frames, budget_min, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    notes = with_validation(load_notes())
    fit, val = notes[notes["split"] == "train"], notes[notes["split"] == "val"]
    print(f"fit: {len(fit)} notes / {fit['instrument'].nunique()} instruments; "
          f"val: {len(val)} notes / {val['instrument'].nunique()} instruments: {sorted(val['instrument'].unique())}")

    start = time.time()
    X_fit = np.ascontiguousarray(load_logmel(fit)[:, :, :frames])  # contiguous: fast per-batch indexing
    X_val = np.ascontiguousarray(load_logmel(val)[:, :, :frames])
    y_fit = fit["family"].map(FAMILIES.index).to_numpy(copy=True)
    y_val = val["family"].map(FAMILIES.index).to_numpy(copy=True)
    mean, std = band_stats(X_fit)  # fit instruments only: nothing from val or test leaks into the normalisation
    print(f"loaded {X_fit.shape} + {X_val.shape} in {time.time() - start:.0f} s")

    model_args = dict(n_classes=len(FAMILIES), channels=(16, 32, 64, 128), dropout=0.3)
    model = LogMelCNN(**model_args)
    print(f"{n_parameters(model):,} parameters, {torch.get_num_threads()} threads")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y_fit), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    config = dict(epochs=epochs, patience=patience, batch_size=batch_size, lr=lr, frames=frames, seed=seed,
                  model_args=model_args, val_instruments=sorted(val["instrument"].unique()))
    history, best, since_best = [], -1.0, 0
    CHECKPOINT.parent.mkdir(exist_ok=True)
    HISTORY.parent.mkdir(exist_ok=True)
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(X_fit))
        losses, correct = [], []
        for i in range(0, len(order), batch_size):
            idx = np.sort(order[i : i + batch_size])
            x = spec_augment(to_input(X_fit[idx], mean, std))
            target = torch.from_numpy(y_fit[idx])
            logits = model(x)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct.append((logits.argmax(1) == target).numpy())
        t_train = time.time() - t0

        val_logits = predict_logits(model, X_val, mean, std)
        val_loss = nn.functional.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(y_val)).item()
        val_bacc = balanced_accuracy(y_val, val_logits.argmax(1))
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), train_acc=float(np.concatenate(correct).mean()),
                   val_loss=val_loss, val_bacc=val_bacc, sec_train=t_train, sec_total=time.time() - t0)
        history.append(row)

        improved = val_bacc > best
        if improved:
            best, since_best = val_bacc, 0
            torch.save(dict(state_dict=model.state_dict(), model_args=model_args, mean=mean, std=std, epoch=epoch,
                            val_bacc=val_bacc, config=config, families=FAMILIES), CHECKPOINT)
        else:
            since_best += 1
        HISTORY.write_text(json.dumps(dict(config=config, history=history, best_epoch=max(history, key=lambda r: r["val_bacc"])["epoch"]), indent=1))
        print(f"epoch {epoch:2d}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}  "
              f"val loss {val_loss:.3f} bacc {val_bacc:.3f}{' *' if improved else ''}  "
              f"({row['sec_train']:.0f} s train, {row['sec_total']:.0f} s total)", flush=True)

        elapsed = (time.time() - train_start) / 60
        if since_best >= patience:
            print(f"early stopping: no improvement for {patience} epochs")
            break
        if elapsed + elapsed / epoch > budget_min:
            print(f"stopping: another epoch would exceed the {budget_min} min budget ({elapsed:.1f} min used)")
            break
    print(f"best val balanced accuracy {best:.3f} at epoch {max(history, key=lambda r: r['val_bacc'])['epoch']}; "
          f"checkpoint {CHECKPOINT.relative_to(ROOT)}; {(time.time() - train_start) / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--frames", type=int, default=126, help="keep only the first N frames: 126 = 2.0 s, 251 = all 4 s")
    p.add_argument("--budget-min", type=float, default=35, help="stop before an epoch that would exceed this")
    p.add_argument("--seed", type=int, default=0)
    main(**vars(p.parse_args()))

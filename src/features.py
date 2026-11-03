"""Audio features. Run `python -m src.features` from the project root to compute and cache them for all notes."""
import time

import librosa
import numpy as np
import pandas as pd

from src.dataset import ROOT, SR, load_notes

N_FFT, HOP, N_MELS = 1024, 256, 128  # same STFT / mel settings as notebook 00
N_MFCC = 20
MFCC_FILE = ROOT / "data" / "features" / "mfcc_stats.npz"
MFCC_NAMES = [f"mean_{i}" for i in range(N_MFCC)] + [f"std_{i}" for i in range(N_MFCC)]


def mfcc(y):
    """MFCC matrix (N_MFCC, frames): DCT of the log-mel spectrogram, keeping the first N_MFCC coefficients."""
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)


def mfcc_stats(y):
    """Mean and standard deviation of each MFCC over time: 2 * N_MFCC numbers per note."""
    m = mfcc(y)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def load_mfcc_stats(notes):
    """Cached MFCC statistics, one row per note in `notes`, in the same order."""
    cache = np.load(MFCC_FILE)
    row = pd.Series(np.arange(len(cache["notes"])), index=cache["notes"])
    return cache["X"][row[notes["note"]].to_numpy()]


if __name__ == "__main__":
    notes = load_notes()
    start, rows = time.time(), []
    for i, path in enumerate(notes["path"]):
        rows.append(mfcc_stats(librosa.load(path, sr=None)[0]))
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(notes)}  ({time.time() - start:.0f} s)")
    MFCC_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MFCC_FILE, notes=notes["note"].to_numpy(dtype=str), X=np.stack(rows).astype(np.float32))
    print(f"{len(rows)} notes x {len(MFCC_NAMES)} features written to {MFCC_FILE.relative_to(ROOT)}")

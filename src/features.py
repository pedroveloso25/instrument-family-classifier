"""Audio features, computed once and cached in `data/features/`.

Run from the project root:
    python -m src.features          # MFCC statistics (Phase 2 baseline), ~6 min on CPU
    python -m src.features logmel   # full log-mel spectrograms (Phase 3 CNN)
"""
import sys
import time

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.dataset import ROOT, SR, load_notes

N_FFT, HOP, N_MELS = 1024, 256, 128  # same STFT / mel settings as notebook 00
N_MFCC = 20
N_SAMPLES = 4 * SR  # every NSynth note is exactly 4 s
N_FRAMES = 1 + N_SAMPLES // HOP  # 251 frames (librosa pads the signal, center=True)
FEATURES = ROOT / "data" / "features"
MFCC_FILE = FEATURES / "mfcc_stats.npz"
LOGMEL_FILE = FEATURES / "logmel_f16.npy"  # (notes, N_MELS, N_FRAMES), float16
LOGMEL_INDEX = FEATURES / "logmel_notes.txt"  # note names, one per line, same order as the rows above
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


def logmel(y):
    """Log-mel spectrogram (N_MELS, N_FRAMES) in dB, exactly as in notebook 00.

    Power mel spectrogram, then dB relative to the loudest bin of *this* note (`ref=np.max`), floored at -80 dB.
    So every note has its maximum at 0 dB: the overall level (velocity, recording gain) is removed and only the
    *shape* of the energy in time and frequency is left. Values lie in [-80, 0].
    """
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def _logmel_files(paths):
    out = np.empty((len(paths), N_MELS, N_FRAMES), dtype=np.float16)
    for i, path in enumerate(paths):
        y = librosa.load(path, sr=None)[0]
        assert len(y) == N_SAMPLES, f"{path}: {len(y)} samples"
        out[i] = logmel(y)
    return out


def load_logmel(notes):
    """Cached log-mel spectrograms (float16) for the notes in `notes`, in the same order, loaded into RAM."""
    names = LOGMEL_INDEX.read_text().split()
    index = pd.Series(np.arange(len(names)), index=names)
    cache = np.load(LOGMEL_FILE, mmap_mode="r")
    rows = index[notes["note"]].to_numpy()
    order = np.argsort(rows)  # read the memmap in file order (much faster on disk), then undo the sort
    X = np.empty((len(rows), N_MELS, N_FRAMES), dtype=np.float16)
    X[order] = cache[rows[order]]
    return X


def cache_mfcc(notes):
    start, rows = time.time(), []
    for i, path in enumerate(notes["path"]):
        rows.append(mfcc_stats(librosa.load(path, sr=None)[0]))
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(notes)}  ({time.time() - start:.0f} s)")
    np.savez(MFCC_FILE, notes=notes["note"].to_numpy(dtype=str), X=np.stack(rows).astype(np.float32))
    print(f"{len(rows)} notes x {len(MFCC_NAMES)} features written to {MFCC_FILE.relative_to(ROOT)}")


def cache_logmel(notes, chunk=256):
    """Writes all log-mels into one float16 .npy (~1.1 GB), filled chunk by chunk by parallel workers."""
    start = time.time()
    out = np.lib.format.open_memmap(LOGMEL_FILE, mode="w+", dtype=np.float16, shape=(len(notes), N_MELS, N_FRAMES))
    chunks = [notes["path"].iloc[i : i + chunk].tolist() for i in range(0, len(notes), chunk)]
    results = Parallel(n_jobs=-1, return_as="generator")(delayed(_logmel_files)(paths) for paths in chunks)
    for k, block in enumerate(results):
        out[k * chunk : k * chunk + len(block)] = block
        if (k + 1) % 10 == 0:
            print(f"{min((k + 1) * chunk, len(notes))}/{len(notes)}  ({time.time() - start:.0f} s)")
    out.flush()
    LOGMEL_INDEX.write_text("\n".join(notes["note"]))
    print(f"{len(notes)} log-mels of shape {N_MELS}x{N_FRAMES} written to {LOGMEL_FILE.relative_to(ROOT)} "
          f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mfcc"
    FEATURES.mkdir(parents=True, exist_ok=True)
    {"mfcc": cache_mfcc, "logmel": cache_logmel}[which](load_notes())

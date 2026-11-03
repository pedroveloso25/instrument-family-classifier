# Instrument Family Classifier (NSynth)

> **A learning project.** This is my first project working with audio. My background is in
> computer vision (YOLO), NLP and LLMs; I had never touched signal processing before this.
> The goal is not state-of-the-art accuracy, but a small, honest project where every design
> decision is explained and every result (good or bad) is reported.

Classifying the **instrument family** (bass, brass, flute, guitar, ...) of a single musical note
from its audio, using the [NSynth dataset](https://magenta.withgoogle.com/datasets/nsynth).

## Plan

| Phase | What | Status |
|---|---|---|
| 0 | Audio fundamentals: waveform, STFT, mel-spectrogram ([notebook](notebooks/00_audio_fundamentals.ipynb)) | done |
| 1 | Dataset and train/test split grouped by instrument ([notebook](notebooks/01_dataset_and_split.ipynb)) | done |
| 2 | Baseline: MFCC mean/std + logistic regression ([notebook](notebooks/02_mfcc_baseline.ipynb)) — 0.19 balanced accuracy (chance 0.10) | done |
| 3 | Small CNN on log-mel spectrograms (PyTorch, CPU) | — |
| 4 | Evaluation, error analysis, write-up | — |

Out of scope: source separation, large pretrained models, deployment, web UI.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Data: NSynth `valid` and `test` splits (json/wav, ~1.4 GB total), extracted into `data/raw/`.
The 23.8 GB `train` split is intentionally not used (CPU-only hardware).

NSynth's `valid` and `test` contain the same 53 instruments, so they can't serve as train/test for each other.
They are pooled and re-split by instrument: one instrument per family goes to test (10 instruments, 3,435 notes),
the other 43 to train. The split is saved in [`splits/instruments.csv`](splits/instruments.csv); to regenerate it:

```bash
python -m src.dataset
```

The MFCC features used in Phase 2 are cached in `data/features/` (~6 min on CPU):

```bash
python -m src.features
```

## License

Code: MIT. NSynth data: CC BY 4.0, by Google Inc. (Engel et al., 2017).

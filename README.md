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
| 0 | Audio fundamentals: waveform, STFT, mel-spectrogram | in progress |
| 1 | Dataset and leakage-free split (grouped by instrument) | — |
| 2 | Baseline: aggregated MFCC statistics + classical model | — |
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
The 23.8 GB `train` split is intentionally not used (CPU-only hardware; see Phase 1 notes).

## License

Code: MIT. NSynth data: CC BY 4.0, by Google Inc. (Engel et al., 2017).

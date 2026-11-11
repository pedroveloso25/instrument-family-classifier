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
| 3 | Small CNN on log-mel spectrograms, PyTorch on CPU ([notebook](notebooks/03_cnn_logmel.ipynb)) — 0.30 balanced accuracy on the same test | done |
| 4 | Evaluation and error analysis: baseline vs CNN, confounds, split variance ([notebook](notebooks/04_evaluation.ipynb)) | done |

A study guide with every design decision, likely interview questions and a glossary is in [STUDY.md](STUDY.md).

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

The log-mel spectrograms used by the CNN are cached the same way (~1 min, 1.1 GB float16), then the CNN is trained
(~25 min on a laptop CPU; early stopping on validation instruments, checkpoint in `models/`):

```bash
python -m src.features logmel
python -m src.train
```

## Results

Same test set for both models: 10 instruments never seen in training (one per family), 3,435 notes. Chance = 0.10.

| Model | Train instruments | Balanced accuracy | Accuracy |
|---|---|---|---|
| MFCC mean/std + logistic regression | 43 | 0.188 | 0.180 |
| same, on the CNN's 36 instruments | 36 | 0.187 | 0.181 |
| Small CNN on log-mel (first 2 s) | 36 (+7 for validation) | **0.302** | 0.287 |
| *MFCC baseline, random note split (leaky, for reference)* | — | *0.602* | — |
| *MFCC baseline over 10 split seeds* | 43 | *0.243 ± 0.062* | — |

What the numbers say:
- **The grouped split matters most.** The same baseline scores 0.60 with a random note split and 0.19 with unseen instruments.
- **The CNN beats the baseline on the same split** (+0.11). It fixes the baseline's "everything steady is an organ" habit and
  reed → brass, but calls every test mallet note a keyboard, and puts 89% of its predictions into four families.
- **Half the families are still near zero** for the CNN (flute, keyboard, mallet, organ, vocal).
- **Shortcuts:** the CNN's errors follow the production source (acoustic / electronic / synthetic) far more than chance,
  and its accuracy drops to ~0.15 above MIDI 72, where high notes tend to be called "bass".
- **One split is not enough to rank models confidently.** The baseline alone moves from 0.16 to 0.33 depending on which 10
  instruments land in test; the CNN was trained once.

## Limitations

- One test instrument per family: per-family results are really per-instrument results.
- Few training instruments per family (1 for flute and vocal, 2 for mallet); flute and vocal are tested on a synthetic
  instrument after training on an acoustic one.
- Family and production source are partly the same label in this data.
- The CNN sees only the first 2 s of each note (CPU budget), was trained once, and was not retrained on validation instruments.
- Validation covers 7 families only and is one instrument per family, so the chosen epoch is noisy.

## Next steps

1. Train the CNN on 3–5 split seeds and compare paired differences with the baseline (the one claim this project cannot make yet).
2. Add instruments from NSynth's `train` split (the main bottleneck is instrument diversity, not model size).
3. Use the full 4 s, then try a pretrained audio embedding (e.g. PANNs) with a linear probe as a strong reference.
4. Listen to the mallet → keyboard and high-pitch → bass errors before changing anything else.
5. Pool over time only (keep the frequency axis) and compare with global average pooling.

## License

Code: MIT. NSynth data: CC BY 4.0, by Google Inc. (Engel et al., 2017).

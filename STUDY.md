# Study guide

Notes for revising this project: why each decision was made, what I would say if asked about it, and the vocabulary.
Everything here is backed by a notebook; the notebook links are in the [README](README.md).

## The project in one minute

I classify the instrument family (10 classes) of a single NSynth note. NSynth's `valid` and `test` folders share all 53
instruments, so I pooled them and split **by instrument**: 43 to train, 10 (one per family) to test. That split is the
main decision. With a random note split, the baseline scores 0.60; on unseen instruments it scores 0.19. A small CNN on
log-mel spectrograms reaches **0.30** on the same unseen instruments. It fixes some of the baseline's structured errors and
makes new ones. The analysis shows it partly relies on shortcuts (production source, pitch). The test set has only 10
instruments, and the baseline alone moves between 0.16 and 0.33 across splits, so the gain is suggestive, not proven. The
bottleneck is instrument diversity, not model size.

## Numbers to remember

| | |
|---|---|
| Notes / instruments / families | 16,774 / 53 / 10 |
| Train / test (instruments) | 43 / 10 (one per family); CNN fits on 36, validates on 7 |
| Chance (balanced accuracy) | 0.10 |
| Baseline, random note split (leaky) | 0.602 |
| Baseline, grouped split | **0.188** (train 0.75) |
| Baseline over 10 split seeds | 0.243 ± 0.062 (0.16–0.33) |
| CNN, grouped split (seed 0) | **0.302**; best validation 0.584 at epoch 6 |
| CNN size / cost | ~99k parameters; ~24 min on a laptop CPU, first 2 s of each note |
| Biggest fixes | organ as default class (24% → 0% of non-organ notes), reed → brass 0.40 → 0.06 |
| Biggest new failure | mallet → keyboard 0.46 → 1.00 |
| Shortcut evidence | CNN predictions 75% source-compatible vs 63% for a source-blind model; accuracy ~0.15 above MIDI 72 |

## Part 1 — Design decisions

Each section: **what** was done, **why**, the **alternative** I discarded, and **what would break** if the premise changed.

### 1. Train/test split grouped by instrument

- **What.** NSynth `valid` + `test` folders pooled (53 instruments, 16,774 notes). For each family, one random instrument
  (seed 0) and *all* its notes go to test: 10 test instruments, 3,435 notes. Saved in `splits/instruments.csv`.
- **Why.** Each instrument is recorded at up to 88 pitches × 5 velocities. Two notes of the same instrument a semitone apart
  are almost the same sound. A random note split puts near-copies of training notes in test, and the model is rewarded for
  recognising *the instrument* instead of *the family*. Measured: same model, same features, **0.60 with a random split
  vs 0.19 grouped**. The question I want answered is "can it name the family of an instrument it has never heard?"
- **Alternative discarded.** Random stratified note split (leaky, see above). Also using NSynth's `valid` as train and
  `test` as test: impossible, both contain the same 53 instruments.
- **What breaks if the premise changes.** If the real use case were "recognise more notes from instruments we already have"
  (e.g. a closed sample library), the random split would be the *right* one and 0.60 the relevant number. And if a family
  had only one instrument, it could not appear in both train and test at all.

### 2. MFCC mean/std as the baseline feature

- **What.** 20 MFCCs per frame, summarised over time by mean and std → 40 numbers per note.
- **Why.** It is the classic pre-deep-learning timbre descriptor: a compact description of the spectral envelope, cheap,
  and it sets a floor that tells me how much the CNN actually adds.
- **Alternative discarded.** Hand-crafted descriptors (spectral centroid, roll-off, zero-crossing rate, attack time) or
  more statistics (deltas, percentiles). Both are reasonable, but more features means more choices to tune with no
  validation set at the time.
- **What breaks.** Mean/std throws away *the order of events in time*: a note and its time-reversal have identical features.
  If families differed mostly by temporal shape (attack, decay), this baseline would be structurally blind to it.

### 3. Standardising features before logistic regression

- **What.** `StandardScaler` (fit on training notes) before the logistic regression.
- **Why.** `mean_0` (≈ loudness) is in the hundreds, other coefficients around ±10. L2 regularisation penalises all weights
  equally, so without scaling it would effectively regularise features very unevenly, and the solver converges worse.
- **Alternative discarded.** No scaling, or min-max scaling (sensitive to outliers).
- **What breaks.** With a tree model (random forest, gradient boosting) scaling would be irrelevant. If the scaler were fit
  on train+test, test statistics would leak into the model (small here, but a bad habit).

### 4. Class weights ("balanced")

- **What.** Each family weighted by `n / (n_families × n_family)` in the loss. Same formula in sklearn
  (`class_weight="balanced"`) and in the CNN's weighted cross-entropy.
- **Why.** Bass has ~6× more notes than vocal overall (~25× in the CNN's fit set). Without weights, the cheapest way to
  lower the loss is to favour big classes, which hurts balanced accuracy. Using the *same* weighting in both models keeps the
  comparison fair.
- **Alternative discarded.** Oversampling small classes (duplicates the very few small-family instruments, which encourages
  memorising them), or no weighting and reporting plain accuracy.
- **What breaks.** If the deployment prior were very unbalanced and we cared about overall accuracy (e.g. 90% of real queries
  are guitar), weighting would cost accuracy where it matters. Weights also do not create new *instruments*: they rebalance
  notes, not the diversity within a family.

### 5. Log-mel spectrogram as the CNN input

- **What.** STFT with a 1024-sample window (64 ms) and 256 hop (16 ms), 128 mel bands, power in dB relative to each note's
  maximum, floored at −80 dB. Image of 128 × 251 per 4 s note (cropped to 126 frames, see 8b).
- **Why.** Mel: more resolution at low frequencies where harmonics are spread apart and perception is finer; 513 → 128 bins.
  Log: audio energy spans ~80 dB; in dB, a gain becomes an additive shift and quiet harmonics become visible. `ref=np.max`
  removes the absolute level (velocity/recording gain), which says nothing about the family.
- **Alternative discarded.** Raw waveform (1D CNN, needs far more data to learn its own filterbank), linear STFT (larger,
  wasteful at high frequencies), CQT (log-frequency, pitch-shift-friendly, but slower and not needed for a first model),
  MFCC matrices (throws away fine spectral detail before the CNN can use it).
- **What breaks.** Mel is a model of *human* hearing. If the discriminative information were in fine high-frequency detail
  (e.g. noise texture above 4 kHz), mel would blur it. Per-note `ref=np.max` would hurt if loudness itself mattered for the label.

### 6. Normalisation per mel band, statistics from the training notes only

- **What.** Each of the 128 mel bands is standardised with its own mean/std, computed over all frames of the **fit** notes.
- **Why per band.** Low bands are much louder on average than high bands; without this, the input scale depends on the
  height in the image. **Why fit only.** The statistics are model parameters. Computing them on validation or test leaks
  information about held-out instruments into training.
- **Alternative discarded.** Per-note normalisation (removes the per-note spectral tilt, which is timbre), a single global
  mean/std (leaves the band-dependent scale), BatchNorm as the first layer (works too, but less explicit).
- **What breaks.** If test audio came from a different pipeline (other sample rate, microphone, mastering), the training band
  statistics would no longer centre it; the model would see shifted inputs (domain shift).

### 7. Validation grouped by instrument, carved from training

- **What.** One random training instrument per family to validation, only if that family has ≥ 3 training instruments:
  7 validation instruments (2,048 notes), 36 left for fitting. Flute, vocal (1 instrument) and mallet (2) give none.
- **Why.** Early stopping needs data the model is not fitted on, the test set must stay untouched, and a note-level
  validation would leak exactly like a note-level test.
- **Alternative discarded.** Random notes as validation (leaky, would keep "improving" while the model memorises instruments);
  using test for early stopping (makes the test score optimistic); cross-validation by instrument (better, but ×k training
  cost on CPU).
- **What breaks.** Validation covers 7 of 10 families and excludes the two hardest (flute, vocal), so it overestimates test
  (best validation 0.584 vs test 0.302).
  It is one instrument per family, so it is noisy. Fine for choosing one number (the epoch); tuning many hyperparameters on
  it would overfit those 7 instruments. And the CNN trains on 36 instruments vs the baseline's 43 (handled by also
  reporting a 36-instrument baseline).

### 8. Architecture: small 4-block CNN

- **What.** 4 × (Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2), channels 16/32/64/128, global average pooling, dropout 0.3,
  Linear(128 → 10). ~99k parameters. Adam 1e-3, batch 64.
- **Why.** The real dataset size is 36 instruments, not 11,291 notes, so capacity is not the limit; a small model memorises
  less. 3×3 kernels with pooling grow the receptive field to ~46 mel bands × 46 frames (~0.7 s). BatchNorm makes a fixed
  learning rate work without tuning. GAP makes the head tiny and the model length-agnostic.
- **Alternative discarded.** A bigger CNN or a pretrained audio model (PANNs, AST). Pretrained models would likely help a lot
  (they have heard thousands of instruments) but the goal was to understand the pipeline end to end, on CPU.
- **What breaks.** GAP averages over frequency as well as time, so absolute frequency position is only weakly kept. If the
  family depended on *where* energy is (e.g. a formant at a fixed Hz), pooling over time only and keeping the frequency axis
  would be better. Square 3×3 kernels treat time and frequency alike although they mean different things (notebook 00).

### 8b. Cropping to the first 2.0 s

- **What.** Only frames 0–125 (2.0 s) are used, for training, validation and test.
- **Why.** CPU budget (~40 min). A full-length epoch measured 224 s, so ~9 epochs would fit; at 2 s an epoch is ~80 s.
  NSynth notes are held 3 s then released, so the first 2 s keep the attack and the start of the sustain/decay.
- **Alternative discarded.** Fewer mel bands (frequency detail *is* the timbre signal), fewer epochs (early stopping would not
  work), random crops (augmentation, but the eval crop must be fixed anyway).
- **What breaks.** More than I expected: in every family, 35–84% of the frames in the last 2 s are still within 60 dB of
  the peak. The crop loses the late decay, the end of the sustain and the release (organ cuts off, piano dampers). It may
  contribute to mallet → keyboard (decay *length* is what separates them). Honest note: part of the 224 s was a slow
  non-contiguous data pipeline, and later epochs slowed down (thermal throttling), so the timing extrapolation was rough.

### 9. SpecAugment, and not pitch shift

- **What.** During training only: 2 frequency masks (≤ 16 mel bands) and 2 time masks (≤ 25 frames ≈ 0.4 s) per note,
  filled with 0 (= band mean after normalisation).
- **Why.** Forces the model not to rely on one narrow band or instant that may be specific to a training instrument. Free:
  it works on the cached spectrograms.
- **Why not pitch shift.** Pitch shifting by resampling or a phase vocoder moves the *whole* spectrum, formants included.
  Real instruments keep their formants (body resonances) roughly fixed when playing higher; a shifted cello sounds like a
  smaller cello (the "chipmunk" effect). Timbre is the class signal, so this creates wrong timbres with right labels. And
  NSynth already has each instrument at up to 88 real pitches, so it adds little.
- **Alternative discarded.** Pitch shift (above), mixup (blends two families into a sound that belongs to neither), adding
  noise/reverb (a good idea for real-world robustness, but NSynth is clean and the test is clean).
- **What breaks.** With formant-preserving shifting (e.g. a vocoder that keeps the envelope), pitch augmentation would be
  defensible. If the test data had reverb or noise, the current augmentation would not cover it.

### 10. Early stopping on validation balanced accuracy

- **What.** Evaluate after every epoch; keep the checkpoint with the best validation balanced accuracy; stop after 5 epochs
  without improvement (max 18 epochs, 35 min budget).
- **Why.** The number of epochs is the one hyperparameter that matters most for overfitting here, and it is chosen without
  touching test. The criterion is the final metric itself, not the loss: cross-entropy also rewards confidence, and with so
  few instruments the model becomes confidently wrong on new ones while accuracy is still flat. In the actual run, validation
  loss was lowest at epoch 4 and then rose (1.43 → 2.3) while balanced accuracy peaked at epoch 6 (0.584); the run stopped
  at epoch 11, after ~24 min.
- **Alternative discarded.** Fixed number of epochs (arbitrary), stopping on validation loss, retraining on fit + val for the
  chosen number of epochs (better use of data, but doubles CPU time and the chosen epoch may not transfer).
- **What breaks.** With such a small, noisy validation set, the "best epoch" is partly luck: a different validation
  instrument could pick a different epoch. If validation and test were very different (they are: no flute/vocal in val),
  the epoch best for val may not be best for test.

### 11. Balanced accuracy as the metric

- **What.** Mean of per-family recall. Chance = 1/10 = 0.10, regardless of class sizes.
- **Why.** Test families are unbalanced (210–440 notes). Plain accuracy rewards favouring big classes; balanced accuracy
  treats each family as equally important, which matches the question "does it know each family?".
- **Alternative discarded.** Plain accuracy (reported too, but secondary), macro-F1 (also fine; mixes in precision, harder to
  say out loud), top-k accuracy (useful for a "suggest a family" UI, not the question here).
- **What breaks.** Balanced accuracy hides precision: a model can get high organ recall by calling everything organ (the
  baseline's 0.85 recall, 0.25 precision). That is why per-family precision is reported next to it.

### 12. Variance across split seeds

- **What.** The baseline is refitted on 10 different `make_split` seeds; mean ± std of test balanced accuracy is reported.
  The CNN is trained once (seed 0).
- **Why.** The test set is 3,435 notes but only 10 instruments, one per family. Per-family recall is recall on *one*
  instrument. The effective sample size is closer to 10 than to 3,435, so a bootstrap over notes would understate the
  uncertainty. Redoing the split measures it directly.
- **Alternative discarded.** Bootstrap over notes (too optimistic, notes are correlated within instrument), retraining the CNN
  on 10 splits (~4 hours on CPU).
- **Result.** Baseline: 0.243 ± 0.062 over 10 seeds (0.158–0.329); seed 0 (the project's split) gives 0.188, one of the
  harder ones. Per family it is extreme: string recall 0.00–0.96, mallet 0.00–0.92.
- **What breaks.** The CNN's spread across splits is assumed similar to the baseline's; it could be larger (more capacity to
  memorise). A difference between CNN and baseline smaller than ~1 std of the seed spread is not evidence of anything. The
  CNN's +0.11 on the same split is ~1.8 std: suggestive, not proven.

## Part 2 — Questions an audio researcher would likely ask

Ordered from most to least likely. Each answer is meant to be said out loud in under 30 seconds.

**1. Walk me through the project. What did you find?**
I classify the instrument family of single NSynth notes and evaluate on instruments never seen in training. The headline is
that the split matters more than the model: 0.60 with a random note split, 0.19 for the baseline on unseen instruments, 0.30
for a small CNN. The error analysis shows the CNN partly uses shortcuts (production source and pitch), and the bottleneck is
having only a handful of instruments per family.

**2. 0.19 balanced accuracy sounds bad. Why isn't it that bad?**
Chance is 0.10, and the test asks for something hard: the family of an instrument the model has *never heard*, learned
from 1 to 9 instruments per family. On notes from known instruments the same model gets 0.60, so the 0.19 is the honest
number for generalisation. Two families (flute, vocal) are near zero by construction: trained on an acoustic instrument,
tested on a synthetic one.

**3. Why didn't you use the official NSynth split?**
I couldn't use `train` on my hardware (23.8 GB), and the official `valid` and `test` contain the *same* 53 instruments.
They were built to evaluate a model trained on `train`, not to be a train/test pair for each other. Using them that way
would put the same instruments on both sides, which is exactly the leakage I measured at 0.60 vs 0.19.

**4. Why should a CNN on log-mel beat MFCC statistics? Did it?**
MFCC mean/std throws away the order of events in time (a note and its reverse give the same features) and the fine spectral
detail above ~20 coefficients. The CNN sees the full time × frequency image, so it can use the attack, the decay and the
harmonic pattern. It did: 0.30 vs 0.19 on the same split. But it is one run on one split, and it fixed some errors while
creating others (mallet → keyboard got worse).

**5. What is an MFCC?**
Take the log-mel spectrum of a frame and apply a DCT along the frequency axis; keep the first ~20 coefficients. The low
coefficients describe the smooth shape of the spectrum (the envelope, a big part of timbre), while the fine harmonic detail
that mostly encodes pitch goes into the dropped high ones. They are compact and decorrelated, which suits linear models.

**6. How confident are you that the CNN is actually better?**
Moderately. On the same split it is +0.11, but the baseline alone moves from 0.16 to 0.33 (± 0.06) just by changing which 10
instruments are in test, and I trained the CNN once. The gain is ~1.8× that spread, so it is suggestive. To claim it, I would
train the CNN on 3–5 split seeds and compare paired differences.

**7. Is your model learning instrument family, or something else?**
Partly something else. Family and production source are confounded in NSynth (organ is all electronic, brass all
acoustic), and the CNN's predictions match the note's source more often than a source-blind model would (75% vs 63%), e.g.
synthetic vocal → bass. It also depends on pitch: above MIDI 72 its accuracy drops to ~0.15 and high notes get called bass.

**8. Why balanced accuracy and not accuracy?**
The families are unbalanced (210–440 test notes; in training, bass has ~25× more notes than vocal). Balanced
accuracy is the mean of per-class recall, so every family counts equally and chance is always 1/10. I report precision next
to it, because recall alone hides a model that dumps everything in one class, like the baseline's organ.

**9. Why didn't you use pitch shifting for augmentation?**
Naive pitch shifting moves the whole spectrum, including the formants, the body resonances that stay roughly fixed when a
real instrument plays higher. A shifted cello sounds like a smaller cello, so I'd be creating wrong timbres with right labels,
and timbre is exactly the signal. NSynth already has every instrument at up to 88 real pitches, so I used SpecAugment masks,
which hide information rather than fake it.

**10. How did you pick the number of epochs without touching the test set?**
I held out 7 training instruments as validation, one per family where the family had at least 3 (grouped, for the same
leakage reason as the test split). I stopped on validation balanced accuracy with patience 5. Loss would have chosen an
earlier epoch: validation loss rose while accuracy still improved, which is the model getting confidently wrong. The
validation set is small and excludes flute and vocal, so it picks an epoch but doesn't estimate test (0.58 vs 0.30).

**11. Why log-mel? Why not the raw waveform, a linear STFT or a CQT?**
Log-mel is the standard compact input: mel gives more resolution where harmonics are far apart, log compresses an ~80 dB
range and turns gain into an offset. A raw-waveform model would have to learn its own filterbank, which needs far more data
than 36 instruments. A CQT would be a good alternative because pitch becomes a shift; I'd try it next, but for a first model
log-mel is the safest choice.

**12. Why such a small model, and why not a pretrained one?**
The real number of examples is the number of instruments (36), not notes; a 99k-parameter CNN started overfitting them by
epoch ~4 (training accuracy up, validation loss up). A pretrained model like PANNs or AST has heard thousands of sounds and would very likely do better; it's the first
reference I'd add. I left it out on purpose because the goal was to understand every step of the pipeline on CPU.

**13. You cropped to 2 seconds. What did you lose?**
CPU budget: a full-length epoch took ~3.7 min, so only ~9 epochs fitted in 40 minutes. NSynth holds notes 3 s, so 2 s keeps
the attack and early decay, but I measured that 35–84% of the frames after 2 s still carry signal: the late decay, the end of
the sustain and the release. That may contribute to mallet ↔ keyboard confusion; it's the first thing I'd revert.

**14. How would you improve this with more time?**
First, more instruments per family from NSynth's `train` split; diversity is the bottleneck. Second, measure properly:
CNN on several split seeds, full 4 s input. Third, a pretrained embedding (PANNs/CLAP) with a linear probe as a strong
reference, and only then architecture changes like pooling over time only. And I would listen to the persistent errors first.

**15. How does this relate to real music, where instruments are mixed?**
Isolated, clean, single notes are the easy case. In a mix, instruments overlap in time and frequency, there are effects and
mastering, and the task becomes multi-label (several families present at once), or a job for source separation first. The
lessons transfer though: split by instrument, recording or song to avoid leakage, and check confounds like recording source.

## Part 3 — Glossary

- **STFT (Short-Time Fourier Transform).** Cut the signal into short overlapping windows and take a Fourier transform of
  each: a matrix of frequency × time. Its magnitude is the spectrogram. Here: 513 frequency bins × 251 frames per 4 s note.
- **Window / hop.** The window is how many samples go into each Fourier transform (1024 = 64 ms at 16 kHz → 15.6 Hz per
  bin); the hop is how far the window moves between frames (256 = 16 ms). Longer window = finer frequency, blurrier time.
  The window is tapered (Hann) so its edges don't create fake frequencies.
- **Mel scale.** A frequency scale fitted to human pitch perception: roughly linear below ~1 kHz, logarithmic above. A mel
  filterbank sums STFT bins with triangular filters, narrow at low Hz and wide at high Hz (513 → 128 bands).
- **Log-mel spectrogram.** The mel spectrogram's power in decibels (10·log10). Compresses an ~80 dB range so quiet
  harmonics are visible, and turns a gain (louder note) into an additive shift. The CNN's input.
- **MFCC (Mel-Frequency Cepstral Coefficients).** For each frame, the DCT of the log-mel spectrum, keeping the first ~20
  coefficients. A compact, decorrelated description of the spectral *envelope* (the broad shape, "colour"), discarding fine
  harmonic detail (much of the pitch). c0 ≈ loudness, c1 ≈ spectral tilt, higher ones ≈ finer bumps.
- **DCT (Discrete Cosine Transform).** Expresses a sequence as a sum of cosines of increasing frequency. Applied along the
  mel axis, the first coefficients capture slow variations across frequency (the envelope) and the later ones fast
  variations (harmonics). Like JPEG compression, but on a spectrum. Its output is decorrelated, which suits linear models.
- **SpecAugment.** Augmentation on the spectrogram itself (Park et al., 2019, from speech recognition): mask random bands of
  frequencies and random stretches of time (the original also warps time). Cheap, and forces the model to use several cues.
- **Formant.** A resonance of the instrument's body or the vocal tract: a peak in the spectral envelope at a roughly fixed
  frequency, regardless of the note played. The harmonics move with pitch; the formants don't. A large part of timbre.
  Naive pitch shifting moves them, which is why it changes the timbre.
- **Balanced accuracy.** The mean of per-class recall (recall = share of a class's examples predicted as that class). Every
  class counts equally whatever its size; chance level = 1 / number of classes (0.10 here).
- **Timbre.** What makes two sounds with the same pitch and loudness different: spectral envelope, harmonic vs inharmonic
  partials, attack and decay, noise components. The thing this classifier has to learn.
- **Harmonic / partial.** A partial is any sinusoidal component of a sound. Harmonic partials sit at integer multiples of the
  fundamental (strings, winds); struck bars and bells have inharmonic partials (e.g. 1, 2.76, 5.40 × f0 for the mallet in
  notebook 00).
- **Leakage.** Any path by which information from the test data reaches training or model selection (near-duplicate notes
  across the split, normalisation statistics computed on test, choosing the epoch on test). Makes the score optimistic.

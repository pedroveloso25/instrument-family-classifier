"""NSynth metadata and the train/test split.

The split is grouped by instrument: every note of a given instrument is either in train or in test,
never both. Run `python -m src.dataset` from the project root to (re)write the split file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLIT_FILE = ROOT / "splits" / "instruments.csv"
SR = 16_000


def load_metadata(raw_dir=RAW):
    """All notes from the NSynth `valid` and `test` folders in one table, one row per note."""
    frames = []
    for name in ["valid", "test"]:
        folder = Path(raw_dir) / f"nsynth-{name}"
        meta = json.loads((folder / "examples.json").read_text(encoding="utf-8"))
        df = pd.DataFrame(meta.values())
        df["path"] = [folder / "audio" / f"{note}.wav" for note in df["note_str"]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    columns = {
        "note_str": "note",
        "instrument_str": "instrument",
        "instrument_family_str": "family",
        "instrument_source_str": "source",
        "pitch": "pitch",
        "velocity": "velocity",
        "path": "path",
    }
    return df[list(columns)].rename(columns=columns)


def make_split(notes, seed=0):
    """One table row per instrument; one random instrument per family goes to test, the rest to train."""
    rng = np.random.default_rng(seed)
    instruments = (
        notes.groupby("instrument")
        .agg(family=("family", "first"), source=("source", "first"), n_notes=("note", "size"))
        .reset_index()
    )
    test = {rng.choice(sorted(group["instrument"])) for _, group in instruments.groupby("family")}
    instruments["split"] = np.where(instruments["instrument"].isin(test), "test", "train")
    return instruments


def make_val_split(instruments, seed=0, min_train_instruments=3):
    """Moves one random TRAIN instrument per family to "val", for early stopping (Phase 3).

    Only families with at least `min_train_instruments` training instruments give one up, so every family keeps
    at least two instruments in training. Test instruments are never touched.
    """
    rng = np.random.default_rng(seed)
    instruments = instruments.copy()
    train = instruments[instruments["split"] == "train"]
    val = {
        rng.choice(sorted(group["instrument"]))
        for _, group in train.groupby("family")
        if len(group) >= min_train_instruments
    }
    instruments.loc[instruments["instrument"].isin(val), "split"] = "val"
    return instruments


def with_validation(notes, seed=0):
    """Copy of `notes` (from `load_notes`) where the validation instruments have split == "val"."""
    instruments = notes.groupby("instrument").agg(family=("family", "first"), split=("split", "first")).reset_index()
    val = make_val_split(instruments, seed=seed)
    notes = notes.drop(columns="split").merge(val[["instrument", "split"]], on="instrument", validate="many_to_one")
    return notes


def load_notes(raw_dir=RAW, split_file=SPLIT_FILE):
    """All notes, with the `split` column taken from the saved split file."""
    notes = load_metadata(raw_dir)
    split = pd.read_csv(split_file)[["instrument", "split"]]
    notes = notes.merge(split, on="instrument", how="left", validate="many_to_one")
    assert notes["split"].notna().all(), "some instruments are missing from the split file"
    return notes


if __name__ == "__main__":
    instruments = make_split(load_metadata())
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    instruments.to_csv(SPLIT_FILE, index=False)
    print(instruments.groupby("split")["n_notes"].agg(["count", "sum"]).rename(columns={"count": "instruments", "sum": "notes"}))
    print(f"\nwritten to {SPLIT_FILE.relative_to(ROOT)}")

"""
preprocess.py
=============
Step 1-2 of the pipeline: collect MIDI data and turn it into note sequences
suitable for training an LSTM.

What it does:
  1. Walks a folder of .mid / .midi files (e.g. a folder of classical piano
     pieces, or a jazz corpus — any MIDI dataset works).
  2. Parses each file with music21, flattening it to a single stream of
     notes and chords in playing order.
  3. Encodes each note as "<midi_pitch>" and each chord as a dot-joined
     string of its pitches, e.g. "60.64.67" — this turns music into a
     vocabulary of "words" an LSTM can be trained on, the same way you'd
     tokenize text.
  4. Builds fixed-length input/output training windows: given the previous
     SEQUENCE_LENGTH notes, predict the next one.
  5. Saves everything (encoded sequences + vocabulary mappings) to disk so
     train.py doesn't need to re-parse MIDI files every run.

Usage:
  python preprocess.py --data_dir ./midi_songs --sequence_length 100

Where to get MIDI data:
  - Classical Piano MIDI Page (piano-midi.de)
  - MAESTRO dataset (magenta.tensorflow.org/datasets/maestro)
  - Lakh MIDI Dataset (colinraffel.com/projects/lmd) for multi-genre/jazz
  Download a folder of .mid files and point --data_dir at it.
"""

import argparse
import glob
import os
import pickle

import numpy as np
from music21 import chord, converter, instrument, note
from tqdm import tqdm


def extract_notes_from_file(file_path):
    """Parse one MIDI file into a flat list of note/chord tokens."""
    midi = converter.parse(file_path)

    try:
        parts = instrument.partitionByInstrument(midi)
        notes_stream = parts.parts[0].recurse() if parts else midi.flat.notes
    except Exception:
        notes_stream = midi.flat.notes

    tokens = []
    for element in notes_stream:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch.midi))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(p.midi) for p in element.pitches))
    return tokens


def collect_corpus(data_dir):
    """Parse every MIDI file in data_dir into one long list of tokens
    per song (kept separate so training windows never cross song
    boundaries)."""
    midi_files = glob.glob(os.path.join(data_dir, "**", "*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(data_dir, "**", "*.midi"), recursive=True)

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid/.midi files found under {data_dir}. "
            "Point --data_dir at a folder of MIDI files."
        )

    print(f"Found {len(midi_files)} MIDI files.")
    songs = []
    for path in tqdm(midi_files, desc="Parsing MIDI files"):
        try:
            tokens = extract_notes_from_file(path)
            if len(tokens) > 10:
                songs.append(tokens)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
    return songs


def build_vocabulary(songs):
    all_tokens = sorted({tok for song in songs for tok in song})
    token_to_int = {tok: i for i, tok in enumerate(all_tokens)}
    int_to_token = {i: tok for tok, i in token_to_int.items()}
    return token_to_int, int_to_token


def build_training_sequences(songs, token_to_int, sequence_length):
    """Slide a window of `sequence_length` tokens across each song to
    build (input_sequence -> next_token) training pairs."""
    network_input = []
    network_output = []

    for song in songs:
        encoded = [token_to_int[tok] for tok in song]
        for i in range(len(encoded) - sequence_length):
            network_input.append(encoded[i:i + sequence_length])
            network_output.append(encoded[i + sequence_length])

    return np.array(network_input), np.array(network_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder containing .mid/.midi files (searched recursively).")
    parser.add_argument("--sequence_length", type=int, default=100,
                         help="Number of previous notes the model sees before predicting the next one.")
    parser.add_argument("--out_dir", type=str, default="./processed",
                         help="Where to save the processed dataset.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    songs = collect_corpus(args.data_dir)
    token_to_int, int_to_token = build_vocabulary(songs)
    print(f"Vocabulary size: {len(token_to_int)} unique notes/chords")

    X, y = build_training_sequences(songs, token_to_int, args.sequence_length)
    print(f"Built {len(X)} training sequences of length {args.sequence_length}")

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump({
            "token_to_int": token_to_int,
            "int_to_token": int_to_token,
            "sequence_length": args.sequence_length,
        }, f)

    print(f"Saved processed dataset to {args.out_dir}/")


if __name__ == "__main__":
    main()

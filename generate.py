"""
generate.py
===========
Step 5 of the pipeline: use the trained model to generate a brand new note
sequence, then convert it back into a real .mid file with music21.

Usage:
  python generate.py --model_path ./checkpoints/final_model.keras \
                      --vocab_path ./processed/vocab.pkl \
                      --num_notes 300 --temperature 1.0 \
                      --out_file generated_song.mid

Playback:
  - Any MIDI file can be opened directly in GarageBand, MuseScore, VLC,
    or a DAW.
  - To render it to a listenable .wav instead, install FluidSynth and a
    soundfont, then:
        pip install midi2audio --break-system-packages
        from midi2audio import FluidSynth
        FluidSynth('/path/to/soundfont.sf2').midi_to_audio('generated_song.mid', 'generated_song.wav')
"""

import argparse
import pickle

import numpy as np
from music21 import chord, instrument, note, stream
from tensorflow import keras


def sample_with_temperature(probabilities, temperature=1.0):
    """Lower temperature -> more conservative/predictable output.
    Higher temperature -> more random/adventurous output."""
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-9) / temperature
    exp_probs = np.exp(probabilities)
    probabilities = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(probabilities), p=probabilities)


def generate_sequence(model, seed_sequence, int_to_token, num_notes, sequence_length, temperature):
    pattern = list(seed_sequence)
    generated_tokens = []

    for _ in range(num_notes):
        input_seq = np.array(pattern[-sequence_length:]).reshape(1, sequence_length)
        prediction = model.predict(input_seq, verbose=0)[0]
        next_id = sample_with_temperature(prediction, temperature)

        generated_tokens.append(int_to_token[next_id])
        pattern.append(next_id)

    return generated_tokens


def tokens_to_midi(tokens, out_file, note_duration=0.5):
    """Convert a list of note/chord tokens (e.g. '60' or '60.64.67')
    back into a music21 stream and write it out as a .mid file."""
    output_notes = []
    offset = 0.0

    for token in tokens:
        if "." in token:
            pitches = [int(p) for p in token.split(".")]
            new_chord = chord.Chord(pitches)
            new_chord.offset = offset
            new_chord.storedInstrument = instrument.Piano()
            output_notes.append(new_chord)
        else:
            new_note = note.Note(int(token))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += note_duration

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--num_notes", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="0.5-0.8 = safer/more repetitive, 1.0-1.3 = more varied/surprising.")
    parser.add_argument("--out_file", type=str, default="generated_song.mid")
    args = parser.parse_args()

    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    int_to_token = vocab["int_to_token"]
    token_to_int = vocab["token_to_int"]
    sequence_length = vocab["sequence_length"]

    model = keras.models.load_model(args.model_path)

    # Random seed sequence to kick off generation; swap this for a
    # real excerpt from your training data if you want a more
    # controlled/consistent opening.
    seed = list(np.random.randint(0, len(token_to_int), size=sequence_length))

    print(f"Generating {args.num_notes} notes (temperature={args.temperature})...")
    tokens = generate_sequence(
        model, seed, int_to_token,
        num_notes=args.num_notes,
        sequence_length=sequence_length,
        temperature=args.temperature,
    )

    tokens_to_midi(tokens, args.out_file)
    print(f"Saved generated music to {args.out_file}")


if __name__ == "__main__":
    main()

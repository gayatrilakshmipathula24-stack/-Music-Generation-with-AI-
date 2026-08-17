# AI Music Generator (LSTM + music21)

Generates new piano music by training an LSTM on a corpus of MIDI files,
then sampling new note sequences from the trained model and writing them
back out as a playable `.mid` file.

## Pipeline

```
midi_songs/ (your dataset)
     │  preprocess.py  — parse MIDI with music21, encode notes/chords, build training windows
     ▼
processed/ (X.npy, y.npy, vocab.pkl)
     │  train.py — train a stacked LSTM to predict the next note
     ▼
checkpoints/final_model.keras
     │  generate.py — sample a new sequence, convert back to MIDI
     ▼
generated_song.mid
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`tensorflow` benefits a lot from a GPU for training — on CPU, expect
training to be slow for anything beyond a small corpus.

## 2. Get MIDI data

Pick a genre and download a folder of `.mid` files into `./midi_songs/`:

- **Classical / piano**: [Classical Piano MIDI Page](http://www.piano-midi.de/), or Google Magenta's [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (~1,300 piano performances)
- **Jazz / multi-genre**: [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)

A few dozen files is enough to see the pipeline work end to end; a few
hundred+ is where the output starts sounding musically coherent.

## 3. Preprocess

```bash
python preprocess.py --data_dir ./midi_songs --sequence_length 100
```

This parses every MIDI file, extracts notes and chords in playing order
with `music21`, encodes them into a vocabulary (like tokenizing text),
and slices the corpus into (100 previous notes → next note) training
windows.

## 4. Train

```bash
python train.py --data_dir ./processed --epochs 60 --batch_size 64
```

The model is a stacked LSTM: an embedding layer, two LSTM layers with
dropout, and a softmax output over the note/chord vocabulary. Checkpoints
save automatically whenever training loss improves.

*(A GAN — e.g. a generator/discriminator pair over piano-roll images —
is a valid alternative architecture for this task and tends to produce
more "creative" but less stable output. LSTMs are the more standard and
easier-to-train starting point, so that's what's implemented here; the
`model.py` file is where you'd swap in a GAN if you want to extend this.)*

## 5. Generate new music

```bash
python generate.py \
  --model_path ./checkpoints/final_model.keras \
  --vocab_path ./processed/vocab.pkl \
  --num_notes 300 \
  --temperature 1.0 \
  --out_file generated_song.mid
```

`--temperature` controls how adventurous the output is: `0.5–0.8` stays
closer to patterns it saw often in training; `1.0–1.3` takes more risks
and can wander further from the training data.

## 6. Listen

- Open `generated_song.mid` directly in GarageBand, MuseScore, VLC, or
  any DAW.
- To render it to a `.wav` you can play anywhere, install
  [FluidSynth](https://www.fluidsynth.org/) plus a soundfont, then:

  ```python
  from midi2audio import FluidSynth
  FluidSynth('/path/to/soundfont.sf2').midi_to_audio('generated_song.mid', 'generated_song.wav')
  ```

## Notes on this implementation

- This code was written and syntax-checked in an environment without
  internet access, so `music21` and `tensorflow` could not be installed
  or run here to produce an actual trained model or sample output. It's
  built to run in your own environment (`pip install -r requirements.txt`)
  against a real MIDI dataset.
- Training quality scales with data: a handful of MIDI files will
  produce a working pipeline but repetitive/simplistic output; a
  few hundred files from one consistent style (e.g. all-Chopin, or
  all-jazz-standards) will generate noticeably more idiomatic music.
- `sequence_length`, `lstm_units`, and `epochs` are the main knobs to
  tune if output sounds too random (increase epochs / data) or too
  repetitive (increase temperature at generation time).

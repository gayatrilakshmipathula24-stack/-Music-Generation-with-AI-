"""
model.py
========
Step 3 of the pipeline: the deep learning model that learns musical
patterns. This is a stacked LSTM sequence model — given a window of
previous notes/chords, it predicts a probability distribution over what
comes next. (A GAN is a valid alternative approach, but LSTMs are the
more standard and more stable starting point for symbolic music
generation, so that's what's implemented here.)
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size, sequence_length, embedding_dim=100, lstm_units=256):
    """
    Architecture:
      Embedding      -> turns each note/chord id into a dense vector
      LSTM (x2)      -> learn short- and longer-range musical patterns
      Dropout        -> reduce overfitting on a limited MIDI corpus
      Dense + Softmax -> probability distribution over the next note/chord
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(lstm_units),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model

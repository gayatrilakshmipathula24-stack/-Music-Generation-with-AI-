"""
train.py
========
Step 4 of the pipeline: train the LSTM on the preprocessed note sequences.

Usage:
  python train.py --data_dir ./processed --epochs 60 --batch_size 64
"""

import argparse
import os
import pickle

import numpy as np
from tensorflow import keras

from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./processed",
                         help="Folder produced by preprocess.py (contains X.npy, y.npy, vocab.pkl).")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    with open(os.path.join(args.data_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab["token_to_int"])
    sequence_length = vocab["sequence_length"]

    print(f"Training on {len(X)} sequences | vocab size {vocab_size} | seq length {sequence_length}")

    model = build_model(vocab_size=vocab_size, sequence_length=sequence_length)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, "model_epoch{epoch:02d}_loss{loss:.4f}.keras"),
            monitor="loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4),
    ]

    model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(args.checkpoint_dir, "final_model.keras")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()

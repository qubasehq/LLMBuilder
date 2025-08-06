#!/bin/bash

# TODO:
# - Automate full pipeline: preprocess -> tokenizer -> training
# - Add CLI arguments to run specific stages

echo "[1/3] Preprocessing data..."
python training/preprocess.py

echo "[2/3] Training tokenizer..."
python training/train_tokenizer.py

echo "[3/3] Training model..."
python training/train.py

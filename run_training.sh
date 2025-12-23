#!/bin/bash
echo "Starting Phase 1: Training top layers..."
poetry run python train_phase1.py
if [ $? -ne 0 ]; then
    echo "Phase 1 failed!"
    exit 1
fi

echo "Phase 1 complete. Starting Phase 2: Fine-tuning..."
poetry run python train_phase2_finetune.py
if [ $? -ne 0 ]; then
    echo "Phase 2 failed!"
    exit 1
fi

echo "Training complete!"

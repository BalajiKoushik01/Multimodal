# Training Guide

The training process consists of two stages: **Self-Supervised Pre-training (SSL)** and **Supervised Fine-tuning**.

## 1. Configuration
All hyperparameters are managed in `configs/config.yaml`. Key parameters include:
*   `batch_size`: Number of images per batch (Default: 8).
*   `learning_rate`: Step size for optimizer (Default: 0.0001).
*   `max_epochs`: Total training rounds (Default: 100).
*   `dataset_path`: Root directory for data.

## 2. Self-Supervised Pre-training
This stage aligns the Colposcopy and Histopathology feature spaces without requiring labels.

**Command:**
```bash
python scripts/train_ssl.py --config configs/ssl_config.yaml
```
*   **Output:** Saves pretrained weights to `checkpoints/`.
*   **Goal:** Minimize contrastive loss between domains.

## 3. Fine-tuning
This stage trains the classification heads (Malignancy and Severity) using the pretrained backbone.

**Command:**
```bash
python scripts/train_finetune.py --config configs/finetune_config.yaml
```
*   **Input:** Automatically loads the latest checkpoint from `checkpoints/`.
*   **Output:** A fully trained classification model.

## 4. Monitoring
Training logs are saved to `train_final.log`. You can view loss curves and metrics in the log file or terminal output.

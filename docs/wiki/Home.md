# Welcome to the MultiModel Cancer Detection Wiki

This project implements a state-of-the-art **Cross-Domain Cervical Cancer Analysis Pipeline**. It leverages Self-Supervised Learning (SSL) to bridge the gap between colposcopy (visual inspection) and histopathology (biopsy) images, enabling robust detection with limited labeled data.

## Key Features
*   **Dual-Domain Learning:** Simultaneously learns from macroscopic (colpo) and microscopic (histo) views.
*   **Self-Supervised Pre-training:** Uses unlabelled data to learn powerful feature representations.
*   **Weakly Supervised Attention:** Highlights suspicious lesion areas automatically.
*   **End-to-End Pipeline:** From data ingestion to malignancy and severity prediction.

## Navigation
*   **[[Installation]]:** Setup your environment and dependencies.
*   **[[Data Preparation]]:** Learn how to structure AnnoCerv and MobileODT datasets.
*   **[[Training]]:** Instructions for pre-training (SSL) and fine-tuning.
*   **[[Model Architecture]]:** Deep dive into the Dual-Encoder and Attention mechanisms.

## Quick Start
```bash
# Clone the repo
git clone https://github.com/BalajiKoushik01/multimodel.git
cd multimodel

# Install dependencies
pip install -r requirements.txt
```

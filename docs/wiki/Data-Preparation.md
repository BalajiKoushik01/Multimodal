# Data Preparation

This pipeline utilizes three primary data sources:

1.  **MobileODT (Colposcopy):** Large-scale dataset from Kaggle.
2.  **AnnoCerv (Colposcopy + Histopathology):** Mock/Real dataset for dual-domain learning.
3.  **Histopathology (Mock):** Supplemental microscopic images.

## Directory Structure
The system expects the following directory layout under `data/`:

```
data/
├── mobileodt/          # Downloaded from Kaggle
├── annocerv_real/      # AnnoCerv dataset
│   ├── colposcopy/
│   └── histopathology/
├── histopathology/     # Standalone histopathology data
└── dataset.py          # PyTorch dataset implementation
```

## Downloading Datasets

### MobileODT
We provide a script to automatically download this from Kaggle.
1.  Ensure you have `kaggle.json` in `~/.kaggle/` (See [[Installation]]).
2.  Run:
    ```bash
    python scripts/download_mobileodt.py
    ```

### Custom Data
To use your own data:
1.  Place images in a new folder under `data/`.
2.  Ensure your filenames or a CSV file contains labels if you are fine-tuning.
3.  Modify `configs/config.yaml` to point `dataset_path` to your new root if needed.

## Preprocessing
*   **Resize:** All images are resized to 256x256 via `transforms.Resize`.
*   **Augmentation:** During training, we apply random flips, rotations, and color jitter. This is configurable in `configs/config.yaml`.

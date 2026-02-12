# Model Architecture

The MultiModel pipeline uses a **Dual-Encoder** design to process multi-modal medical images.

## High-Level Overview
```mermaid
graph TD
    A[Colposcopy Image] --> B[Encoder A (ResNet50)]
    C[Histopathology Image] --> D[Encoder B (ResNet50)]
    B --> E[Projection Head]
    D --> E
    E --> F{Contrastive Loss}
    B --> G[Attention Module]
    G --> H[Malignancy Head]
    G --> I[Severity Head]
```

## Components

### 1. Dual Backbones
We use **ResNet50** as the backbone for both domains.
*   **Colpo Encoder:** Extracts macroscopic features (lesion shape, color).
*   **Histo Encoder:** Extracts microscopic features (cell density, atypia).

### 2. Attention Module
The weakly supervised attention mechanism computes a spatial attention map from the convolutional feature maps.
*   **Purpose:** Highlight the specific region of the image that contributed most to the prediction.
*   **Visualizing:** You can inspect these maps using `scripts/visualize_attention.py`.

### 3. Prediction Heads
*   **Malignancy Head:** Binary classification (Benign vs. Malignant).
*   **Severity Head:** Multi-class classification (Low, Moderate, Severe).

## Loss Functions
*   **SSL Phase:** Contrastive Loss (SimCLR style) to maximize similarity between related pairs.
*   **Fine-tuning:** Cross-Entropy Loss for classification tasks.

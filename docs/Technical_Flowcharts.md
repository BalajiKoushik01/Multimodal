# Technical Implementation Flowcharts

## System Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├──────────────────────────┬──────────────────────────────────────────┤
│   Colposcopy Images      │    Histopathology Images                 │
│      (224×224×3)         │         (224×224×3)                      │
└──────────┬───────────────┴──────────────┬───────────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│  Vision Transformer      │   │  Vision Transformer      │
│  + ASPP Module           │   │  + ASPP Module           │
│                          │   │                          │
│  • 12 Transformer Layers │   │  • 12 Transformer Layers │
│  • 768-dim Features      │   │  • 768-dim Features      │
│  • Multi-scale ASPP:     │   │  • Multi-scale ASPP:     │
│    - Dilation 6 (cell)   │   │    - Dilation 6 (cell)   │
│    - Dilation 12 (tissue)│   │    - Dilation 12 (tissue)│
│    - Dilation 18 (tissue)│   │    - Dilation 18 (tissue)│
│    - Dilation 24 (lesion)│   │    - Dilation 24 (lesion)│
└──────────┬───────────────┘   └──────────┬───────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
           ┌──────────────────────────────┐
           │  Cross-Modal Attention Fusion │
           │                               │
           │  • 8-head Attention          │
           │  • Residual Connections      │
           │  • Layer Normalization       │
           └──────────────┬────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │    Fused Feature Vector      │
           │         (2048-dim)            │
           └──────────────┬────────────────┘
                          │
           ┌──────────────┴────────────────┐
           │                               │
           ▼                ▼              ▼
┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
│  Malignancy      │ │ CIN Severity│ │ Progression Risk │
│  Classification  │ │   Grading   │ │    Prediction    │
│                  │ │             │ │                  │
│  Binary:         │ │ 4-class:    │ │ Continuous:      │
│  • Benign        │ │ • CIN 0     │ │ • P ∈ [0,1]      │
│  • Malignant     │ │ • CIN I     │ │                  │
│                  │ │ • CIN II    │ │                  │
│                  │ │ • CIN III   │ │                  │
└──────────────────┘ └─────────────┘ └──────────────────┘
```

---

## Training Pipeline Flowchart

### Stage 1: Self-Supervised Pre-Training (100 Epochs)

```
┌─────────────────────────────────────────────────────────┐
│              12,437 Unlabeled Images                     │
│         (Data1: 754 + Data2: 11,683)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Medical-Aware Augmentation Pipeline             │
│  • Elastic Deformation (tissue-preserving)              │
│  • Stain Normalization (Macenko method)                 │
│  • Illumination Correction (CLAHE)                      │
│  • Random Rotation (±15°)                               │
│  • Color Jittering (brightness, contrast)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Generate Two Augmented Views                   │
│              View 1  ←→  View 2                         │
│         (Same image, different augmentations)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         ViT+ASPP Feature Extraction                     │
│              (ResNet50 backbone)                         │
│         Output: 2048-dim feature vectors                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Projection Head (MLP)                          │
│         2048-dim → 4096-dim → 256-dim                   │
│              (Embedding space)                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Multimodal Contrastive Loss                     │
│                                                          │
│  L = -log(exp(sim(z₁,z₂)/τ) / Σₖ exp(sim(z₁,zₖ)/τ))   │
│                                                          │
│  • Maximize similarity between augmented views          │
│  • Minimize similarity with other images                │
│  • Temperature τ = 0.07                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Backpropagation & Weight Update                 │
│  • Optimizer: AdamW (lr=1e-3, wd=1e-4)                  │
│  • Mixed Precision (FP16)                               │
│  • Gradient Accumulation (4 steps)                      │
│  • Cosine LR Schedule + Warmup                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Checkpoint Saving                           │
│  • Every 5 epochs                                       │
│  • Best validation loss                                 │
│  • Model state + optimizer state                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Repeat for 100 epochs
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Pre-trained Model Checkpoint                     │
│      (Learned visual representations)                    │
└─────────────────────────────────────────────────────────┘
```

### Stage 2: Supervised Fine-Tuning (50 Epochs)

```
┌─────────────────────────────────────────────────────────┐
│      Load Pre-trained Model from Stage 1                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         ~1,300 Labeled Images                           │
│  • Malignancy labels (Normal/Abnormal)                  │
│  • CIN severity grades (0/I/II/III)                     │
│  • Progression risk scores                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│    Stratified Train/Val/Test Split (70/15/15)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Phase 1: Freeze Backbone (10 epochs)            │
│  • Train only task-specific heads                       │
│  • Learning rate: 1e-4                                  │
│  • Batch size: 16                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Phase 2: Gradual Unfreezing (40 epochs)         │
│  • Unfreeze backbone layers progressively               │
│  • Lower learning rate: 1e-5                            │
│  • Fine-tune entire model                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Multi-Task Learning                           │
│                                                          │
│  L_total = α·L_malignancy + β·L_severity + γ·L_risk     │
│                                                          │
│  • L_malignancy: Binary cross-entropy                   │
│  • L_severity: Categorical cross-entropy                │
│  • L_risk: Mean squared error                           │
│  • Weights: α=1.0, β=0.8, γ=0.5                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Validation & Early Stopping                      │
│  • Evaluate on validation set every epoch               │
│  • Track AUC-ROC, sensitivity, specificity              │
│  • Stop if no improvement for 10 epochs                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Final Trained Model                              │
│  • Best validation AUC checkpoint                       │
│  • Ready for deployment                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow During Inference

```
Patient Examination
        │
        ▼
┌──────────────────┐
│ Acquire Images   │
│ • Colposcopy     │
│ • Histopathology │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │
│ • Resize 224×224 │
│ • Normalize      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Inference │
│ • Forward pass   │
│ • <1 second      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Multi-Task Predictions          │
│  ┌────────────────────────────┐  │
│  │ Malignancy: 87% Abnormal   │  │
│  │ Severity: CIN II (72%)     │  │
│  │ Progression Risk: 0.65     │  │
│  └────────────────────────────┘  │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Attention Maps   │
│ (Interpretability)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Clinical Report  │
│ • Predictions    │
│ • Confidence     │
│ • Visualizations │
│ • Recommendations│
└──────────────────┘
         │
         ▼
  Physician Review
  Treatment Decision
```

---

## GPU Optimization Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              Training Optimization Stack                 │
└─────────────────────────────────────────────────────────┘

Level 1: Hardware
├─ NVIDIA GTX 1650 (4.29 GB VRAM)
├─ CUDA 11.8+
└─ cuDNN 8.x

Level 2: Framework Optimizations
├─ torch.backends.cudnn.benchmark = True
├─ torch.backends.cuda.matmul.allow_tf32 = True
└─ torch.backends.cudnn.allow_tf32 = True

Level 3: Memory Optimizations
├─ Mixed Precision Training (AMP)
│  ├─ FP16 for forward/backward
│  ├─ FP32 for weight updates
│  └─ 50% memory reduction
├─ Gradient Accumulation
│  ├─ Batch size: 12
│  ├─ Accumulation steps: 4
│  └─ Effective batch: 48
└─ Gradient Checkpointing (optional)

Level 4: Data Loading Optimizations
├─ Num Workers: 4 (parallel loading)
├─ Pin Memory: True (faster CPU→GPU)
├─ Persistent Workers: True (stay alive)
├─ Prefetch Factor: 2 (prefetch batches)
└─ Image Caching: 1000 images in RAM

Level 5: Training Optimizations
├─ Learning Rate Schedule
│  ├─ Warmup: 10 epochs (0 → 1e-3)
│  └─ Cosine Annealing: 90 epochs
├─ Optimizer: AdamW
│  ├─ β₁ = 0.9, β₂ = 0.999
│  └─ Weight decay: 1e-4
└─ Regularization
   ├─ Dropout: 0.3
   └─ Label Smoothing: 0.1

Result: 95-100% GPU Utilization
        ~1.37 seconds/batch
        ~15-20 minutes/epoch
```

---

## Performance Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────┐
│              Training Metrics (Real-time)                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Epoch: 0/100                Progress: ████░░░░░░ 5%   │
│  Batch: 16/319               ETA: 7m 15s               │
│                                                          │
│  Loss: 2.74 ↓                Best: 2.74                │
│  Learning Rate: 1.00e-04     GPU Util: 98%             │
│  GPU Memory: 3.6/4.3 GB      Temperature: 72°C         │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Loss Curve (Last 100 batches)            │   │
│  │  5.0 ┤                                           │   │
│  │  4.0 ┤●                                          │   │
│  │  3.0 ┤ ●●●                                       │   │
│  │  2.7 ┤     ●●●●●●●●●●●●●●●                      │   │
│  │  2.0 ┤                                           │   │
│  │      └────────────────────────────────────────→ │   │
│  │         0        50       100      150      200  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  Checkpoints: outputs/ssl_worldclass/checkpoints/       │
│  Logs: outputs/ssl_worldclass/logs/metrics.json        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

# Multimodal Deep Learning for Cervical Cancer Detection

A state-of-the-art deep learning framework for automated cervical cancer screening using multimodal medical imaging data (colposcopy and histopathology).

## 🎯 Project Overview

This project implements a Vision Transformer-based multimodal deep learning system for cervical cancer detection, addressing critical challenges in early diagnosis and screening, particularly in resource-constrained healthcare settings.

### Key Features

- **Multimodal Architecture**: Fuses colposcopy and histopathology images using cross-modal attention
- **Vision Transformer Backbone**: ViT with Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- **Self-Supervised Pre-training**: Leverages 12,437+ unlabeled images using SimCLR v2, MoCo v3, BYOL, and DINO strategies
- **Multi-Task Learning**: Simultaneous prediction of malignancy, severity (CIN grading), and progression risk
- **Medical-Aware Augmentation**: Specialized augmentation pipeline preserving clinical features

## 📊 Dataset

The system is trained on **12,437 cervical cancer images** from multiple sources:

- **Data1**: 754 images (Normal/Abnormal classification)
- **Data2**: 11,683 images (Kaggle cervical cancer dataset)
- **AnnoCerv**: Clinical annotations for CIN grading
- **MobileODT**: Intel MobileODT cervical cancer screening dataset
- **Expert Eyes**: Colposcopy images with expert annotations

> **Note**: Dataset files are not included in this repository due to size and privacy constraints. See [Data Preparation](#data-preparation) for download instructions.

## 🏗️ Architecture

### System Pipeline

```
Input Images → ViT+ASPP Feature Extraction → Cross-Modal Attention → Multi-Task Prediction
```

### Key Components

1. **Multi-Scale ASPP**: Captures features at different scales
   - Cellular level (dilation=6)
   - Tissue level (dilation=12, 18)
   - Lesion level (dilation=24)

2. **Cross-Modal Attention**: Fuses complementary information from colposcopy and histopathology

3. **Self-Supervised Pre-Training**: Learns robust representations from unlabeled data
   - SimCLR v2: Contrastive learning with strong augmentations
   - MoCo v3: Momentum contrast with queue mechanism
   - BYOL: Bootstrap your own latent without negative pairs
   - DINO: Self-distillation with no labels

4. **Multi-Task Head**: Predicts multiple clinical outcomes
   - Malignancy classification (binary)
   - Severity grading (4-class CIN: Normal, CIN1, CIN2, CIN3)
   - Progression risk estimation (regression)

## 📁 Project Structure

```
multimodal/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── advanced_augmentation.py # Medical-aware augmentation
│   ├── data_loader_v2.py        # Data loading utilities
│   ├── dataset.py               # PyTorch dataset classes
│   └── preprocess_data.py       # Data preprocessing
├── models/
│   ├── attention_modules.py     # Attention mechanisms
│   ├── backbone.py              # ViT+ASPP backbone
│   ├── dual_encoder.py          # Multimodal encoder
│   └── ssl_strategies.py        # Self-supervised learning
├── scripts/
│   ├── train_ssl.py             # Self-supervised pre-training
│   ├── train_finetune.py        # Supervised fine-tuning
│   ├── train_worldclass.py      # Full training pipeline
│   ├── evaluate_model.py        # Model evaluation
│   ├── run_inference.py         # Inference script
│   ├── visualize_embeddings.py  # Embedding visualization
│   └── visualize_metrics.py     # Metrics visualization
├── training/
│   └── ssl_trainer.py           # SSL training utilities
├── utils/
│   ├── external_data.py         # External data utilities
│   └── losses.py                # Custom loss functions
├── docs/
│   ├── Technical_Flowcharts.md  # Technical architecture
│   ├── research_proposal.md     # Research proposal
│   └── GPU_Optimization_Guide.md # GPU optimization tips
├── checkpoints/                 # Trained model checkpoints
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- PyTorch 2.0+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BalajiKoushik01/Multimodal.git
   cd Multimodal
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

Due to size constraints, datasets are not included. Download them separately:

1. **Kaggle Dataset**: Download from [Kaggle Cervical Cancer Competition](https://www.kaggle.com/competitions/intel-mobileodt-cervical-cancer-screening)
2. **AnnoCerv**: Request access from the AnnoCerv project
3. **Place datasets** in the `data/` directory following this structure:
   ```
   data/
   ├── annocerv_real/
   ├── mobileodt/
   ├── colposcopy/
   └── histopathology/
   ```

## 🎓 Training

### Self-Supervised Pre-Training

```bash
python scripts/train_ssl.py --config configs/config.yaml --epochs 100
```

### Supervised Fine-Tuning

```bash
python scripts/train_finetune.py --config configs/config.yaml --checkpoint checkpoints/checkpoint_epoch_100.pth
```

### Full Training Pipeline

```bash
python scripts/train_worldclass.py --config configs/config.yaml
```

## 📈 Evaluation

```bash
python scripts/evaluate_model.py --checkpoint checkpoints/finetuned_epoch_100.pth
```

## 🔬 Results

### Expected Performance Targets

- **HSIL Sensitivity**: >95%
- **Malignancy AUC-ROC**: >0.95
- **Severity Accuracy**: >85% (4-class CIN)
- **Progression Risk AUC**: >0.92
- **Turnaround Time**: <24 hours (vs weeks in traditional screening)

### Clinical Impact

- Real-time triage at Primary Health Centres
- Reduced diagnostic variability (30% inter-observer variation → <10%)
- Faster treatment decisions
- Improved access in resource-constrained settings

## 📚 Documentation

- [Technical Flowcharts](docs/Technical_Flowcharts.md) - Detailed architecture diagrams
- [Research Proposal](docs/research_proposal.md) - Academic research proposal
- [GPU Optimization Guide](docs/GPU_Optimization_Guide.md) - Performance optimization tips

## 🛠️ Configuration

Edit `configs/config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Loss function weights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

**Balaji Koushik**
- GitHub: [@BalajiKoushik01](https://github.com/BalajiKoushik01)

## 🙏 Acknowledgments

- Vision Transformer implementation inspired by Dosovitskiy et al. (2020)
- Self-supervised learning strategies based on SimCLR, MoCo, BYOL, and DINO
- Dataset providers: Kaggle, AnnoCerv project, Intel MobileODT

## 📖 References

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
3. He et al. (2020). "Momentum Contrast for Unsupervised Visual Representation Learning." CVPR 2020.
4. Caron et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
5. Chen et al. (2018). "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.

## ⚠️ Disclaimer

This is a research project and should not be used for clinical diagnosis without proper validation and regulatory approval.

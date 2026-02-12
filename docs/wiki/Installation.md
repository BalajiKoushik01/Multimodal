# Installation Guide

Follow these steps to set up the Multimodal Cervical Cancer Analysis environment.

## Prerequisites
*   **OS:** Windows / Linux / macOS
*   **Python:** 3.8+
*   **CUDA:** 11.x (Recommended for GPU acceleration)

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/BalajiKoushik01/multimodel.git
cd multimodel
```

### 2. Create a Virtual Environment (Recommended)
It is best practice to use a virtual environment to manage dependencies.

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
Run the following command to check if PyTorch is detecting your GPU:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting
*   **Kaggle API Error:** If seeing `kaggle.json not found`, ensure you have placed your API key in `~/.kaggle/kaggle.json`. See [[Data Preparation]] for details.
*   **CUDA OOM:** If you encounter Out Of Memory errors, try reducing the `batch_size` in `configs/config.yaml`.

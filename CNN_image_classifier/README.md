# CNN Image Classifier - CIFAR-10

Image classification using Convolutional Neural Networks on the CIFAR-10 dataset.

## Hardware & Acceleration

### What is CUDA?

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and programming model developed by NVIDIA. It allows software to use NVIDIA GPUs for general purpose processing (not just graphics), dramatically accelerating deep learning computations.

**Key Benefits:**
- **Speed**: Neural network training that takes hours on CPU can complete in minutes on GPU
- **Parallelism**: GPUs have thousands of cores designed for parallel operations, perfect for matrix computations in deep learning
- **Efficiency**: Modern deep learning frameworks (TensorFlow, PyTorch) automatically leverage CUDA when available

### System Hardware

- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **Architecture**: Blackwell (Compute Capability 12.0)
- **CUDA Version**: 13.0

### Current Status: CPU Mode

⚠️ **Important**: The RTX 5080 is very new (released 2025) and current stable versions of TensorFlow (2.20.0) and PyTorch (2.6.0) don't yet fully support compute capability 12.0. The project is configured to run on CPU for now.

**Alternatives for GPU training:**
- **Google Colab**: Free T4/A100 GPUs with full library support
- **Wait**: TensorFlow 2.21+ / PyTorch 2.7+ will likely add Blackwell support
- **Note**: CPU training for CIFAR-10 (32x32 images) is actually quite fast on modern CPUs

---

## Project Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Jupyter Kernel
```bash
python -m ipykernel install --user --name=cnn_classifier --display-name="CNN Classifier (Python 3.12)"
```

### 5. Use the Notebook
1. Launch Jupyter: `jupyter notebook` or `jupyter lab`
2. In the notebook, go to **Kernel → Change Kernel → CNN Classifier (Python 3.12)**
3. Add `import setup_gpu` at the start to configure TensorFlow
4. Run the cells - TensorFlow will use CPU mode

---

## Project Structure
```
CNN_image_classifier/
├── venv/                      # Virtual environment (not committed)
├── data/                      # Dataset
├── cnn_cifar10.ipynb         # Main notebook
├── setup_gpu.py              # GPU configuration for RTX 5080
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                  # This file
```

## Key Dependencies
- TensorFlow 2.20.0 (with CUDA support)
- NumPy
- Pandas
- Matplotlib
- Jupyter
- NVIDIA CUDA libraries (installed automatically with TensorFlow)

## Notes
- The virtual environment (`venv/`) is not committed to Git
- Always activate the venv when working on the project: `source venv/bin/activate`
- First GPU run may be slower due to CUDA kernel compilation for compute capability 12.0
- GPU memory growth is enabled to prevent OOM errors

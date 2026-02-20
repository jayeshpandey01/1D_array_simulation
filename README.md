# RNN - Recurrent Neural Networks & Signal Processing

A comprehensive collection of RNN implementations, signal processing utilities, and deep learning models for time series analysis, video classification, and regression tasks.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Source Modules](#source-modules)
  - [Linear Regression Module](#linear-regression-module)
  - [Signal Generation Module](#signal-generation-module)
- [Notebooks & Experiments](#notebooks--experiments)
- [Performance Benchmarks](#performance-benchmarks)
- [Usage Examples](#usage-examples)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This repository contains:
- **Production-ready PyTorch models** for regression and time series analysis
- **Signal generation utilities** for creating synthetic datasets
- **RNN/GRU/LSTM implementations** from scratch
- **Video classification** using CRNN architectures
- **Comprehensive tutorials** and Jupyter notebooks

---

## 📁 Project Structure

```
RNN-main/
│
├── src/
│   ├── linear_regression.py       # Advanced PyTorch regression model
│   ├── multiple_signal.py         # Signal generation utilities
│   └── RNN.py                      # RNN implementations
│
├── Notebooks/
│   ├── 01- RNN_Classification.ipynb
│   ├── 02- RNN_Regression.ipynb
│   ├── 03- RNN_vs_GRU_Classification.ipynb
│   ├── 04- RNN_vs_GRU_Regression.ipynb
│   ├── 05- Ball_Move_Data_Generation.ipynb
│   ├── 06- GRU_Implementation_from_Scratch.ipynb
│   ├── 07- LSTM_Implementation_from_Scratch.ipynb
│   ├── 08- Ball_Move_Direction_Classification.ipynb
│   └── create_dataset.ipynb
│
├── 09- VideoClassificationCRNN/
│   ├── 09- Video_Classification_CRNN.ipynb
│   ├── inference.py
│   ├── models.py
│   ├── load_video.py
│   └── requirements.txt
│
├── 10- Video_Classification_CRNN/
│   ├── 10- Video_Classification_CRNN.ipynb
│   ├── models.py
│   └── README.md
│
└── README.md
```

---

## 🚀 Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RNN-main.git
cd RNN-main

# Install core dependencies
pip install torch torchvision numpy pandas matplotlib scipy scikit-learn
```

### For Video Classification Projects

```bash
# Install additional dependencies
cd 09-VideoClassificationCRNN
pip install -r requirements.txt
```

---

## 📦 Source Modules

### Linear Regression Module

**File:** `src/linear_regression.py`

A production-ready PyTorch regression model with state-of-the-art optimization techniques.

#### Features

| Feature | Description |
|---------|-------------|
| **Architecture** | 5-layer deep neural network with BatchNorm and Dropout |
| **Activation** | LeakyReLU (prevents dying neurons) |
| **Regularization** | Dropout (0.15), Weight Decay (1e-4) |
| **Initialization** | Xavier uniform for optimal convergence |
| **Normalization** | Automatic feature and target scaling |
| **Scheduler** | CosineAnnealingWarmRestarts |
 | Configurable patience (default: 30) |
| **Device Support** | Automatic GPU/CPU detection |

#### Model Architecture

```
Input Layer (n_features)
    ↓
Linear(n_features → 512) → BatchNorm → LeakyReLU → Dropout(0.15)
    ↓
Linear(512 → 512) → BatchNorm → LeakyReLU → Dropout(0.15)
    ↓
Linear(512 → 256) → BatchNorm → LeakyReLU → Dropout(0.075)
    ↓
Linear(256 → 128) → BatchNorm → LeakyReLU
    ↓
Linear(128 → 1)
    ↓
Output (prediction)
```

#### Evaluation Metrics

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
age Error)
- **Accuracy** (with configurable tolerance)

#### Classes & Functions

| Class/Function | Purpose |
|----------------|---------|
| `LinearRegressionModel` | Main neural network model |
| `LinearRegressionDataModule` | Data handling with normalization |
| `linear_regression_dataset` | PyTorch Dataset wrapper |
| `train_model()` | Training loop with early stopping |
| `evaluate_model()` | Comprehensive evaluation |
| `load_model()` | Load saved checkpoints |
| `predict()` | Make predictions data |
| `plot_training_history()` | Visualize training metrics |
| `plot_predictions()` | Visualize predictions vs actual |

#### Quick Start

```python
from src.linear_regression import (
    LinearRegressionModel,
    LinearRegressionDataModule,
    train_model,
    evaluate_model
)
import torch
import torch.nn as nn
from torch.optim import Adam

# 1. Prepare data
data_module = LinearRegressionDataModule(X, Y, normalize=True)
train_loader, val_loader,28)

# 2. Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LinearRegressionModel(input_size=10, hidden_neurons=512, dropout_rate=0.15)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

# 3. Setup scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# 4. Train
history = train_model(
    model, train_loader, val_loader, criterion, optimizer,
ochs=300, device=device, early_stopping_patience=30
)

# 5. Evaluate
test_results = evaluate_model(model, test_loader, criterion, device=device)

# 6. Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'test_results': test_results
}, 'model.pth')
```

#### Loading & Inference

```python
from src.linear_regression import load_model, predict

# Load model
model, history, test_results = load_model(
    'model.pth',
    input_size=10,
    hidden_neurons=512,
ce='cpu'
)

# Make predictions
predictions = predict(model, new_data, device='cpu')
```

---

### Signal Generation Module

**File:** `src/multiple_signal.py`

Comprehensive signal generation toolkit for creating synthetic time series datasets.

#### Supported Waveforms

| Waveform | Method | Parameters |
|----------|--------|------------|
| Sine | `sin_wave()` | freq, length, amp, phase |
| Cosine | `cos_wave()` | freq, length, amp, phase |
| Triangle | `triangle_wave()` | freq, length, amp, phase |
| Sre | `square_wave()` | freq, length, amp, phase |
| Sawtooth | `sawtooth_wave()` | freq, length, amp, phase |
| Pulse Rectangle | `pulse_rectangle_wave()` | freq, length, amp, phase |
| Pulse Triangle | `pulse_triangle_wave()` | freq, length, amp, phase |
| Gaussian | `gaussian_wave()` | freq, length, amp, phase |
| White Noise | `white_noise()` | length, amp |

#### Classes

| Class | Purpose |
|-------|---------|
| `MultipleSignal` | Base class for signal generation |
| `CreateDataSet_1D_Array` | Dataset creation for model training |

#### Quick Start

```python
from src.multiple_signal import MultipleSignal, CreateDataSet_1D_Array

# Generate single signal
signal_gen = MultipleSignal()
sine = signal_gen.sin_wave(freq=5, length=256, amp=1, phase=0)
signal_gen.plot_signal(sine, title='Sine Wave')

# Create dataset
dataset = CreateDataSet_1D_Array(length=256)

# Generate specific waveform
signal = dataset.generate_wave(label='sine', length=256)

# Plot multiple samples
dataset.plot_samples(num_samples=5)
```

#### Available Labels for Dataset Generation

- `'sine'`
- `'cosine'`
- `'triangle'`
- `'square'`
- `'pulse_rectangle'`
- `'pulse_triangle'`
- `'gaussian'`
- `'sawtooth'`
- `'white_noise'`

---

## 📚 Notebooks & Experiments

### 01 - RNN Classification

**Objective:** Simple RNN training for classification of 3 signals (Sine, Square, Triangle)

**Key Concepts:**
- Basic RNN architecture
- Multi-class classification
- Signal preprocessing

---

### 02 - RNN Regression

**Objective:** Simple RNN trstimation

**Key Concepts:**
- Sequence-to-sequence prediction
- Regression with RNN
- Time series forecasting

![Sine Wave Estimation](https://user-images.githubusercontent.com/82975802/161122428-711b0824-f819-40c4-92ab-ffa7384ce342.png)

---

### 03 - RNN vs GRU Classification

**Objective:** Compare RNN and GRU models for signal classification

**Results (100 epochs):**

| Model | Accuracy |
|-------|----------|
| RNN   | 93.15%   |
| GRU   | 93.83%   |

**Key Insights:**
er vanilla RNN
- Better gradient flow in GRU architecture
- Faster convergence with GRU

---

### 04 - RNN vs GRU Regression

**Objective:** Compare RNN and GRU models for sine wave estimation

**Results (100 epochs):**

|## 🔄 Version History

- **v2.0** - Added production-ready linear regression module with advanced features
- **v1.5** - Added signal generation utilities
- **v1.0** - Initial release with RNN/GRU/LSTM implementations

---

**⭐ If you find this project useful, please consider giving it a star!**

**RNN Project Contributors**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **PyTorch Team** - Excellent deep learning framework
- **SciPy Community** - Signal processing utilities
- **scikit-learn** - Machine learning tools
- **Open Source Community** - Continuous feedback and improvements

---

## 📞 Support

For questions, issues, or suggestions:

- Open an issue on GitHub
- Contact via email
- Check existing documentation

---

 Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author
ent memory management

✅ **Advanced Training**
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Data normalization

✅ **Comprehensive Evaluation**
- Multiple metrics (MSE, MAE, RMSE, R², MAPE)
- Visualization tools
- Model comparison utilities

✅ **Model Persistence**
- Save/load functionality
- Checkpoint management
- History tracking

✅ **Modular Design**
- Easy integration
- Reusable components
- Clear API

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.ependencies

```
tensorflow>=2.8.0  (for some notebooks)
opencv-python>=4.5.0  (for video processing)
jupyter>=1.0.0  (for notebooks)
```

### Installation Command

```bash
pip install torch torchvision numpy pandas matplotlib scipy scikit-learn jupyter
```

---

## 🎯 Key Features

✅ **Production-Ready Code**
- Optimized for performance and scalability
- Comprehensive error handling
- Type hints and documentation

✅ **GPU Acceleration**
- Automatic CUDA detection
- Mixed precision training support
- Effici   hidden_neurons=config['hidden_neurons'],
    dropout_rate=config['dropout_rate']
)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=config['learning_rate'])

# Train
history = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    epochs=config['epochs'], device='cuda'
)
```

---

## 📋 Requirements

### Core Dependencies

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### Optional DegressionDataModule,
    train_model
)
import torch.nn as nn
from torch.optim import Adam

# Custom configuration
config = {
    'input_size': 20,
    'hidden_neurons': 256,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100
}

# Setup
data_module = LinearRegressionDataModule(X, Y, normalize=True)
train_loader, val_loader, test_loader = data_module.get_dataloaders(
    batch_size=config['batch_size']
)

model = LinearRegressionModel(
    input_size=config['input_size'],
 n = []

labels = ['sine', 'square', 'triangle']
for label_idx, label in enumerate(labels):
    for _ in range(1000):
        signal = dataset.generate_wave(label=label, length=256)
        X_train.append(signal)
        y_train.append(label_idx)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Dataset shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
```

### Example 3: Custom Model Training

```python
from src.linear_regression import (
    LinearRegressionModel,
    LinearRRU Accuracy | Winner |
|---------|--------------|--------------|--------|
| UCF101 Top-5 | 87% | 94% | GRU |

---

## 💡 Usage Examples

### Example 1: Train Linear Regression Model

```python
from src.linear_regression import main

# Run complete training pipeline
main()
```

### Example 2: Generate Signal Dataset

```python
from src.multiple_signal import CreateDataSet_1D_Array
import numpy as np

# Create dataset generator
dataset = CreateDataSet_1D_Array(length=256)

# Generate training data
X_train = []
y_trai|
| Features | 10 |
| Training Time (GPU) | ~3 minutes |
| R² Score | >0.9938 |
| Test Accuracy (tol=0.3) | 90-95% |
| MAE | <0.25 |
| RMSE | <0.32 |

### Signal Classification

| Task | RNN Accuracy | GRU Accuracy | Winner |
|------|--------------|--------------|--------|
| 3-Class Signal | 93.15% | 93.83% | GRU |

### Signal Regression

| Task | RNN Loss | GRU Loss | Winner |
|------|----------|----------|--------|
| Sine Wave | 0.0027 | 0.0026 | GRU |

### Video Classification

| Dataset | RNN Accuracy | G:**
- Backbone: Custom VGG-based model
- Temporal: RNN / GRU

**Results:**

| Model | Validation Accuracy |
|-------|---------------------|
| RNN   | 87%                 |
| GRU   | 94%                 |

**Key Insights:**
- GRU significantly outperforms RNN (7% improvement)
- Temporal modeling crucial for video understanding
- Feature extraction quality impacts final performance

---

## 📊 Performance Benchmarks

### Linear Regression Model

| Metric | Value |
|--------|-------|
| Dataset Size | 5,000 samples ```

**Backbone Options:**
- ResNet50V2 (pretrained)
- Custom VGG-based model

**RNN Modules:**
- RNN
- GRU (best performance)
- LSTM

**Dataset:**
- 2 classes
- Limited data (proof of concept)

**Note:** *Due to insufficient data, training results are limited. However, this architecture is production-ready for larger video classification tasks.*

---

### 10 - Video Classification CRNN (UCF101)

**Location:** `10-Video_Classification_CRNN/`

**Dataset:** UCF101 Top-5
- 573 videos
- 5 action classes

**Architecture
- Vanilla RNN
- GRU
- LSTM

**Pipeline:**
- Data generation
- Preprocessing
- Model training
- Evaluation

---

### 09 - Video Classification CRNN

**Location:** `09-VideoClassificationCRNN/`

**Components:**
- ✅ Training notebook
- ✅ Inference script (`inference.py`)
- ✅ Model definitions (`models.py`)
- ✅ Video loader (`load_video.py`)
- ✅ Requirements file

**Architecture:**

```
Video Frames
    ↓
Feature Extraction (ResNet50V2 / Custom VGG)
    ↓
Temporal Modeling (RNN / GRU / LSTM)
    ↓
Classification
 mechanism
- Reset gate mechanism
- Hidden state computation
- Forward pass implementation
- Inference pipeline

---

### 07 - LSTM Implementation from Scratch

**Objective:** Build LSTM from first principles

**Topics Covered:**
- Forget gate mechanism
- Input gate mechanism
- Output gate mechanism
- Cell state management
- Forward pass implementation
- Inference pipeline

---

### 08 - Ball Move Direction Classification

**Objective:** Classify ball movement direction using RNN variants

**Models Tested:**m/82975802/161124491-0d1061c0-c7e9-428a-98a7-77f93c762a71.png)

---

### 05 - Ball Move Data Generation

**Objective:** Generate synthetic data for ball movement direction prediction

**Features:**
- Trajectory simulation
- Direction labeling
- Data augmentation

![Ball Movement](https://user-images.githubusercontent.com/82975802/163728473-e6681737-8077-464e-a9c3-17a9a3e40115.png)

---

### 06 - GRU Implementation from Scratch

**Objective:** Build GRU from first principles

**Topics Covered:**
- Update gate Model | Loss (MSE) |
|-------|------------|
| RNN   | 0.0027     |
| GRU   | 0.0026     |

**Visualizations:**

![RNN Regression](https://user-images.githubusercontent.com/82975802/161124533-97516304-d0e1-4b09-9889-48d259a5274a.png)

![GRU Regression](https://user-images.githubusercontent.co
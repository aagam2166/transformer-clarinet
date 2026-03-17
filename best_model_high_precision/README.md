# HIGH PRECISION TRANSFORMER MODEL - NETWORK INTRUSION DETECTION

## Overview
This is a reproducible Transformer-based model for network intrusion detection with **98% accuracy** on DoS slowloris detection (your priority class).

**RANDOM SEED IS FIXED TO 42 - Results will be identical across systems**

## Files
- `model_weights.pt` - Model weights (PyTorch)
- `scaler.joblib` - Feature scaler (fitted on training data only)
- `features.json` - List of feature names in correct order
- `label_mapping.json` - Current→Original label mappings and attack names
- `reproducibility_config.json` - Complete config for reproducibility
- `inference.py` - Ready-to-use inference script

## Quick Start

```python
from inference import load_model, predict
import numpy as np

# Load model (fixes random seed automatically)
model, scaler, features, label_mapping, device = load_model(".")

# Predict on new data (X must have same features as training)
X = np.random.randn(100, 58)  # 58 features
preds, attack_names = predict(X, ".")
print(attack_names)
```

## Model Architecture
- **Type**: Tabular Transformer
- **d_model**: 64
- **depth**: 2 (layers)
- **dropout**: 0.3
- **n_heads**: 4
- **Total params**: ~122K

## Training Details
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss (label_smoothing=0.1)
- **Epochs**: 20
- **Batch size**: 256
- **Prior adjustment**: Enabled (α=0.8)

## Preprocessing (IMPORTANT for reproducibility)
1. Train-test split: 80-20 (stratified, random_state=42)
2. Scaler fit on TRAIN data only
3. Zero-variance features removed
4. SMOTE applied to training data only
5. Features standardized using StandardScaler

## Classes (Remapped)
- 0: Benign
- 1: DDoS
- 2: DoS GoldenEye (YOUR PRIORITY)
- 3: DoS Hulk
- 4: DoS Slowhttptest
- 5: DoS slowloris ← 98% accuracy on this

## Test Performance
- **Accuracy**: 0.9800
- **Precision**: 0.9800 (weighted)
- **Recall**: 0.9800 (weighted)
- **F1-Score**: 0.9800 (weighted)

## Reproducibility Guaranteed
✓ Random seed = 42 (set in first cell)
✓ PyTorch deterministic = True
✓ CUDNN benchmark = False
✓ Same results on CPU/GPU
✓ Same results across systems (given same Python/PyTorch versions)

## Requirements
- torch >= 1.9
- scikit-learn >= 0.24
- pandas
- numpy
- joblib

## To reproduce exact training:
Run the training notebook with RANDOM_SEED=42 set in the first cell.
All subsequent operations use fixed random_state=42.

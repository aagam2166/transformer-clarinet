# Transformer-Based Intrusion Detection Model

A pre-trained Transformer model for network intrusion detection. Use the saved model for inference on new data.

⚠️ **IMPORTANT:** Install dependencies first! See [Setup & Installation](#-setup--installation) below.

---

## 📁 What's Included

```
best_model_high_precision/
  ├── model_weights.pt          (Pre-trained model)
  ├── scaler.joblib             (Feature scaler)
  ├── label_mapping.json        (Attack class names)
  ├── features.json             (Feature names)
  ├── reproducibility_config.json (Model config)
  └── inference.py              (Reference code)

test_model_reproduction.ipynb   (How to use the model)
```

---

## 🚀 Quick Start

### For Inference Only (No Dataset Needed)
If you just want to use the pre-trained model:
```python
# See Step 1-3 below - works without df_cleaned.csv
```

### For Full Testing (Requires Dataset)
If you want to test the model on test data:
1. Download `df_cleaned.csv` from: https://drive.google.com/file/d/1oHnlFd44RHHOPud9DoWAEtxi7cMrZMNc/view?usp=sharing
2. Place it in the project root folder
3. Run the notebook

### Step 1: Load the Model
```python
import torch
import joblib
import json

model_dir = 'best_model_high_precision'
device = 'cpu'  # or 'cuda' for GPU

# Load model weights
model.load_state_dict(torch.load(f'{model_dir}/model_weights.pt', map_location=device))

# Load preprocessor
scaler = joblib.load(f'{model_dir}/scaler.joblib')

# Load class names
with open(f'{model_dir}/label_mapping.json') as f:
    labels = json.load(f)
```

### Step 2: Prepare Data
```python
# Scale your input features
X_scaled = scaler.transform(X_raw)

# Convert to tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
```

### Step 3: Run Inference
```python
model.eval()
with torch.no_grad():
    logits = model(X_tensor)
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

# Get attack names
attack_names = labels['attack_names']
predicted_labels = [attack_names[str(int(p))] for p in predictions]
```

---

## 📋 Setup & Installation

### Prerequisites
- **Python:** 3.8 or higher
- **pip:** Package manager (comes with Python)

### Step 1: Install Dependencies
Open your terminal/command prompt and run:

```bash
pip install numpy pandas scikit-learn torch joblib
```

**Or use the requirements file:**
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import torch; import joblib; import pandas; print('✓ All dependencies installed')"
```

### Step 3: Run the Model
Open `test_model_reproduction.ipynb` in Jupyter/VS Code and run all cells

---

## 📊 Model Info

- **Type:** Tabular Transformer
- **Parameters:** 122K
- **Classes:** 6 attack types (Benign, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris)
- **Accuracy:** 94.52% on test set
- **Input Features:** 58 network features

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch` |
| `ModuleNotFoundError: No module named 'joblib'` | Run `pip install joblib` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| CUDA/GPU issues (optional) | Use CPU: Change `device='cuda'` to `device='cpu'` |

---

## 🔍 See Also

- `test_model_reproduction.ipynb` - Step-by-step inference example
- `best_model_high_precision/inference.py` - Reference implementation

---





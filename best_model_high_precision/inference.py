
"""
Inference script for High Precision Transformer Model
Ensures reproducible results across systems
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

# ============ REPRODUCIBILITY ============
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ MODEL DEFINITION ============
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(num_features, d_model))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.value_projection(x)
        x = x + self.feature_embedding
        return x

class TabularTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model, n_heads, depth, dropout):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits

# ============ LOAD ARTIFACTS ============
def load_model(model_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(Path(model_dir) / "reproducibility_config.json") as f:
        config = json.load(f)

    # Load features
    with open(Path(model_dir) / "features.json") as f:
        features = json.load(f)

    # Load label mapping
    with open(Path(model_dir) / "label_mapping.json") as f:
        label_mapping = json.load(f)

    # Load scaler
    scaler = joblib.load(Path(model_dir) / "scaler.joblib")

    # Initialize model
    arch = config["MODEL_ARCHITECTURE"]
    model = TabularTransformer(
        num_features=arch["num_features"],
        num_classes=arch["num_classes"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        depth=arch["depth"],
        dropout=arch["dropout"]
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(Path(model_dir) / "model_weights.pt", map_location=device))
    model.eval()

    return model, scaler, features, label_mapping, config, device

# ============ PRIOR ADJUSTMENT ============
def get_log_prior_adjustment(config, device, model_dir="."):
    """Get log-prior adjustment tensor for predictions."""
    if not config.get("PRIOR_ADJUSTMENT", {}).get("enabled", False):
        return None

    # Try to load from parent directory first (for GitHub compatibility)
    label_count_paths = [
        Path(model_dir).parent / "label_count_tracker.csv",
        Path(model_dir) / "label_count_tracker.csv",
        Path("label_count_tracker.csv")
    ]

    counts_df = None
    for path in label_count_paths:
        if path.exists():
            counts_df = pd.read_csv(path).sort_values("current_label")
            break

    if counts_df is None:
        print("⚠ WARNING: label_count_tracker.csv not found. Skipping prior adjustment.")
        return None

    source_col = config["PRIOR_ADJUSTMENT"].get("source_column", "filtered_total")
    if source_col not in counts_df.columns:
        print(f"⚠ WARNING: '{source_col}' not found. Skipping prior adjustment.")
        return None

    counts = counts_df[source_col].astype(float).to_numpy()
    counts = counts + 1.0  # Laplace smoothing
    priors = counts / counts.sum()

    alpha = float(config["PRIOR_ADJUSTMENT"].get("alpha", 1.0))
    adjustment = alpha * np.log(priors)

    return torch.tensor(adjustment, dtype=torch.float32, device=device)

# ============ INFERENCE ============
def predict(X, model_dir="."):
    """
    X: pandas DataFrame or numpy array with feature columns
    Returns: class predictions and attack names
    """
    model, scaler, features, label_mapping, config, device = load_model(model_dir)

    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)

    # Ensure all features present
    assert all(f in X.columns for f in features), f"Missing features: {set(features) - set(X.columns)}"

    # Scale features
    X_scaled = scaler.transform(X[features])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        
        # Apply prior adjustment if enabled
        prior_adjustment = get_log_prior_adjustment(config, device, model_dir)
        if prior_adjustment is not None:
            outputs = outputs + prior_adjustment
        
        preds = outputs.argmax(1).cpu().numpy()

    # Map to attack names
    attack_names = [label_mapping["attack_names"][str(p)] for p in preds]

    return preds, attack_names

if __name__ == "__main__":
    print("Model loaded successfully!")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Seed: {RANDOM_SEED} (for reproducibility)")
    print("\n✓ Model ready for inference with prior adjustment enabled!")
    print("  - Uses log-prior adjustment for class imbalance")
    print("  - Alpha: 0.8, Source: filtered_total")

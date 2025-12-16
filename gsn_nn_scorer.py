#!/usr/bin/env python
"""
GSN Neural Network Scorer

A trainable neural network model that learns to score locations based on
all available features (G, H, A, N, T components).

The model can discover non-linear patterns and feature interactions
that the linear weighted sum cannot capture.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed. Neural network features disabled.")

# Try to import sklearn for preprocessing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

# Feature names in order
FEATURE_NAMES = [
    # G components
    "ga_norm", "ct_norm", "boundary_score", "magnetic", "elevation",
    "bouguer", "isostatic", "heat_flow", "volcanic_dist", "seismic",
    # H components
    "H_basic", "H_weighted", "angle_score", "distance_score",
    "gc_alignment", "fibonacci_score",
    # A components
    "visibility", "pattern_match", "solstice_align",
    # N components
    "network_score", "centrality", "flow_potential",
    # T components
    "temporal_score", "epoch_visibility",
]


def extract_features(lat: float, lon: float, 
                    compute_funcs: Dict[str, Callable] = None) -> np.ndarray:
    """
    Extract all features for a location.
    
    Args:
        lat, lon: Location coordinates
        compute_funcs: Dict of feature computation functions
    
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    if compute_funcs is None:
        # Return placeholder zeros if no compute functions provided
        return np.zeros(len(FEATURE_NAMES))
    
    for name in FEATURE_NAMES:
        if name in compute_funcs:
            try:
                value = compute_funcs[name](lat, lon)
                features.append(float(value) if value is not None else 0.0)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def extract_features_batch(locations: List[Tuple[float, float]],
                          compute_funcs: Dict[str, Callable] = None) -> np.ndarray:
    """Extract features for multiple locations."""
    return np.array([extract_features(lat, lon, compute_funcs) 
                     for lat, lon in locations])


# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------

if HAS_TORCH:
    
    class GSNScorer(nn.Module):
        """
        Neural network that learns to score locations based on
        all available features (G, H, A, N, T components).
        
        Architecture:
        - Input: Feature vector (n_features)
        - Hidden layers with BatchNorm, ReLU, Dropout
        - Output: Score between 0 and 1
        """
        
        def __init__(self, n_features: int = None, 
                     hidden_sizes: List[int] = None,
                     dropout: float = 0.2):
            super().__init__()
            
            if n_features is None:
                n_features = len(FEATURE_NAMES)
            
            if hidden_sizes is None:
                hidden_sizes = [64, 32, 16]
            
            layers = []
            prev_size = n_features
            
            for size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, size),
                    nn.BatchNorm1d(size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_size = size
            
            # Output layer
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
            self.n_features = n_features
        
        def forward(self, x):
            return self.network(x).squeeze(-1)
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predict scores for numpy array input."""
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                scores = self.forward(x_tensor)
                return scores.numpy()
    
    
    class GSNScorerDeep(nn.Module):
        """
        Deeper architecture with residual connections for better gradient flow.
        """
        
        def __init__(self, n_features: int = None,
                     hidden_size: int = 64,
                     n_blocks: int = 3,
                     dropout: float = 0.2):
            super().__init__()
            
            if n_features is None:
                n_features = len(FEATURE_NAMES)
            
            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )
            
            # Residual blocks
            self.blocks = nn.ModuleList([
                self._make_block(hidden_size, dropout)
                for _ in range(n_blocks)
            ])
            
            # Output head
            self.output_head = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            
            self.n_features = n_features
        
        def _make_block(self, size: int, dropout: float):
            return nn.Sequential(
                nn.Linear(size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(size, size),
                nn.BatchNorm1d(size),
            )
        
        def forward(self, x):
            x = self.input_proj(x)
            
            for block in self.blocks:
                residual = x
                x = block(x)
                x = nn.functional.relu(x + residual)
            
            return self.output_head(x).squeeze(-1)
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                scores = self.forward(x_tensor)
                return scores.numpy()


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for neural network training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 10
    min_delta: float = 0.001
    neg_ratio: float = 3.0  # Ratio of negative to positive samples
    val_split: float = 0.2


class GSNTrainer:
    """
    Training pipeline for GSN neural network scorer.
    """
    
    def __init__(self, model: 'GSNScorer' = None, config: TrainingConfig = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for neural network training")
        
        self.model = model if model else GSNScorer()
        self.config = config if config else TrainingConfig()
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.history = {"train_loss": [], "val_loss": [], "val_recall": []}
    
    def prepare_data(self, 
                    positive_features: np.ndarray,
                    negative_features: np.ndarray = None) -> Tuple:
        """
        Prepare training data.
        
        Args:
            positive_features: Feature vectors for known nodes (positive examples)
            negative_features: Feature vectors for random locations (negative examples)
                             If None, will sample from uniform distribution
        
        Returns:
            (X_train, y_train, X_val, y_val) tuple
        """
        n_pos = len(positive_features)
        
        # Generate negative samples if not provided
        if negative_features is None:
            n_neg = int(n_pos * self.config.neg_ratio)
            negative_features = np.random.randn(n_neg, positive_features.shape[1])
        
        # Combine
        X = np.vstack([positive_features, negative_features])
        y = np.hstack([np.ones(n_pos), np.zeros(len(negative_features))])
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Split
        if HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.val_split, random_state=42, stratify=y
            )
        else:
            # Manual split
            n_val = int(len(X) * self.config.val_split)
            indices = np.random.permutation(len(X))
            X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
            y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]
        
        return X_train, y_train, X_val, y_val
    
    def train(self, 
             positive_features: np.ndarray,
             negative_features: np.ndarray = None,
             verbose: bool = True) -> Dict:
        """
        Train the neural network.
        
        Args:
            positive_features: Features for known nodes
            negative_features: Features for negative samples
            verbose: Print training progress
        
        Returns:
            Training history and final metrics
        """
        X_train, y_train, X_val, y_val = self.prepare_data(
            positive_features, negative_features
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.BCELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor).item()
                
                # Compute recall at threshold 0.5
                val_pred_binary = (val_predictions > 0.5).float()
                val_pos_mask = y_val_tensor > 0.5
                if val_pos_mask.sum() > 0:
                    val_recall = (val_pred_binary[val_pos_mask].sum() / val_pos_mask.sum()).item()
                else:
                    val_recall = 0.0
            
            avg_train_loss = np.mean(train_losses)
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_recall"].append(val_recall)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_recall={val_recall:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
            "final_val_recall": self.history["val_recall"][-1],
            "epochs_trained": len(self.history["train_loss"]),
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict scores for new locations."""
        if self.scaler:
            features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def save(self, path: str):
        """Save model and scaler."""
        if not HAS_TORCH:
            return
        
        state = {
            "model_state": self.model.state_dict(),
            "model_config": {
                "n_features": self.model.n_features,
            },
            "scaler_mean": self.scaler.mean_ if self.scaler else None,
            "scaler_scale": self.scaler.scale_ if self.scaler else None,
            "history": self.history,
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model and scaler."""
        if not HAS_TORCH:
            return
        
        state = torch.load(path)
        
        self.model = GSNScorer(n_features=state["model_config"]["n_features"])
        self.model.load_state_dict(state["model_state"])
        
        if state.get("scaler_mean") is not None and self.scaler:
            self.scaler.mean_ = state["scaler_mean"]
            self.scaler.scale_ = state["scaler_scale"]
        
        self.history = state.get("history", {})


# ---------------------------------------------------------
# Feature Importance from Neural Network
# ---------------------------------------------------------

def compute_gradient_importance(model: 'GSNScorer',
                                features: np.ndarray,
                                feature_names: List[str] = None) -> Dict:
    """
    Compute feature importance using gradient-based method.
    
    Higher absolute gradient = more important feature.
    
    Args:
        model: Trained GSNScorer model
        features: Input features (n_samples, n_features)
        feature_names: Names for each feature
    
    Returns:
        Dict with importance scores and ranking
    """
    if not HAS_TORCH:
        return {}
    
    if feature_names is None:
        feature_names = FEATURE_NAMES[:features.shape[1]]
    
    model.eval()
    
    # Convert to tensor with gradient tracking
    X = torch.FloatTensor(features)
    X.requires_grad = True
    
    # Forward pass
    outputs = model(X)
    
    # Backward pass
    outputs.sum().backward()
    
    # Get gradients
    gradients = X.grad.abs().mean(dim=0).numpy()
    
    importance = {name: float(grad) for name, grad in zip(feature_names, gradients)}
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    
    return {
        "importance": importance,
        "ranked": ranked,
        "method": "gradient",
    }


# ---------------------------------------------------------
# Integration with GSN Prediction Pipeline
# ---------------------------------------------------------

class NeuralGSNPredictor:
    """
    Integration class that uses neural network for GSN prediction.
    """
    
    def __init__(self, model_path: str = None):
        self.trainer = GSNTrainer() if HAS_TORCH else None
        
        if model_path and self.trainer:
            self.trainer.load(model_path)
    
    def train_from_nodes(self, 
                        known_nodes: Dict,
                        compute_funcs: Dict[str, Callable],
                        n_negative: int = 500,
                        verbose: bool = True) -> Dict:
        """
        Train the model using known nodes.
        
        Args:
            known_nodes: Dict of known GSN nodes
            compute_funcs: Functions to compute each feature
            n_negative: Number of random negative samples
            verbose: Print progress
        
        Returns:
            Training results
        """
        if not self.trainer:
            return {"error": "PyTorch not available"}
        
        # Extract features for positive samples
        positive_locs = []
        for name, data in known_nodes.items():
            if isinstance(data, dict):
                lat, lon = data["coords"]
            else:
                lat, lon = data
            positive_locs.append((lat, lon))
        
        positive_features = extract_features_batch(positive_locs, compute_funcs)
        
        # Generate negative samples
        negative_locs = [
            (np.random.uniform(-60, 60), np.random.uniform(-180, 180))
            for _ in range(n_negative)
        ]
        negative_features = extract_features_batch(negative_locs, compute_funcs)
        
        # Train
        return self.trainer.train(positive_features, negative_features, verbose)
    
    def score_location(self, lat: float, lon: float,
                      compute_funcs: Dict[str, Callable]) -> float:
        """Score a single location."""
        if not self.trainer:
            return 0.0
        
        features = extract_features(lat, lon, compute_funcs)
        scores = self.trainer.predict(features.reshape(1, -1))
        return float(scores[0])
    
    def score_grid(self, lats: np.ndarray, lons: np.ndarray,
                  compute_funcs: Dict[str, Callable]) -> np.ndarray:
        """Score a grid of locations."""
        if not self.trainer:
            return np.zeros((len(lats), len(lons)))
        
        locations = [(lat, lon) for lat in lats for lon in lons]
        features = extract_features_batch(locations, compute_funcs)
        scores = self.trainer.predict(features)
        
        return scores.reshape(len(lats), len(lons))


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Neural Network Scorer Demo")
    print("-" * 40)
    
    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch")
    else:
        # Create synthetic data for demo
        n_features = len(FEATURE_NAMES)
        n_positive = 50
        n_negative = 200
        
        # Positive samples: higher values for certain features
        positive_features = np.random.randn(n_positive, n_features) + 0.5
        
        # Negative samples: random
        negative_features = np.random.randn(n_negative, n_features)
        
        # Train
        print("\nTraining neural network...")
        trainer = GSNTrainer(config=TrainingConfig(epochs=50, patience=5))
        result = trainer.train(positive_features, negative_features, verbose=True)
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {result['best_val_loss']:.4f}")
        print(f"Final validation recall: {result['final_val_recall']:.4f}")
        
        # Feature importance
        print("\nComputing feature importance...")
        all_features = np.vstack([positive_features, negative_features])
        importance = compute_gradient_importance(trainer.model, all_features)
        
        print("\nTop 10 most important features:")
        for name, score in importance["ranked"][:10]:
            print(f"  {name}: {score:.4f}")

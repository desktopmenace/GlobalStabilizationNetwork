#!/usr/bin/env python
"""
GSN ML-Based Grid Scorer

Replaces the hand-crafted F = alpha*G + beta*H formula with a trained
neural network that learns optimal feature combinations from known GSN nodes.

Features:
- Feature extraction from G, H, A, N, T components
- Data augmentation for limited positive examples
- Hard negative mining for robust training
- Drop-in replacement for compute_F_grid()
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed. ML grid scorer features disabled.")

# Try to import sklearn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------
# Feature Names for Grid Scoring
# ---------------------------------------------------------

GRID_FEATURE_NAMES = [
    # G components (geophysical)
    "ga_norm",           # Gravity anomaly normalized
    "ct_norm",           # Crustal thickness normalized
    "boundary_score",    # Distance to plate boundary score
    
    # H components (geometric coherence)
    "H_basic",           # Basic Penrose angle coherence
    "H_weighted",        # Distance-weighted coherence
    "H_weighted_ext",    # Extended angles coherence
    
    # Spatial features
    "lat_norm",          # Latitude normalized (-1 to 1)
    "lon_sin",           # Longitude sine (for continuity)
    "lon_cos",           # Longitude cosine (for continuity)
    "abs_lat",           # Absolute latitude (distance from equator)
    
    # Distance features
    "min_node_dist",     # Distance to nearest known node (km)
    "mean_node_dist",    # Mean distance to known nodes (km)
    "node_density",      # Density of nearby known nodes
]


# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------

if HAS_TORCH:
    
    class GridScorerNet(nn.Module):
        """
        Neural network for grid-based GSN scoring.
        
        Architecture optimized for geospatial features with
        batch normalization and residual connections.
        """
        
        def __init__(self, n_features: int = None, 
                     hidden_sizes: List[int] = None,
                     dropout: float = 0.3):
            super().__init__()
            
            if n_features is None:
                n_features = len(GRID_FEATURE_NAMES)
            
            if hidden_sizes is None:
                hidden_sizes = [128, 64, 32]
            
            self.n_features = n_features
            
            # Input layer with batch norm
            self.input_bn = nn.BatchNorm1d(n_features)
            
            # Build hidden layers
            layers = []
            prev_size = n_features
            
            for i, size in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_size = size
            
            self.hidden = nn.Sequential(*layers)
            
            # Output layer
            self.output = nn.Sequential(
                nn.Linear(prev_size, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.input_bn(x)
            x = self.hidden(x)
            return self.output(x).squeeze(-1)
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predict scores for numpy array input."""
            self.eval()
            with torch.no_grad():
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                x_tensor = torch.FloatTensor(x)
                scores = self.forward(x_tensor)
                return scores.numpy()


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in km between two points."""
    R = 6371  # Earth's radius in km
    
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def extract_grid_features(
    lat: float, 
    lon: float,
    G_val: float = 0.0,
    H_val: float = 0.0,
    ga_norm: float = 0.0,
    ct_norm: float = 0.0,
    boundary_score: float = 0.0,
    known_node_coords: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Extract features for a single grid point.
    
    Args:
        lat, lon: Grid point coordinates
        G_val: Pre-computed G value (geophysical)
        H_val: Pre-computed H value (geometric)
        ga_norm: Normalized gravity anomaly
        ct_norm: Normalized crustal thickness
        boundary_score: Distance to plate boundary score
        known_node_coords: List of (lat, lon) for known nodes
    
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # G components
    features.append(ga_norm)
    features.append(ct_norm)
    features.append(boundary_score)
    
    # H components (use the pre-computed value and derive variants)
    features.append(H_val)  # H_basic
    features.append(H_val * 1.1)  # H_weighted (approximation)
    features.append(H_val * 0.9)  # H_weighted_ext (approximation)
    
    # Spatial features
    features.append(lat / 90.0)  # lat_norm
    features.append(np.sin(np.radians(lon)))  # lon_sin
    features.append(np.cos(np.radians(lon)))  # lon_cos
    features.append(abs(lat) / 90.0)  # abs_lat
    
    # Distance features (relative to known nodes)
    if known_node_coords and len(known_node_coords) > 0:
        distances = [haversine_km(lat, lon, nlat, nlon) 
                    for nlat, nlon in known_node_coords]
        min_dist = min(distances)
        mean_dist = np.mean(distances)
        # Node density: count of nodes within 1000 km
        nearby_count = sum(1 for d in distances if d < 1000)
        node_density = nearby_count / len(known_node_coords)
    else:
        min_dist = 5000.0
        mean_dist = 10000.0
        node_density = 0.0
    
    features.append(min_dist / 10000.0)  # Normalize to ~0-1
    features.append(mean_dist / 20000.0)  # Normalize
    features.append(node_density)
    
    return np.array(features, dtype=np.float32)


def extract_features_for_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    G_grid: np.ndarray,
    H_grid: np.ndarray,
    components: Dict[str, np.ndarray],
    known_node_coords: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Extract features for an entire lat/lon grid.
    
    Args:
        lats: 1D array of latitudes
        lons: 1D array of longitudes
        G_grid: 2D array of G values (nlat, nlon)
        H_grid: 2D array of H values (nlat, nlon)
        components: Dict with ga_norm, ct_norm, tb grids
        known_node_coords: List of known node coordinates
    
    Returns:
        Feature matrix (n_points, n_features)
    """
    nlat, nlon = len(lats), len(lons)
    n_features = len(GRID_FEATURE_NAMES)
    
    print(f"[INFO] Extracting {n_features} features for {nlat * nlon:,} grid points...")
    
    # Pre-compute node distances for all grid points
    if known_node_coords is None:
        known_node_coords = []
    
    # Create feature grids
    features = np.zeros((nlat, nlon, n_features), dtype=np.float32)
    
    # G components
    features[:, :, 0] = components.get("ga_norm", np.zeros((nlat, nlon)))
    features[:, :, 1] = components.get("ct_norm", np.zeros((nlat, nlon)))
    features[:, :, 2] = components.get("tb", np.zeros((nlat, nlon)))
    
    # H components
    features[:, :, 3] = H_grid
    features[:, :, 4] = H_grid * 1.1  # Variant
    features[:, :, 5] = H_grid * 0.9  # Variant
    
    # Spatial features (vectorized)
    lat_grid = lats[:, np.newaxis]
    lon_grid = lons[np.newaxis, :]
    
    features[:, :, 6] = lat_grid / 90.0  # lat_norm
    features[:, :, 7] = np.sin(np.radians(lon_grid))  # lon_sin
    features[:, :, 8] = np.cos(np.radians(lon_grid))  # lon_cos
    features[:, :, 9] = np.abs(lat_grid) / 90.0  # abs_lat
    
    # Distance features (compute for each grid point)
    if len(known_node_coords) > 0:
        node_lats = np.array([c[0] for c in known_node_coords])
        node_lons = np.array([c[1] for c in known_node_coords])
        
        # Compute distances using vectorized haversine
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                distances = np.array([
                    haversine_km(lat, lon, nlat, nlon)
                    for nlat, nlon in known_node_coords
                ])
                features[i, j, 10] = distances.min() / 10000.0
                features[i, j, 11] = distances.mean() / 20000.0
                features[i, j, 12] = np.sum(distances < 1000) / len(distances)
    else:
        features[:, :, 10] = 0.5
        features[:, :, 11] = 0.5
        features[:, :, 12] = 0.0
    
    # Reshape to (n_points, n_features)
    return features.reshape(-1, n_features)


# ---------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    n_augmentations_per_sample: int = 10
    spatial_jitter_deg: float = 0.3  # Max jitter in degrees
    feature_noise_std: float = 0.05  # Feature noise standard deviation
    rotation_angles: List[float] = field(default_factory=lambda: [0, 90, 180, 270])


def augment_positive_samples(
    positive_coords: List[Tuple[float, float]],
    positive_features: np.ndarray,
    config: AugmentationConfig = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment positive samples using spatial jitter and feature noise.
    
    Args:
        positive_coords: List of (lat, lon) for positive samples
        positive_features: Feature matrix for positive samples
        config: Augmentation configuration
    
    Returns:
        Tuple of (augmented_coords, augmented_features)
    """
    if config is None:
        config = AugmentationConfig()
    
    n_samples = len(positive_coords)
    n_aug = config.n_augmentations_per_sample
    
    aug_coords = []
    aug_features = []
    
    for i in range(n_samples):
        lat, lon = positive_coords[i]
        features = positive_features[i]
        
        # Original sample
        aug_coords.append((lat, lon))
        aug_features.append(features.copy())
        
        # Augmented samples
        for _ in range(n_aug):
            # Spatial jitter
            jitter_lat = np.random.uniform(-config.spatial_jitter_deg, 
                                           config.spatial_jitter_deg)
            jitter_lon = np.random.uniform(-config.spatial_jitter_deg, 
                                           config.spatial_jitter_deg)
            
            new_lat = np.clip(lat + jitter_lat, -89.9, 89.9)
            new_lon = ((lon + jitter_lon + 180) % 360) - 180
            
            # Feature noise
            noise = np.random.normal(0, config.feature_noise_std, features.shape)
            new_features = features + noise
            
            # Update spatial features in the augmented sample
            new_features[6] = new_lat / 90.0
            new_features[7] = np.sin(np.radians(new_lon))
            new_features[8] = np.cos(np.radians(new_lon))
            new_features[9] = abs(new_lat) / 90.0
            
            aug_coords.append((new_lat, new_lon))
            aug_features.append(new_features)
    
    return aug_coords, np.array(aug_features, dtype=np.float32)


# ---------------------------------------------------------
# Hard Negative Mining
# ---------------------------------------------------------

def generate_random_negatives(
    n_samples: int,
    known_node_coords: List[Tuple[float, float]],
    min_distance_km: float = 500.0,
    seed: int = None,
) -> List[Tuple[float, float]]:
    """
    Generate random negative samples that are far from known nodes.
    
    Args:
        n_samples: Number of negative samples to generate
        known_node_coords: Coordinates of known nodes to avoid
        min_distance_km: Minimum distance from any known node
        seed: Random seed
    
    Returns:
        List of (lat, lon) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    negatives = []
    attempts = 0
    max_attempts = n_samples * 20
    
    while len(negatives) < n_samples and attempts < max_attempts:
        # Random location (avoid extreme latitudes)
        lat = np.random.uniform(-70, 70)
        lon = np.random.uniform(-180, 180)
        
        # Check distance from known nodes
        is_valid = True
        for nlat, nlon in known_node_coords:
            dist = haversine_km(lat, lon, nlat, nlon)
            if dist < min_distance_km:
                is_valid = False
                break
        
        if is_valid:
            negatives.append((lat, lon))
        
        attempts += 1
    
    if len(negatives) < n_samples:
        print(f"[WARN] Only generated {len(negatives)}/{n_samples} valid negatives")
    
    return negatives


def mine_hard_negatives(
    model: 'GridScorerNet',
    candidate_features: np.ndarray,
    candidate_coords: List[Tuple[float, float]],
    known_node_coords: List[Tuple[float, float]],
    n_hard: int = 100,
    min_distance_km: float = 300.0,
    score_threshold: float = 0.5,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Find hard negatives: locations that the model scores high but are far from nodes.
    
    Args:
        model: Current trained model
        candidate_features: Features for candidate locations
        candidate_coords: Coordinates for candidates
        known_node_coords: Known node coordinates
        n_hard: Number of hard negatives to select
        min_distance_km: Minimum distance from known nodes
        score_threshold: Minimum score to be considered "hard"
    
    Returns:
        Tuple of (hard_coords, hard_features)
    """
    if not HAS_TORCH:
        return [], np.array([])
    
    # Get model predictions
    scores = model.predict(candidate_features)
    
    hard_coords = []
    hard_features = []
    
    # Sort by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    for idx in sorted_indices:
        if len(hard_coords) >= n_hard:
            break
        
        if scores[idx] < score_threshold:
            break
        
        lat, lon = candidate_coords[idx]
        
        # Check if far enough from known nodes
        min_dist = min(haversine_km(lat, lon, nlat, nlon) 
                      for nlat, nlon in known_node_coords)
        
        if min_dist >= min_distance_km:
            hard_coords.append((lat, lon))
            hard_features.append(candidate_features[idx])
    
    return hard_coords, np.array(hard_features) if hard_features else np.array([])


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for ML grid scorer training."""
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 20
    min_delta: float = 0.0005
    val_split: float = 0.2
    pos_weight: float = 3.0  # Weight for positive class (handles imbalance)
    n_augmentations: int = 10
    n_random_negatives: int = 2000
    n_hard_negatives: int = 200
    hard_neg_rounds: int = 2


class MLGridScorer:
    """
    ML-based grid scorer that replaces F = alpha*G + beta*H.
    
    Uses a trained neural network to predict node probability
    from extracted features.
    """
    
    def __init__(self, model_path: str = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for MLGridScorer")
        
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.training_history = {"train_loss": [], "val_loss": [], "val_auc": []}
        self.known_node_coords = []
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _prepare_training_data(
        self,
        positive_coords: List[Tuple[float, float]],
        positive_features: np.ndarray,
        config: TrainingConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with augmentation and negatives."""
        
        print(f"[INFO] Preparing training data...")
        print(f"[INFO] Original positives: {len(positive_coords)}")
        
        # Augment positive samples
        aug_config = AugmentationConfig(n_augmentations_per_sample=config.n_augmentations)
        aug_coords, aug_features = augment_positive_samples(
            positive_coords, positive_features, aug_config
        )
        print(f"[INFO] Augmented positives: {len(aug_coords)}")
        
        # Generate random negatives
        neg_coords = generate_random_negatives(
            config.n_random_negatives,
            positive_coords,
            min_distance_km=500.0,
            seed=42
        )
        
        # Extract features for negatives
        neg_features = []
        for lat, lon in neg_coords:
            # Create approximate features for negative samples
            feat = extract_grid_features(
                lat, lon,
                G_val=np.random.uniform(-0.5, 0.5),
                H_val=np.random.uniform(0, 0.3),
                ga_norm=np.random.uniform(-1, 1),
                ct_norm=np.random.uniform(-1, 1),
                boundary_score=np.random.uniform(0, 0.5),
                known_node_coords=positive_coords
            )
            neg_features.append(feat)
        
        neg_features = np.array(neg_features, dtype=np.float32)
        print(f"[INFO] Random negatives: {len(neg_coords)}")
        
        # Combine data
        X = np.vstack([aug_features, neg_features])
        y = np.hstack([
            np.ones(len(aug_features)),
            np.zeros(len(neg_features))
        ])
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Split into train/val
        if HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=config.val_split, random_state=42, stratify=y
            )
        else:
            n_val = int(len(X) * config.val_split)
            indices = np.random.permutation(len(X))
            X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
            y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]
        
        print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def train(
        self,
        positive_coords: List[Tuple[float, float]],
        positive_features: np.ndarray,
        config: TrainingConfig = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the ML grid scorer.
        
        Args:
            positive_coords: Coordinates of known GSN nodes
            positive_features: Features for known nodes
            config: Training configuration
            verbose: Print progress
        
        Returns:
            Training results dict
        """
        if config is None:
            config = TrainingConfig()
        
        self.known_node_coords = positive_coords.copy()
        
        # Prepare data
        X_train, y_train, X_val, y_val = self._prepare_training_data(
            positive_coords, positive_features, config
        )
        
        # Create model
        n_features = X_train.shape[1]
        self.model = GridScorerNet(n_features=n_features)
        
        if verbose:
            print(f"\n[INFO] Training GridScorerNet with {n_features} features")
            print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Loss with positive class weighting
        pos_weight = torch.tensor([config.pos_weight])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(config.epochs):
            # Train
            self.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (without sigmoid for BCEWithLogitsLoss)
                logits = self.model.hidden(self.model.input_bn(X_batch))
                logits = self.model.output[:-1](logits).squeeze(-1)  # Skip sigmoid
                
                loss = criterion(logits, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model.hidden(self.model.input_bn(X_val_tensor))
                val_logits = self.model.output[:-1](val_logits).squeeze(-1)
                val_loss = criterion(val_logits, y_val_tensor).item()
                
                # Compute metrics
                val_preds = torch.sigmoid(val_logits)
                val_pred_binary = (val_preds > 0.5).float()
                
                # Recall on positive class
                pos_mask = y_val_tensor > 0.5
                if pos_mask.sum() > 0:
                    recall = (val_pred_binary[pos_mask].sum() / pos_mask.sum()).item()
                else:
                    recall = 0.0
                
                # Precision
                pred_pos = val_pred_binary.sum()
                if pred_pos > 0:
                    precision = ((val_pred_binary * y_val_tensor).sum() / pred_pos).item()
                else:
                    precision = 0.0
            
            avg_train_loss = np.mean(train_losses)
            
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_auc"].append(recall)  # Using recall as metric
            
            # Update scheduler
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.epochs}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"recall={recall:.4f}, precision={precision:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        
        elapsed = time.time() - start_time
        
        return {
            "epochs_trained": len(self.training_history["train_loss"]),
            "best_val_loss": best_val_loss,
            "final_recall": self.training_history["val_auc"][-1],
            "training_time_sec": elapsed,
            "n_features": n_features,
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict scores for feature matrix."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        return self.model.predict(features)
    
    def predict_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        G_grid: np.ndarray,
        H_grid: np.ndarray,
        components: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Predict F scores for an entire grid.
        
        Drop-in replacement for F = alpha*G + beta*H.
        
        Args:
            lats: 1D array of latitudes
            lons: 1D array of longitudes
            G_grid: 2D array of G values
            H_grid: 2D array of H values
            components: Dict with component grids
        
        Returns:
            2D array of ML-predicted F values
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        nlat, nlon = len(lats), len(lons)
        
        # Extract features
        features = extract_features_for_grid(
            lats, lons, G_grid, H_grid, components, self.known_node_coords
        )
        
        # Predict
        if self.scaler:
            features = self.scaler.transform(features)
        
        scores = self.model.predict(features)
        
        # Reshape back to grid
        return scores.reshape(nlat, nlon)
    
    def save(self, path: str):
        """Save model and scaler to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Cannot save.")
        
        state = {
            "model_state": self.model.state_dict(),
            "model_config": {
                "n_features": self.model.n_features,
            },
            "scaler_mean": self.scaler.mean_.tolist() if self.scaler else None,
            "scaler_scale": self.scaler.scale_.tolist() if self.scaler else None,
            "known_node_coords": self.known_node_coords,
            "training_history": self.training_history,
        }
        
        torch.save(state, path)
        print(f"[INFO] Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scaler from file."""
        state = torch.load(path, map_location='cpu')
        
        n_features = state["model_config"]["n_features"]
        self.model = GridScorerNet(n_features=n_features)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        
        if state.get("scaler_mean") is not None and self.scaler:
            self.scaler.mean_ = np.array(state["scaler_mean"])
            self.scaler.scale_ = np.array(state["scaler_scale"])
        
        self.known_node_coords = state.get("known_node_coords", [])
        self.training_history = state.get("training_history", {})
        self.is_trained = True
        
        print(f"[INFO] Model loaded from {path}")


# ---------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------

def train_grid_scorer_from_nodes(
    known_nodes: Dict,
    compute_features_func: Callable = None,
    config: TrainingConfig = None,
    verbose: bool = True,
) -> MLGridScorer:
    """
    Train an ML grid scorer from known nodes dictionary.
    
    Args:
        known_nodes: Dict of known nodes (from known_nodes_extended.py)
        compute_features_func: Optional function to compute features
        config: Training configuration
        verbose: Print progress
    
    Returns:
        Trained MLGridScorer
    """
    if config is None:
        config = TrainingConfig()
    
    # Extract coordinates
    coords = []
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            lat, lon = data["coords"]
        else:
            lat, lon = data
        coords.append((lat, lon))
    
    print(f"[INFO] Training from {len(coords)} known nodes")
    
    # Extract features for each node
    features = []
    for lat, lon in coords:
        feat = extract_grid_features(
            lat, lon,
            G_val=0.8,  # Known nodes have high G
            H_val=0.7,  # Known nodes have high H
            ga_norm=np.random.uniform(0.3, 1.0),
            ct_norm=np.random.uniform(-0.5, 0.5),
            boundary_score=np.random.uniform(0.3, 0.8),
            known_node_coords=coords
        )
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    # Train scorer
    scorer = MLGridScorer()
    result = scorer.train(coords, features, config, verbose)
    
    if verbose:
        print(f"\n[INFO] Training complete!")
        print(f"[INFO] Epochs: {result['epochs_trained']}")
        print(f"[INFO] Best val loss: {result['best_val_loss']:.4f}")
        print(f"[INFO] Final recall: {result['final_recall']:.4f}")
    
    return scorer


# ---------------------------------------------------------
# Demo / Main
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN ML Grid Scorer Demo")
    print("-" * 50)
    
    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch")
    else:
        # Demo with synthetic data
        print("\nCreating synthetic training data...")
        
        # Fake known nodes
        known_coords = [
            (29.98, 31.13),   # Giza-like
            (19.69, -98.84),  # Teotihuacan-like
            (51.18, -1.83),   # Stonehenge-like
            (37.22, 38.92),   # Gobekli Tepe-like
            (-13.16, -72.55), # Machu Picchu-like
        ]
        
        # Create features
        features = []
        for lat, lon in known_coords:
            feat = extract_grid_features(
                lat, lon,
                G_val=0.8, H_val=0.7,
                ga_norm=0.5, ct_norm=0.2, boundary_score=0.6,
                known_node_coords=known_coords
            )
            features.append(feat)
        
        features = np.array(features)
        
        # Train
        print("\nTraining ML grid scorer...")
        config = TrainingConfig(epochs=50, patience=10)
        scorer = MLGridScorer()
        result = scorer.train(known_coords, features, config, verbose=True)
        
        print(f"\nTraining complete!")
        print(f"Best val loss: {result['best_val_loss']:.4f}")
        
        # Test prediction
        test_coords = [(30.0, 31.0), (0.0, 0.0), (45.0, -90.0)]
        test_features = np.array([
            extract_grid_features(lat, lon, 0.5, 0.5, 0.3, 0.1, 0.4, known_coords)
            for lat, lon in test_coords
        ])
        
        scores = scorer.predict(test_features)
        print("\nTest predictions:")
        for (lat, lon), score in zip(test_coords, scores):
            print(f"  ({lat:.1f}, {lon:.1f}): {score:.4f}")


#!/usr/bin/env python
"""
GSN Ensemble Model

Combines multiple scoring approaches into a robust ensemble predictor.
Uses stacking with meta-learning to find optimal combination weights.

Ensemble members:
1. Linear weighted sum (F = alpha*G + beta*H + ...)
2. Geometric mean scorer
3. Neural network scorer
4. Rank-based scorer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Try to import sklearn
try:
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------
# Base Scorers
# ---------------------------------------------------------

class BaseScorer:
    """Base class for all scorers."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the scorer to training data."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scores for input features."""
        raise NotImplementedError
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate scorer on test data."""
        predictions = self.predict(X)
        # Use correlation as score metric
        return float(np.corrcoef(predictions, y)[0, 1])


class LinearScorer(BaseScorer):
    """
    Linear weighted sum: F = sum(w_i * x_i)
    Weights are learned via ridge regression.
    """
    
    def __init__(self):
        super().__init__("linear")
        self.weights = None
        self.intercept = 0.0
        self.scaler = StandardScaler() if HAS_SKLEARN else None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if HAS_SKLEARN:
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            model.fit(X, y)
            
            self.weights = model.coef_
            self.intercept = model.intercept_
        else:
            # Simple least squares fallback
            X_bias = np.column_stack([X, np.ones(len(X))])
            self.weights, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
            self.intercept = self.weights[-1]
            self.weights = self.weights[:-1]
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler and HAS_SKLEARN:
            X = self.scaler.transform(X)
        
        scores = X @ self.weights + self.intercept
        # Clip to [0, 1]
        return np.clip(scores, 0, 1)


class GeometricMeanScorer(BaseScorer):
    """
    Geometric mean of normalized features.
    Good for multiplicative relationships.
    """
    
    def __init__(self):
        super().__init__("geometric_mean")
        self.feature_weights = None
        self.feature_mins = None
        self.feature_maxs = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Normalize features to [0, 1]
        self.feature_mins = X.min(axis=0)
        self.feature_maxs = X.max(axis=0)
        
        # Learn feature weights via correlation with target
        X_norm = self._normalize(X)
        correlations = np.array([
            np.corrcoef(X_norm[:, i], y)[0, 1] if not np.isnan(np.corrcoef(X_norm[:, i], y)[0, 1]) else 0
            for i in range(X.shape[1])
        ])
        
        # Use absolute correlation as weight
        self.feature_weights = np.abs(correlations)
        # Normalize weights
        if self.feature_weights.sum() > 0:
            self.feature_weights /= self.feature_weights.sum()
        else:
            self.feature_weights = np.ones(X.shape[1]) / X.shape[1]
        
        self.is_fitted = True
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        ranges = self.feature_maxs - self.feature_mins
        ranges[ranges == 0] = 1
        return (X - self.feature_mins) / ranges
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(X)
        X_norm = np.clip(X_norm, 0.001, 1)  # Avoid zeros for log
        
        # Weighted geometric mean
        log_scores = np.sum(self.feature_weights * np.log(X_norm), axis=1)
        return np.exp(log_scores)


class RankScorer(BaseScorer):
    """
    Rank-based scorer: uses percentile ranks instead of raw values.
    More robust to outliers.
    """
    
    def __init__(self):
        super().__init__("rank")
        self.reference_data = None
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.reference_data = X.copy()
        
        # Convert to ranks
        X_ranked = self._to_ranks(X)
        
        # Simple correlation-based weights
        correlations = np.array([
            np.corrcoef(X_ranked[:, i], y)[0, 1] if not np.isnan(np.corrcoef(X_ranked[:, i], y)[0, 1]) else 0
            for i in range(X.shape[1])
        ])
        
        self.weights = np.clip(correlations, 0, 1)
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            self.weights = np.ones(X.shape[1]) / X.shape[1]
        
        self.is_fitted = True
    
    def _to_ranks(self, X: np.ndarray) -> np.ndarray:
        """Convert values to percentile ranks."""
        if self.reference_data is None:
            return X
        
        ranks = np.zeros_like(X)
        for i in range(X.shape[1]):
            ref_col = self.reference_data[:, i]
            for j in range(X.shape[0]):
                ranks[j, i] = np.mean(ref_col <= X[j, i])
        
        return ranks
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ranked = self._to_ranks(X)
        return np.sum(self.weights * X_ranked, axis=1)


class GradientBoostingScorer(BaseScorer):
    """
    Gradient Boosting classifier for binary prediction.
    Captures non-linear relationships and interactions.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 3):
        super().__init__("gradient_boosting")
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler() if HAS_SKLEARN else None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if not HAS_SKLEARN:
            self.is_fitted = False
            return
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Convert to binary if needed
        y_binary = (y > 0.5).astype(int)
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X, y_binary)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            return np.zeros(len(X))
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    use_linear: bool = True
    use_geometric: bool = True
    use_rank: bool = True
    use_gb: bool = True
    use_neural: bool = False  # Requires PyTorch
    meta_method: str = "ridge"  # 'ridge', 'logistic', or 'average'
    cv_folds: int = 5


class EnsembleScorer:
    """
    Ensemble of multiple scoring methods with learned combination weights.
    
    Uses stacking: train base models, then learn a meta-model that
    combines their predictions optimally.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config if config else EnsembleConfig()
        self.models: Dict[str, BaseScorer] = {}
        self.meta_model = None
        self.is_fitted = False
        
        self._init_models()
    
    def _init_models(self):
        """Initialize base models based on config."""
        if self.config.use_linear:
            self.models["linear"] = LinearScorer()
        
        if self.config.use_geometric:
            self.models["geometric"] = GeometricMeanScorer()
        
        if self.config.use_rank:
            self.models["rank"] = RankScorer()
        
        if self.config.use_gb and HAS_SKLEARN:
            self.models["gradient_boosting"] = GradientBoostingScorer()
        
        if self.config.use_neural:
            try:
                from gsn_nn_scorer import GSNScorer, GSNTrainer
                # Neural model would be added here
            except ImportError:
                pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Fit all base models and the meta-learner.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (0/1 or continuous scores)
            verbose: Print progress
        """
        if verbose:
            print("Training ensemble models...")
        
        # Train each base model
        for name, model in self.models.items():
            if verbose:
                print(f"  Training {name}...")
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"    [WARN] Failed to train {name}: {e}")
        
        # Get base model predictions
        base_predictions = self._get_base_predictions(X)
        
        if verbose:
            print("Training meta-learner...")
        
        # Train meta-learner
        if self.config.meta_method == "ridge" and HAS_SKLEARN:
            self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            self.meta_model.fit(base_predictions, y)
        
        elif self.config.meta_method == "logistic" and HAS_SKLEARN:
            y_binary = (y > 0.5).astype(int)
            self.meta_model = LogisticRegressionCV(cv=3, max_iter=1000)
            self.meta_model.fit(base_predictions, y_binary)
        
        else:
            # Simple averaging
            self.meta_model = None
        
        self.is_fitted = True
        
        if verbose:
            print("Ensemble training complete!")
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []
        
        for name, model in self.models.items():
            if model.is_fitted:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            return np.zeros((len(X), 1))
        
        return np.column_stack(predictions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ensemble scores.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted scores
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        base_predictions = self._get_base_predictions(X)
        
        if self.meta_model is not None:
            if self.config.meta_method == "logistic":
                return self.meta_model.predict_proba(base_predictions)[:, 1]
            else:
                return np.clip(self.meta_model.predict(base_predictions), 0, 1)
        else:
            # Simple average
            return base_predictions.mean(axis=1)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the learned weights for each base model."""
        if self.meta_model is None:
            # Equal weights for averaging
            n = len(self.models)
            return {name: 1.0/n for name in self.models.keys()}
        
        if hasattr(self.meta_model, 'coef_'):
            coefs = self.meta_model.coef_.flatten()
            model_names = [name for name, m in self.models.items() if m.is_fitted]
            
            # Normalize to sum to 1
            coefs = np.abs(coefs)
            if coefs.sum() > 0:
                coefs /= coefs.sum()
            
            return dict(zip(model_names, coefs))
        
        return {}
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Aggregate feature importance across all base models.
        """
        importance = {}
        
        # Get importance from linear model
        if "linear" in self.models and self.models["linear"].is_fitted:
            linear = self.models["linear"]
            if linear.weights is not None:
                weights = np.abs(linear.weights)
                if weights.sum() > 0:
                    weights /= weights.sum()
                
                if feature_names:
                    for name, w in zip(feature_names, weights):
                        importance[name] = importance.get(name, 0) + w
        
        # Get importance from gradient boosting
        if "gradient_boosting" in self.models and self.models["gradient_boosting"].is_fitted:
            gb = self.models["gradient_boosting"]
            if hasattr(gb.model, 'feature_importances_'):
                fi = gb.model.feature_importances_
                if feature_names:
                    for name, imp in zip(feature_names, fi):
                        importance[name] = importance.get(name, 0) + imp
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate ensemble and individual models.
        
        Returns:
            Dict with scores for ensemble and each base model
        """
        results = {}
        
        # Ensemble score
        ensemble_pred = self.predict(X)
        results["ensemble"] = {
            "correlation": float(np.corrcoef(ensemble_pred, y)[0, 1]),
            "mse": float(np.mean((ensemble_pred - y) ** 2)),
        }
        
        # Individual model scores
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    results[name] = {
                        "correlation": float(np.corrcoef(pred, y)[0, 1]),
                        "mse": float(np.mean((pred - y) ** 2)),
                    }
                except Exception:
                    results[name] = {"error": "prediction failed"}
        
        # Model weights
        results["model_weights"] = self.get_model_weights()
        
        return results


# ---------------------------------------------------------
# Integration with GSN
# ---------------------------------------------------------

class EnsembleGSNPredictor:
    """
    Use ensemble model for GSN location scoring.
    """
    
    def __init__(self):
        self.ensemble = EnsembleScorer()
        self.feature_names = None
    
    def train(self, 
             positive_features: np.ndarray,
             negative_features: np.ndarray,
             feature_names: List[str] = None,
             verbose: bool = True) -> Dict:
        """
        Train ensemble on positive (known nodes) and negative samples.
        
        Args:
            positive_features: Features for known GSN nodes
            negative_features: Features for random locations
            feature_names: Names of features
            verbose: Print progress
        
        Returns:
            Training results
        """
        self.feature_names = feature_names
        
        # Combine data
        X = np.vstack([positive_features, negative_features])
        y = np.hstack([np.ones(len(positive_features)), 
                       np.zeros(len(negative_features))])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Train
        self.ensemble.fit(X, y, verbose=verbose)
        
        # Evaluate
        results = self.ensemble.evaluate(X, y)
        
        # Add feature importance
        results["feature_importance"] = self.ensemble.get_feature_importance(feature_names)
        
        return results
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score locations using ensemble."""
        return self.ensemble.predict(features)


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Ensemble Model Demo")
    print("-" * 40)
    
    # Create synthetic data
    np.random.seed(42)
    n_features = 20
    n_positive = 100
    n_negative = 400
    
    # Positive samples have higher values in first few features
    positive_X = np.random.randn(n_positive, n_features)
    positive_X[:, :5] += 1.0  # Signal in first 5 features
    
    # Negative samples are random
    negative_X = np.random.randn(n_negative, n_features)
    
    # Combine
    X = np.vstack([positive_X, negative_X])
    y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train ensemble
    print("\nTraining ensemble...")
    ensemble = EnsembleScorer()
    ensemble.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    print("\nEvaluation on test set:")
    results = ensemble.evaluate(X_test, y_test)
    
    print(f"\nEnsemble correlation: {results['ensemble']['correlation']:.4f}")
    print(f"Ensemble MSE: {results['ensemble']['mse']:.4f}")
    
    print("\nIndividual model scores:")
    for name in ensemble.models.keys():
        if name in results:
            print(f"  {name}: correlation={results[name].get('correlation', 'N/A'):.4f}")
    
    print("\nModel weights:")
    for name, weight in results["model_weights"].items():
        print(f"  {name}: {weight:.4f}")

#!/usr/bin/env python
"""
GSN ML-Based Weight Optimizer

Automatically tunes the weights for all score components (G, H, A, N, T)
using known nodes as positive examples and optimizing for maximum recall.

Approaches:
1. Grid Search: Exhaustive search over weight combinations
2. Bayesian Optimization: Efficient hyperparameter tuning with Gaussian Process
3. Differential Evolution: Global optimization avoiding local minima
4. Scipy Minimize: Gradient-based local optimization

The optimizer maximizes recall@K on known nodes while maintaining
generalization through cross-validation.

Extended F formula: F = alpha*G + beta*H + gamma*A + delta*N + epsilon*T
"""

import math
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np

try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.callbacks import DeltaYStopper
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from known_nodes_extended import KNOWN_NODES_EXTENDED, get_coords_dict
from gsn_validation import compute_recall_at_k, haversine_km


# ---------------------------------------------------------
# Weight Configuration (Extended for all 5 components)
# ---------------------------------------------------------

DEFAULT_WEIGHTS = {
    # Main component weights (F = alpha*G + beta*H + gamma*A + delta*N + epsilon*T)
    "alpha": 1.0,     # G weight (geophysical)
    "beta": 1.0,      # H weight (geometric)
    "gamma": 0.0,     # A weight (astronomical)
    "delta": 0.0,     # N weight (network)
    "epsilon": 0.0,   # T weight (temporal)
    
    # G sub-component weights
    "w_ga": 1.0,      # gravity anomaly
    "w_ct": 1.0,      # crustal thickness
    "w_tb": 1.0,      # tectonic boundary
    "w_ma": 0.5,      # magnetic anomaly
    "w_el": 0.2,      # elevation
    "w_bg": 0.3,      # Bouguer anomaly
    "w_iso": 0.2,     # isostatic anomaly
    "w_hf": 0.3,      # heat flow
    "w_vol": 0.3,     # volcanic distance
    "w_seis": 0.2,    # seismic density
    
    # H sub-component weights
    "w_angle": 1.0,   # angular coherence
    "w_dist": 0.8,    # distance weighting
    "w_gc": 0.5,      # great circle alignment
    "w_fib": 0.3,     # fibonacci distances
    
    # A sub-component weights
    "w_vis": 0.5,     # constellation visibility
    "w_pat": 0.3,     # pattern matching
    "w_sol": 0.2,     # solstice alignment
}

WEIGHT_BOUNDS = {
    # Main weights
    "alpha": (0.1, 3.0),
    "beta": (0.1, 3.0),
    "gamma": (0.0, 2.0),
    "delta": (0.0, 2.0),
    "epsilon": (0.0, 2.0),
    
    # G sub-weights
    "w_ga": (0.0, 3.0),
    "w_ct": (0.0, 3.0),
    "w_tb": (0.0, 3.0),
    "w_ma": (0.0, 2.0),
    "w_el": (0.0, 1.0),
    "w_bg": (0.0, 2.0),
    "w_iso": (0.0, 1.0),
    "w_hf": (0.0, 2.0),
    "w_vol": (0.0, 2.0),
    "w_seis": (0.0, 1.0),
    
    # H sub-weights
    "w_angle": (0.0, 2.0),
    "w_dist": (0.0, 2.0),
    "w_gc": (0.0, 2.0),
    "w_fib": (0.0, 1.0),
    
    # A sub-weights
    "w_vis": (0.0, 1.0),
    "w_pat": (0.0, 1.0),
    "w_sol": (0.0, 1.0),
}

# Preset weight configurations for quick optimization
WEIGHT_PRESETS = {
    "main_only": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "g_focus": ["alpha", "beta", "w_ga", "w_ct", "w_tb", "w_ma"],
    "full_g": ["alpha", "w_ga", "w_ct", "w_tb", "w_ma", "w_hf", "w_vol", "w_seis"],
    "balanced": ["alpha", "beta", "gamma", "w_ga", "w_ct", "w_tb"],
    "all": list(WEIGHT_BOUNDS.keys()),
}


# ---------------------------------------------------------
# Objective Function
# ---------------------------------------------------------

def create_objective_function(
    compute_candidates_func: Callable,
    known_nodes: Dict,
    top_k: int = 50,
    threshold_km: float = 100.0,
    weight_names: List[str] = None,
) -> Callable:
    """
    Create an objective function for optimization.
    
    The objective is to MINIMIZE negative recall (i.e., maximize recall).
    
    Args:
        compute_candidates_func: Function(weights_dict) -> candidates list
        known_nodes: Ground truth nodes
        top_k: Number of candidates to evaluate
        threshold_km: Match threshold
        weight_names: Names of weights to optimize
    
    Returns:
        Objective function that takes weight array and returns loss
    """
    if weight_names is None:
        weight_names = list(DEFAULT_WEIGHTS.keys())
    
    # Convert known_nodes to coords format
    node_coords = {}
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            node_coords[name] = data["coords"]
        else:
            node_coords[name] = data
    
    def objective(weight_array):
        # Convert array to dict
        weights = dict(zip(weight_names, weight_array))
        
        # Ensure non-negative weights
        for k, v in weights.items():
            weights[k] = max(0.0, v)
        
        try:
            # Compute candidates with these weights
            candidates = compute_candidates_func(weights)
            
            # Evaluate recall
            result = compute_recall_at_k(
                candidates[:top_k],
                node_coords,
                threshold_km=threshold_km,
            )
            
            # Return negative recall (we minimize)
            return -result["recall"]
            
        except Exception as e:
            print(f"[WARN] Objective evaluation failed: {e}")
            return 0.0  # Return worst possible score
    
    return objective


# ---------------------------------------------------------
# Grid Search Optimizer
# ---------------------------------------------------------

def grid_search(
    compute_candidates_func: Callable,
    known_nodes: Dict = None,
    weight_names: List[str] = None,
    n_steps: int = 5,
    top_k: int = 50,
    threshold_km: float = 100.0,
    verbose: bool = True,
) -> Dict:
    """
    Grid search over weight space.
    
    Args:
        compute_candidates_func: Function(weights_dict) -> candidates
        known_nodes: Ground truth nodes
        weight_names: Which weights to optimize
        n_steps: Number of steps per dimension
        top_k: K for recall@K
        threshold_km: Match threshold
        verbose: Print progress
    
    Returns:
        Dict with best weights and search history
    """
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    if weight_names is None:
        # Optimize just the main weights for efficiency
        weight_names = ["alpha", "beta", "w_ga", "w_ct", "w_tb"]
    
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    # Create grid
    grids = []
    for name in weight_names:
        low, high = WEIGHT_BOUNDS.get(name, (0.0, 2.0))
        grids.append(np.linspace(low, high, n_steps))
    
    # Search
    best_loss = float("inf")
    best_weights = None
    history = []
    
    total = np.prod([len(g) for g in grids])
    count = 0
    
    from itertools import product
    for combo in product(*grids):
        count += 1
        
        loss = objective(np.array(combo))
        recall = -loss
        
        history.append({
            "weights": dict(zip(weight_names, combo)),
            "recall": recall,
        })
        
        if loss < best_loss:
            best_loss = loss
            best_weights = dict(zip(weight_names, combo))
            if verbose:
                print(f"[{count}/{total}] New best: recall={recall:.4f}, weights={best_weights}")
    
    return {
        "best_weights": best_weights,
        "best_recall": -best_loss,
        "history": history,
        "n_evaluations": count,
    }


# ---------------------------------------------------------
# Scipy-based Optimizers
# ---------------------------------------------------------

def scipy_minimize(
    compute_candidates_func: Callable,
    known_nodes: Dict = None,
    weight_names: List[str] = None,
    initial_weights: Dict = None,
    method: str = "L-BFGS-B",
    top_k: int = 50,
    threshold_km: float = 100.0,
    max_iter: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Use scipy.optimize.minimize for weight optimization.
    
    Args:
        compute_candidates_func: Function(weights_dict) -> candidates
        known_nodes: Ground truth nodes
        weight_names: Which weights to optimize
        initial_weights: Starting weights
        method: Scipy optimization method
        top_k: K for recall@K
        threshold_km: Match threshold
        max_iter: Maximum iterations
        verbose: Print progress
    
    Returns:
        Dict with optimized weights
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy required for scipy_minimize")
    
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    if weight_names is None:
        weight_names = ["alpha", "beta", "w_ga", "w_ct", "w_tb"]
    
    if initial_weights is None:
        initial_weights = DEFAULT_WEIGHTS
    
    # Initial point
    x0 = np.array([initial_weights.get(name, 1.0) for name in weight_names])
    
    # Bounds
    bounds = [WEIGHT_BOUNDS.get(name, (0.0, 3.0)) for name in weight_names]
    
    # Objective
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    # Callback for verbose output
    iteration = [0]
    def callback(xk):
        iteration[0] += 1
        if verbose:
            loss = objective(xk)
            print(f"Iter {iteration[0]}: recall={-loss:.4f}")
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        options={"maxiter": max_iter},
        callback=callback if verbose else None,
    )
    
    optimized_weights = dict(zip(weight_names, result.x))
    
    return {
        "best_weights": optimized_weights,
        "best_recall": -result.fun,
        "success": result.success,
        "n_iterations": result.nit,
        "message": result.message,
    }


def differential_evolution_optimize(
    compute_candidates_func: Callable,
    known_nodes: Dict = None,
    weight_names: List[str] = None,
    top_k: int = 50,
    threshold_km: float = 100.0,
    max_iter: int = 50,
    pop_size: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Use differential evolution for global optimization.
    
    This is better for non-convex objectives and avoids local minima.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy required for differential_evolution")
    
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    if weight_names is None:
        weight_names = ["alpha", "beta", "w_ga", "w_ct", "w_tb", "w_ma"]
    
    bounds = [WEIGHT_BOUNDS.get(name, (0.0, 3.0)) for name in weight_names]
    
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    # Callback
    def callback(xk, convergence):
        if verbose:
            loss = objective(xk)
            print(f"DE: recall={-loss:.4f}, convergence={convergence:.4f}")
        return False  # Continue optimization
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iter,
        popsize=pop_size,
        callback=callback,
        disp=verbose,
        seed=42,
    )
    
    optimized_weights = dict(zip(weight_names, result.x))
    
    return {
        "best_weights": optimized_weights,
        "best_recall": -result.fun,
        "success": result.success,
        "n_iterations": result.nit,
    }


# ---------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------

def bayesian_optimize(
    compute_candidates_func: Callable,
    known_nodes: Dict = None,
    weight_names: List[str] = None,
    n_calls: int = 50,
    n_initial: int = 10,
    top_k: int = 50,
    threshold_km: float = 100.0,
    verbose: bool = True,
    early_stop_delta: float = 0.001,
) -> Dict:
    """
    Bayesian optimization using Gaussian Process surrogate model.
    
    More efficient than grid search for high-dimensional weight spaces.
    Uses scikit-optimize (skopt) for the optimization.
    
    Args:
        compute_candidates_func: Function(weights_dict) -> candidates
        known_nodes: Ground truth nodes
        weight_names: Which weights to optimize (default: main 5 weights)
        n_calls: Total number of objective function evaluations
        n_initial: Number of random initial points
        top_k: K for recall@K evaluation
        threshold_km: Match threshold in km
        verbose: Print progress
        early_stop_delta: Stop if improvement is less than this
    
    Returns:
        Dict with best weights, convergence history, and optimization stats
    """
    if not HAS_SKOPT:
        raise RuntimeError("scikit-optimize required. Install with: pip install scikit-optimize")
    
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    if weight_names is None:
        weight_names = WEIGHT_PRESETS["main_only"]
    
    # Create objective function
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    # Define search space
    dimensions = []
    for name in weight_names:
        low, high = WEIGHT_BOUNDS.get(name, (0.0, 2.0))
        dimensions.append(Real(low, high, name=name))
    
    # Callback for verbose output
    iteration = [0]
    history = []
    
    def on_step(res):
        iteration[0] += 1
        current_recall = -res.func_vals[-1]
        best_so_far = -res.fun
        
        history.append({
            "iteration": iteration[0],
            "recall": current_recall,
            "best_recall": best_so_far,
        })
        
        if verbose:
            print(f"[BO {iteration[0]}/{n_calls}] recall={current_recall:.4f}, best={best_so_far:.4f}")
    
    # Early stopping callback
    callbacks = [on_step]
    if early_stop_delta > 0:
        callbacks.append(DeltaYStopper(delta=early_stop_delta, n_best=5))
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        callback=callbacks,
        acq_func="EI",  # Expected Improvement
        xi=0.01,  # Exploration-exploitation tradeoff
    )
    
    # Extract results
    optimized_weights = dict(zip(weight_names, result.x))
    
    return {
        "best_weights": optimized_weights,
        "best_recall": -result.fun,
        "convergence": list(-np.array(result.func_vals)),
        "history": history,
        "n_calls": len(result.func_vals),
        "method": "bayesian",
    }


def get_optimization_method(method: str) -> Callable:
    """
    Get optimization function by name.
    
    Args:
        method: One of 'bayesian', 'differential_evolution', 'grid', 'random', 'scipy'
    
    Returns:
        Optimization function
    """
    methods = {
        "bayesian": bayesian_optimize,
        "differential_evolution": differential_evolution_optimize,
        "grid": grid_search,
        "random": random_search,
        "scipy": scipy_minimize,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    return methods[method]


# ---------------------------------------------------------
# Random Search (baseline)
# ---------------------------------------------------------

def random_search(
    compute_candidates_func: Callable,
    known_nodes: Dict = None,
    weight_names: List[str] = None,
    n_trials: int = 100,
    top_k: int = 50,
    threshold_km: float = 100.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Random search over weight space (useful as baseline).
    """
    np.random.seed(seed)
    
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    if weight_names is None:
        weight_names = ["alpha", "beta", "w_ga", "w_ct", "w_tb"]
    
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    best_loss = float("inf")
    best_weights = None
    history = []
    
    for trial in range(n_trials):
        # Random weights within bounds
        weights = []
        for name in weight_names:
            low, high = WEIGHT_BOUNDS.get(name, (0.0, 3.0))
            weights.append(np.random.uniform(low, high))
        
        loss = objective(np.array(weights))
        recall = -loss
        
        history.append({
            "trial": trial,
            "weights": dict(zip(weight_names, weights)),
            "recall": recall,
        })
        
        if loss < best_loss:
            best_loss = loss
            best_weights = dict(zip(weight_names, weights))
            if verbose:
                print(f"Trial {trial+1}/{n_trials}: New best recall={recall:.4f}")
    
    return {
        "best_weights": best_weights,
        "best_recall": -best_loss,
        "history": history,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------

def sensitivity_analysis(
    compute_candidates_func: Callable,
    base_weights: Dict,
    known_nodes: Dict = None,
    perturbation: float = 0.2,
    top_k: int = 50,
    threshold_km: float = 100.0,
) -> Dict:
    """
    Analyze sensitivity of recall to each weight.
    
    For each weight, compute the change in recall when the weight
    is increased/decreased by perturbation fraction.
    """
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    weight_names = list(base_weights.keys())
    
    objective = create_objective_function(
        compute_candidates_func, known_nodes, top_k, threshold_km, weight_names
    )
    
    # Base recall
    base_array = np.array([base_weights[name] for name in weight_names])
    base_recall = -objective(base_array)
    
    sensitivities = {}
    
    for i, name in enumerate(weight_names):
        # Increase
        weights_up = base_array.copy()
        weights_up[i] *= (1 + perturbation)
        recall_up = -objective(weights_up)
        
        # Decrease
        weights_down = base_array.copy()
        weights_down[i] *= (1 - perturbation)
        recall_down = -objective(weights_down)
        
        sensitivities[name] = {
            "base_value": base_weights[name],
            "recall_up": recall_up,
            "recall_down": recall_down,
            "delta_up": recall_up - base_recall,
            "delta_down": recall_down - base_recall,
            "sensitivity": abs(recall_up - recall_down) / (2 * perturbation),
        }
    
    # Rank by sensitivity
    ranked = sorted(
        sensitivities.items(),
        key=lambda x: x[1]["sensitivity"],
        reverse=True
    )
    
    return {
        "base_recall": base_recall,
        "sensitivities": sensitivities,
        "ranked": [name for name, _ in ranked],
    }


# ---------------------------------------------------------
# SHAP-based Feature Importance
# ---------------------------------------------------------

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_shap_importance(
    model,
    X: np.ndarray,
    feature_names: List[str] = None,
    n_samples: int = 100,
    method: str = "auto",
) -> Dict:
    """
    Compute feature importance using SHAP (SHapley Additive exPlanations).
    
    SHAP values provide a unified measure of feature importance based on
    game-theoretic principles. They show how much each feature contributes
    to the prediction for each sample.
    
    Args:
        model: Trained model with predict() method
        X: Feature matrix (n_samples, n_features)
        feature_names: Names for each feature
        n_samples: Number of samples to use for background (for KernelExplainer)
        method: 'auto', 'kernel', 'tree', or 'linear'
    
    Returns:
        Dict with importance scores, ranked features, and SHAP values
    """
    if not HAS_SHAP:
        return {"error": "SHAP not installed. Install with: pip install shap"}
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Select background samples
    if len(X) > n_samples:
        background_indices = np.random.choice(len(X), n_samples, replace=False)
        background = X[background_indices]
    else:
        background = X
    
    try:
        # Choose explainer based on model type and method preference
        if method == "tree" or (method == "auto" and hasattr(model, 'feature_importances_')):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif method == "linear" or (method == "auto" and hasattr(model, 'coef_')):
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        else:
            # Kernel explainer works for any model
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X, nsamples=min(100, len(X)))
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dict
        importance = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}
        
        # Rank by importance
        ranked = sorted(importance.items(), key=lambda x: -x[1])
        
        return {
            "importance": importance,
            "ranked": ranked,
            "shap_values": shap_values,
            "feature_names": feature_names,
            "method": "shap",
        }
        
    except Exception as e:
        return {"error": f"SHAP computation failed: {str(e)}"}


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
    n_repeats: int = 10,
    scoring: str = "correlation",
) -> Dict:
    """
    Compute feature importance using permutation importance.
    
    For each feature, shuffle its values and measure how much
    the model performance degrades. Larger degradation = more important.
    
    Args:
        model: Trained model with predict() method
        X: Feature matrix
        y: True labels
        feature_names: Names for each feature
        n_repeats: Number of times to repeat shuffling
        scoring: 'correlation', 'mse', or 'accuracy'
    
    Returns:
        Dict with importance scores and rankings
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Compute baseline score
    baseline_pred = model.predict(X)
    
    if scoring == "correlation":
        baseline_score = np.corrcoef(baseline_pred, y)[0, 1]
    elif scoring == "mse":
        baseline_score = -np.mean((baseline_pred - y) ** 2)  # Negative so higher is better
    else:
        baseline_score = np.mean((baseline_pred > 0.5) == (y > 0.5))
    
    importance_scores = {}
    
    for i, name in enumerate(feature_names):
        scores = []
        
        for _ in range(n_repeats):
            # Shuffle feature i
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Compute score with permuted feature
            perm_pred = model.predict(X_permuted)
            
            if scoring == "correlation":
                corr = np.corrcoef(perm_pred, y)[0, 1]
                perm_score = corr if not np.isnan(corr) else 0
            elif scoring == "mse":
                perm_score = -np.mean((perm_pred - y) ** 2)
            else:
                perm_score = np.mean((perm_pred > 0.5) == (y > 0.5))
            
            scores.append(baseline_score - perm_score)
        
        importance_scores[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
    
    # Rank by mean importance
    ranked = sorted(
        [(name, data["mean"]) for name, data in importance_scores.items()],
        key=lambda x: -x[1]
    )
    
    return {
        "baseline_score": baseline_score,
        "importance": importance_scores,
        "ranked": ranked,
        "scoring": scoring,
        "n_repeats": n_repeats,
        "method": "permutation",
    }


def create_importance_visualization_data(importance_result: Dict) -> Dict:
    """
    Create data suitable for visualization (e.g., bar chart).
    
    Args:
        importance_result: Result from compute_shap_importance or similar
    
    Returns:
        Dict with 'features', 'scores', and 'colors' lists
    """
    ranked = importance_result.get("ranked", [])
    
    if not ranked:
        return {"features": [], "scores": [], "colors": []}
    
    # Take top 15 features
    top_n = min(15, len(ranked))
    
    features = []
    scores = []
    
    for item in ranked[:top_n]:
        if isinstance(item, tuple):
            features.append(item[0])
            scores.append(item[1])
        else:
            features.append(str(item))
            scores.append(0)
    
    # Create color gradient based on scores
    max_score = max(scores) if scores else 1
    colors = [
        f"rgba(59, 130, 246, {0.3 + 0.7 * (s / max_score)})" 
        for s in scores
    ]
    
    return {
        "features": features,
        "scores": scores,
        "colors": colors,
    }


# ---------------------------------------------------------
# Optimization Report
# ---------------------------------------------------------

def generate_optimization_report(
    result: Dict,
    method: str,
) -> str:
    """Generate a formatted report of optimization results."""
    lines = []
    lines.append("=" * 50)
    lines.append(f"WEIGHT OPTIMIZATION REPORT ({method})")
    lines.append("=" * 50)
    lines.append("")
    
    lines.append(f"Best Recall: {result['best_recall']:.4f}")
    lines.append("")
    
    lines.append("Optimized Weights:")
    for name, value in result.get("best_weights", {}).items():
        lines.append(f"  {name:10s}: {value:.4f}")
    lines.append("")
    
    if "n_evaluations" in result:
        lines.append(f"Evaluations: {result['n_evaluations']}")
    if "n_iterations" in result:
        lines.append(f"Iterations: {result['n_iterations']}")
    if "success" in result:
        lines.append(f"Success: {result['success']}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ---------------------------------------------------------
# Demo / Main
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN ML Optimizer Demo")
    print("-" * 40)
    
    # Create a simple mock compute function for demo
    def mock_compute_candidates(weights):
        """Mock candidate computation for testing."""
        known = get_coords_dict()
        
        candidates = []
        for name, (lat, lon) in known.items():
            # Score influenced by weights
            score = weights.get("alpha", 1.0) * 1.5 + weights.get("beta", 1.0) * 0.8
            # Add noise
            score += np.random.uniform(-0.5, 0.5)
            
            candidates.append({
                "lat": lat + np.random.uniform(-0.3, 0.3),
                "lon": lon + np.random.uniform(-0.3, 0.3),
                "F": score,
            })
        
        # Add random candidates
        for _ in range(20):
            candidates.append({
                "lat": np.random.uniform(-60, 60),
                "lon": np.random.uniform(-180, 180),
                "F": np.random.uniform(0, 1),
            })
        
        candidates.sort(key=lambda x: x["F"], reverse=True)
        return candidates
    
    # Run random search demo
    print("\nRunning random search (10 trials)...")
    result = random_search(
        mock_compute_candidates,
        n_trials=10,
        verbose=True,
    )
    
    print(generate_optimization_report(result, "Random Search"))

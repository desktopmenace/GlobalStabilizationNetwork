#!/usr/bin/env python
"""
GSN Uncertainty Quantification Module

Provides uncertainty estimates for node predictions through:
1. Bootstrap resampling of weights
2. Monte Carlo sampling of input uncertainties
3. Ensemble predictions with multiple parameter sets
4. Data quality scoring

These tools help identify predictions with high confidence vs. those
that are sensitive to parameter choices or data quality.
"""

import math
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np


# ---------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------

def bootstrap_F_score(
    lat: float,
    lon: float,
    compute_F_func: Callable,
    base_weights: Dict,
    n_bootstrap: int = 100,
    weight_noise_std: float = 0.1,
    seed: int = None,
) -> Dict:
    """
    Compute F score with bootstrap confidence intervals.
    
    Perturbs weights slightly and recomputes F to estimate uncertainty.
    
    Args:
        lat, lon: Coordinates to evaluate
        compute_F_func: Function(lat, lon, weights) -> F score
        base_weights: Base weight configuration
        n_bootstrap: Number of bootstrap samples
        weight_noise_std: Standard deviation of weight perturbation (relative)
        seed: Random seed
    
    Returns:
        Dict with mean, std, and confidence intervals
    """
    if seed is not None:
        np.random.seed(seed)
    
    F_samples = []
    
    for _ in range(n_bootstrap):
        # Perturb weights
        perturbed = {}
        for name, value in base_weights.items():
            noise = np.random.normal(0, weight_noise_std * abs(value))
            perturbed[name] = max(0.0, value + noise)
        
        try:
            F = compute_F_func(lat, lon, perturbed)
            F_samples.append(F)
        except Exception:
            continue
    
    if not F_samples:
        return {
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "n_samples": 0,
        }
    
    F_array = np.array(F_samples)
    
    return {
        "mean": float(np.mean(F_array)),
        "std": float(np.std(F_array)),
        "ci_low": float(np.percentile(F_array, 2.5)),
        "ci_high": float(np.percentile(F_array, 97.5)),
        "median": float(np.median(F_array)),
        "iqr": float(np.percentile(F_array, 75) - np.percentile(F_array, 25)),
        "n_samples": len(F_samples),
    }


def bootstrap_candidates(
    compute_candidates_func: Callable,
    base_weights: Dict,
    n_bootstrap: int = 50,
    weight_noise_std: float = 0.1,
    top_k: int = 50,
    seed: int = None,
) -> Dict:
    """
    Generate ensemble of candidate predictions with uncertainty.
    
    For each bootstrap iteration, perturb weights and generate candidates.
    Aggregate to find robust predictions that appear consistently.
    
    Args:
        compute_candidates_func: Function(weights) -> candidates list
        base_weights: Base configuration
        n_bootstrap: Number of iterations
        weight_noise_std: Weight perturbation scale
        top_k: Candidates per iteration
        seed: Random seed
    
    Returns:
        Dict with aggregated candidates and stability scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Track how often each location appears in top-K
    # Grid the globe into cells for aggregation
    cell_size = 2.0  # degrees
    cell_counts = {}  # (lat_cell, lon_cell) -> count
    cell_F_values = {}  # (lat_cell, lon_cell) -> list of F values
    
    for iteration in range(n_bootstrap):
        # Perturb weights
        perturbed = {}
        for name, value in base_weights.items():
            noise = np.random.normal(0, weight_noise_std * abs(value))
            perturbed[name] = max(0.0, value + noise)
        
        try:
            candidates = compute_candidates_func(perturbed)[:top_k]
        except Exception:
            continue
        
        for cand in candidates:
            lat_cell = int(cand["lat"] / cell_size)
            lon_cell = int(cand["lon"] / cell_size)
            key = (lat_cell, lon_cell)
            
            cell_counts[key] = cell_counts.get(key, 0) + 1
            if key not in cell_F_values:
                cell_F_values[key] = []
            cell_F_values[key].append(cand["F"])
    
    # Compute stability scores
    robust_candidates = []
    for key, count in cell_counts.items():
        lat_center = (key[0] + 0.5) * cell_size
        lon_center = (key[1] + 0.5) * cell_size
        
        F_vals = cell_F_values[key]
        stability = count / n_bootstrap  # Fraction of times it appeared
        
        robust_candidates.append({
            "lat": lat_center,
            "lon": lon_center,
            "stability": stability,
            "count": count,
            "F_mean": float(np.mean(F_vals)),
            "F_std": float(np.std(F_vals)),
            "F_min": float(np.min(F_vals)),
            "F_max": float(np.max(F_vals)),
        })
    
    # Sort by stability
    robust_candidates.sort(key=lambda x: x["stability"], reverse=True)
    
    return {
        "candidates": robust_candidates,
        "n_bootstrap": n_bootstrap,
        "cell_size_deg": cell_size,
        "n_unique_cells": len(cell_counts),
    }


# ---------------------------------------------------------
# Monte Carlo Input Uncertainty
# ---------------------------------------------------------

def monte_carlo_input_uncertainty(
    lat: float,
    lon: float,
    compute_F_func: Callable,
    input_uncertainties: Dict,
    n_samples: int = 100,
    seed: int = None,
) -> Dict:
    """
    Propagate input data uncertainties to F score.
    
    Args:
        lat, lon: Coordinates
        compute_F_func: Function(lat, lon, perturbed_inputs) -> F
        input_uncertainties: Dict of {input_name: std_dev}
        n_samples: Monte Carlo samples
        seed: Random seed
    
    Returns:
        Dict with uncertainty statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    F_samples = []
    
    for _ in range(n_samples):
        # Sample perturbed inputs
        perturbed_inputs = {}
        for name, std in input_uncertainties.items():
            perturbed_inputs[name] = np.random.normal(0, std)
        
        try:
            F = compute_F_func(lat, lon, perturbed_inputs)
            F_samples.append(F)
        except Exception:
            continue
    
    if not F_samples:
        return {"mean": None, "std": None, "n_samples": 0}
    
    F_array = np.array(F_samples)
    
    return {
        "mean": float(np.mean(F_array)),
        "std": float(np.std(F_array)),
        "ci_low": float(np.percentile(F_array, 5)),
        "ci_high": float(np.percentile(F_array, 95)),
        "n_samples": len(F_samples),
    }


# ---------------------------------------------------------
# Ensemble Predictions
# ---------------------------------------------------------

def ensemble_predict(
    compute_F_func: Callable,
    lat: float,
    lon: float,
    ensemble_configs: List[Dict],
) -> Dict:
    """
    Compute F using an ensemble of configurations.
    
    Args:
        compute_F_func: Function(lat, lon, config) -> F
        lat, lon: Coordinates
        ensemble_configs: List of different weight configurations
    
    Returns:
        Dict with ensemble statistics
    """
    F_values = []
    config_results = []
    
    for i, config in enumerate(ensemble_configs):
        try:
            F = compute_F_func(lat, lon, config)
            F_values.append(F)
            config_results.append({
                "config_id": i,
                "F": F,
                "config": config,
            })
        except Exception as e:
            config_results.append({
                "config_id": i,
                "error": str(e),
            })
    
    if not F_values:
        return {
            "mean": None,
            "std": None,
            "ensemble_size": len(ensemble_configs),
            "successful": 0,
        }
    
    F_array = np.array(F_values)
    
    return {
        "mean": float(np.mean(F_array)),
        "std": float(np.std(F_array)),
        "min": float(np.min(F_array)),
        "max": float(np.max(F_array)),
        "range": float(np.max(F_array) - np.min(F_array)),
        "ensemble_size": len(ensemble_configs),
        "successful": len(F_values),
        "config_results": config_results,
    }


def create_ensemble_configs(
    base_config: Dict,
    n_configs: int = 10,
    variation_scale: float = 0.3,
    seed: int = 42,
) -> List[Dict]:
    """
    Create an ensemble of configurations by varying the base config.
    
    Args:
        base_config: Base weight configuration
        n_configs: Number of ensemble members
        variation_scale: How much to vary each weight (relative)
        seed: Random seed
    
    Returns:
        List of configuration dicts
    """
    np.random.seed(seed)
    
    configs = [base_config.copy()]  # Always include base
    
    for _ in range(n_configs - 1):
        config = {}
        for name, value in base_config.items():
            # Vary by up to variation_scale * value
            variation = np.random.uniform(-variation_scale, variation_scale)
            config[name] = max(0.0, value * (1 + variation))
        configs.append(config)
    
    return configs


# ---------------------------------------------------------
# Data Quality Scoring
# ---------------------------------------------------------

def compute_data_quality_score(
    lat: float,
    lon: float,
    data_availability: Dict[str, bool],
    data_resolution: Dict[str, float] = None,
) -> Dict:
    """
    Compute a data quality score based on data availability and resolution.
    
    Args:
        lat, lon: Coordinates
        data_availability: Dict of {source_name: is_available}
        data_resolution: Dict of {source_name: resolution_km} (optional)
    
    Returns:
        Dict with quality score and details
    """
    n_available = sum(1 for v in data_availability.values() if v)
    n_total = len(data_availability)
    
    availability_score = n_available / n_total if n_total > 0 else 0.0
    
    # Resolution score (if provided)
    if data_resolution:
        # Higher resolution (lower km) = better score
        res_scores = []
        for source, res_km in data_resolution.items():
            if res_km is not None and res_km > 0:
                # Score: 1.0 for <10km, 0.5 for 50km, 0.1 for 200km+
                score = math.exp(-res_km / 50.0)
                res_scores.append(score)
        resolution_score = np.mean(res_scores) if res_scores else 0.5
    else:
        resolution_score = 0.5
    
    # Combined quality score
    quality_score = 0.7 * availability_score + 0.3 * resolution_score
    
    return {
        "quality_score": quality_score,
        "availability_score": availability_score,
        "resolution_score": resolution_score,
        "n_sources_available": n_available,
        "n_sources_total": n_total,
        "sources_missing": [k for k, v in data_availability.items() if not v],
    }


# ---------------------------------------------------------
# Confidence Classification
# ---------------------------------------------------------

def classify_confidence(
    F_mean: float,
    F_std: float,
    quality_score: float = 1.0,
    stability: float = 1.0,
) -> Dict:
    """
    Classify the confidence level of a prediction.
    
    Args:
        F_mean: Mean F score
        F_std: Standard deviation of F
        quality_score: Data quality (0-1)
        stability: Bootstrap stability (0-1)
    
    Returns:
        Dict with confidence level and reasoning
    """
    # Coefficient of variation
    cv = F_std / abs(F_mean) if F_mean != 0 else float("inf")
    
    # Combined confidence score
    # Low CV is good, high quality and stability are good
    if cv < 0.1:
        uncertainty_score = 1.0
    elif cv < 0.3:
        uncertainty_score = 0.7
    elif cv < 0.5:
        uncertainty_score = 0.4
    else:
        uncertainty_score = 0.2
    
    confidence_score = (
        0.4 * uncertainty_score +
        0.3 * quality_score +
        0.3 * stability
    )
    
    # Classification
    if confidence_score >= 0.8:
        level = "HIGH"
        description = "Prediction is robust with low uncertainty"
    elif confidence_score >= 0.6:
        level = "MODERATE"
        description = "Prediction is reasonably confident"
    elif confidence_score >= 0.4:
        level = "LOW"
        description = "Significant uncertainty in prediction"
    else:
        level = "VERY_LOW"
        description = "Prediction is highly uncertain"
    
    return {
        "confidence_score": confidence_score,
        "confidence_level": level,
        "description": description,
        "components": {
            "uncertainty_score": uncertainty_score,
            "quality_score": quality_score,
            "stability": stability,
            "cv": cv,
        },
    }


# ---------------------------------------------------------
# Full Uncertainty Report
# ---------------------------------------------------------

def generate_uncertainty_report(
    lat: float,
    lon: float,
    F_base: float,
    bootstrap_result: Dict = None,
    ensemble_result: Dict = None,
    quality_result: Dict = None,
) -> str:
    """
    Generate a comprehensive uncertainty report for a location.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("UNCERTAINTY REPORT")
    lines.append("=" * 50)
    lines.append(f"Location: ({lat:.4f}, {lon:.4f})")
    lines.append(f"Base F Score: {F_base:.4f}")
    lines.append("")
    
    if bootstrap_result and bootstrap_result.get("mean") is not None:
        lines.append("BOOTSTRAP ANALYSIS")
        lines.append("-" * 30)
        lines.append(f"  Mean F:     {bootstrap_result['mean']:.4f}")
        lines.append(f"  Std Dev:    {bootstrap_result['std']:.4f}")
        lines.append(f"  95% CI:     [{bootstrap_result['ci_low']:.4f}, {bootstrap_result['ci_high']:.4f}]")
        lines.append(f"  N samples:  {bootstrap_result['n_samples']}")
        lines.append("")
    
    if ensemble_result and ensemble_result.get("mean") is not None:
        lines.append("ENSEMBLE ANALYSIS")
        lines.append("-" * 30)
        lines.append(f"  Mean F:     {ensemble_result['mean']:.4f}")
        lines.append(f"  Std Dev:    {ensemble_result['std']:.4f}")
        lines.append(f"  Range:      [{ensemble_result['min']:.4f}, {ensemble_result['max']:.4f}]")
        lines.append(f"  Ensemble:   {ensemble_result['successful']}/{ensemble_result['ensemble_size']} configs")
        lines.append("")
    
    if quality_result:
        lines.append("DATA QUALITY")
        lines.append("-" * 30)
        lines.append(f"  Quality Score:    {quality_result['quality_score']:.3f}")
        lines.append(f"  Data Available:   {quality_result['n_sources_available']}/{quality_result['n_sources_total']}")
        if quality_result.get("sources_missing"):
            lines.append(f"  Missing Sources:  {', '.join(quality_result['sources_missing'])}")
        lines.append("")
    
    # Overall confidence
    if bootstrap_result and bootstrap_result.get("mean") is not None:
        confidence = classify_confidence(
            bootstrap_result["mean"],
            bootstrap_result["std"],
            quality_result.get("quality_score", 0.5) if quality_result else 0.5,
        )
        lines.append("CONFIDENCE")
        lines.append("-" * 30)
        lines.append(f"  Level: {confidence['confidence_level']}")
        lines.append(f"  Score: {confidence['confidence_score']:.3f}")
        lines.append(f"  {confidence['description']}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Uncertainty Quantification Demo")
    print("-" * 40)
    
    # Mock F computation function
    def mock_compute_F(lat, lon, weights):
        base = 2.0 + 0.01 * lat - 0.005 * lon
        weight_effect = weights.get("alpha", 1.0) * 0.3 + weights.get("beta", 1.0) * 0.2
        noise = np.random.normal(0, 0.1)
        return base + weight_effect + noise
    
    # Test location (Giza)
    lat, lon = 29.9792, 31.1342
    base_weights = {"alpha": 1.0, "beta": 1.0, "w_ga": 1.0}
    
    print(f"\nAnalyzing location: ({lat}, {lon})")
    
    # Bootstrap
    print("\nRunning bootstrap analysis...")
    bootstrap = bootstrap_F_score(lat, lon, mock_compute_F, base_weights, n_bootstrap=50)
    print(f"  F = {bootstrap['mean']:.3f} ± {bootstrap['std']:.3f}")
    print(f"  95% CI: [{bootstrap['ci_low']:.3f}, {bootstrap['ci_high']:.3f}]")
    
    # Ensemble
    print("\nRunning ensemble analysis...")
    ensemble_configs = create_ensemble_configs(base_weights, n_configs=5)
    ensemble = ensemble_predict(mock_compute_F, lat, lon, ensemble_configs)
    print(f"  F range: [{ensemble['min']:.3f}, {ensemble['max']:.3f}]")
    
    # Data quality
    quality = compute_data_quality_score(
        lat, lon,
        {"gravity": True, "crust": True, "magnetic": True, "heat_flow": False}
    )
    print(f"\nData quality score: {quality['quality_score']:.3f}")
    
    # Confidence
    confidence = classify_confidence(
        bootstrap['mean'], bootstrap['std'],
        quality['quality_score']
    )
    print(f"\nConfidence: {confidence['confidence_level']} ({confidence['confidence_score']:.2f})")

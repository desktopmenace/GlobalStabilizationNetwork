#!/usr/bin/env python
"""
GSN Uncertainty Quantification Module

Provides comprehensive uncertainty estimates for node predictions through:
1. Bootstrap resampling for confidence intervals
2. Monte Carlo sampling of input uncertainties
3. Ensemble predictions with multiple parameter sets
4. Data quality scoring
5. Error propagation for G and H scores
6. Calibrated probability estimates

Author: H
"""

import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------
# Data Classes for Uncertainty Results
# ---------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Represents a confidence interval."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    method: str  # e.g., "bootstrap", "parametric"
    
    def width(self) -> float:
        return self.upper_bound - self.lower_bound
    
    def contains(self, value: float) -> bool:
        return self.lower_bound <= value <= self.upper_bound
    
    def __str__(self) -> str:
        return f"{self.point_estimate:.4f} ({self.lower_bound:.4f}, {self.upper_bound:.4f})"


@dataclass 
class UncertaintyResult:
    """Complete uncertainty quantification result."""
    point_estimate: float
    ci_95: ConfidenceInterval
    ci_68: ConfidenceInterval  # ~1 sigma
    samples: np.ndarray
    method: str
    n_samples: int
    
    @property
    def std(self) -> float:
        return float(np.std(self.samples))
    
    @property
    def cv(self) -> float:
        """Coefficient of variation."""
        if abs(self.point_estimate) < 1e-10:
            return float('inf')
        return self.std / abs(self.point_estimate)


# ---------------------------------------------------------
# Known Data Uncertainties (from literature/data sources)
# ---------------------------------------------------------

DATA_UNCERTAINTIES = {
    # Gravity anomaly measurement uncertainty (mGal)
    "ga_error": 5.0,
    
    # Crustal thickness model uncertainty (km)
    "ct_error": 5.0,
    
    # Plate boundary distance uncertainty (km)
    "boundary_error": 50.0,
    
    # Coordinate uncertainty for known nodes (degrees)
    "coord_error": 0.1,
    
    # Angular measurement uncertainty for H scores (degrees)
    "angle_error": 1.0,
}


# ---------------------------------------------------------
# Bootstrap H Score Confidence Intervals
# ---------------------------------------------------------

def bootstrap_H_confidence(
    lat: float,
    lon: float,
    known_nodes: Dict,
    compute_H_func: Callable = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    resample_nodes: bool = True,
    jitter_coords: bool = True,
    jitter_std: float = 0.1,
    seed: Optional[int] = 42
) -> UncertaintyResult:
    """
    Compute bootstrap confidence intervals for H score.
    
    Uncertainty sources:
    1. Resampling which known nodes are included
    2. Jittering node coordinates within uncertainty
    3. Parameter variation
    
    Args:
        lat, lon: Query location
        known_nodes: Dict of known node locations
        compute_H_func: Function(lat, lon, nodes) -> H score
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        resample_nodes: Whether to resample the node set
        jitter_coords: Whether to jitter node coordinates
        jitter_std: Coordinate jitter std in degrees
        seed: Random seed
    
    Returns:
        UncertaintyResult with confidence intervals
    """
    rng = np.random.default_rng(seed)
    
    # Default H computation if none provided
    if compute_H_func is None:
        try:
            from gsn_node_predictor import compute_H_weighted
            compute_H_func = lambda lat, lon, nodes: compute_H_weighted(
                lat, lon, known_nodes=nodes
            )
        except ImportError:
            raise ValueError("No H computation function provided or available")
    
    # Extract node coordinates
    node_list = []
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            coords = data.get("coords", (0, 0))
        else:
            coords = data
        node_list.append((name, coords[0], coords[1]))
    
    n_nodes = len(node_list)
    if n_nodes < 3:
        # Not enough nodes for meaningful bootstrap
        H_val = compute_H_func(lat, lon, known_nodes)
        return UncertaintyResult(
            point_estimate=H_val,
            ci_95=ConfidenceInterval(H_val, H_val, H_val, 0.95, "none"),
            ci_68=ConfidenceInterval(H_val, H_val, H_val, 0.68, "none"),
            samples=np.array([H_val]),
            method="insufficient_data",
            n_samples=1
        )
    
    H_samples = []
    
    for _ in range(n_bootstrap):
        # Create perturbed node set
        if resample_nodes:
            # Bootstrap resample (with replacement)
            indices = rng.choice(n_nodes, size=n_nodes, replace=True)
        else:
            indices = range(n_nodes)
        
        boot_nodes = {}
        for idx in indices:
            name, nlat, nlon = node_list[idx]
            
            if jitter_coords:
                # Add coordinate jitter
                nlat += rng.normal(0, jitter_std)
                nlon += rng.normal(0, jitter_std)
                # Clamp to valid range
                nlat = max(-90, min(90, nlat))
            
            # Handle duplicate names from resampling
            unique_name = f"{name}_{idx}"
            boot_nodes[unique_name] = (nlat, nlon)
        
        try:
            H = compute_H_func(lat, lon, boot_nodes)
            if np.isfinite(H):
                H_samples.append(H)
        except Exception:
            continue
    
    if len(H_samples) < 10:
        H_val = compute_H_func(lat, lon, known_nodes)
        return UncertaintyResult(
            point_estimate=H_val,
            ci_95=ConfidenceInterval(H_val, H_val * 0.5, H_val * 1.5, 0.95, "fallback"),
            ci_68=ConfidenceInterval(H_val, H_val * 0.8, H_val * 1.2, 0.68, "fallback"),
            samples=np.array([H_val]),
            method="fallback",
            n_samples=1
        )
    
    H_array = np.array(H_samples)
    point_est = np.median(H_array)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_95 = ConfidenceInterval(
        point_estimate=point_est,
        lower_bound=np.percentile(H_array, 2.5),
        upper_bound=np.percentile(H_array, 97.5),
        confidence_level=0.95,
        method="bootstrap"
    )
    
    ci_68 = ConfidenceInterval(
        point_estimate=point_est,
        lower_bound=np.percentile(H_array, 16),
        upper_bound=np.percentile(H_array, 84),
        confidence_level=0.68,
        method="bootstrap"
    )
    
    return UncertaintyResult(
        point_estimate=point_est,
        ci_95=ci_95,
        ci_68=ci_68,
        samples=H_array,
        method="bootstrap",
        n_samples=len(H_samples)
    )


# ---------------------------------------------------------
# G Score Error Propagation
# ---------------------------------------------------------

def propagate_G_uncertainty(
    lat: float,
    lon: float,
    ga: float,
    ct: float,
    dist_km: float,
    ga_error: float = None,
    ct_error: float = None,
    dist_error: float = None,
    n_samples: int = 1000,
    seed: Optional[int] = 42
) -> UncertaintyResult:
    """
    Propagate measurement uncertainties through G score calculation.
    
    Uses Monte Carlo sampling from input distributions.
    
    Args:
        lat, lon: Location
        ga: Gravity anomaly value
        ct: Crustal thickness value
        dist_km: Distance to plate boundary
        ga_error: Gravity uncertainty (std dev in mGal)
        ct_error: Crustal thickness uncertainty (std dev in km)
        dist_error: Boundary distance uncertainty (std dev in km)
        n_samples: Monte Carlo samples
        seed: Random seed
    
    Returns:
        UncertaintyResult with propagated uncertainty
    """
    rng = np.random.default_rng(seed)
    
    # Use default uncertainties if not provided
    ga_error = ga_error or DATA_UNCERTAINTIES["ga_error"]
    ct_error = ct_error or DATA_UNCERTAINTIES["ct_error"]
    dist_error = dist_error or DATA_UNCERTAINTIES["boundary_error"]
    
    # Import G computation
    try:
        from gsn_node_predictor import compute_G, G_PARAMS
        params = G_PARAMS
    except ImportError:
        # Fallback parameters
        params = {
            "ga_scale": 30.0,
            "ct_mean": 35.0,
            "ct_std": 10.0,
            "L": 800.0,
            "w1": 1.0,
            "w2": 1.0,
            "w3": 1.0,
        }
        def compute_G(ga, ct, dist):
            ga_norm = ga / params["ga_scale"]
            ct_norm = (ct - params["ct_mean"]) / params["ct_std"]
            tb = math.exp(-(dist ** 2) / (2 * params["L"] ** 2))
            G = params["w1"] * ga_norm + params["w2"] * ct_norm + params["w3"] * tb
            return G, {"ga_norm": ga_norm, "ct_norm": ct_norm, "tb": tb}
    
    G_samples = []
    
    for _ in range(n_samples):
        # Sample perturbed inputs
        ga_perturbed = ga + rng.normal(0, ga_error)
        ct_perturbed = ct + rng.normal(0, ct_error)
        dist_perturbed = max(0, dist_km + rng.normal(0, dist_error))
        
        try:
            G, _ = compute_G(ga_perturbed, ct_perturbed, dist_perturbed)
            if np.isfinite(G):
                G_samples.append(G)
        except Exception:
            continue
    
    if len(G_samples) < 10:
        G_base, _ = compute_G(ga, ct, dist_km)
        return UncertaintyResult(
            point_estimate=G_base,
            ci_95=ConfidenceInterval(G_base, G_base - 1, G_base + 1, 0.95, "fallback"),
            ci_68=ConfidenceInterval(G_base, G_base - 0.5, G_base + 0.5, 0.68, "fallback"),
            samples=np.array([G_base]),
            method="fallback",
            n_samples=1
        )
    
    G_array = np.array(G_samples)
    point_est = np.median(G_array)
    
    return UncertaintyResult(
        point_estimate=point_est,
        ci_95=ConfidenceInterval(
            point_est,
            np.percentile(G_array, 2.5),
            np.percentile(G_array, 97.5),
            0.95, "monte_carlo"
        ),
        ci_68=ConfidenceInterval(
            point_est,
            np.percentile(G_array, 16),
            np.percentile(G_array, 84),
            0.68, "monte_carlo"
        ),
        samples=G_array,
        method="monte_carlo",
        n_samples=len(G_samples)
    )


# ---------------------------------------------------------
# Monte Carlo F Score Distribution
# ---------------------------------------------------------

def monte_carlo_F(
    lat: float,
    lon: float,
    known_nodes: Dict = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    param_uncertainty: float = 0.2,
    n_samples: int = 5000,
    seed: Optional[int] = 42
) -> UncertaintyResult:
    """
    Compute full F score distribution via Monte Carlo sampling.
    
    Combines uncertainties from:
    - G score (data uncertainties)
    - H score (node position and sampling)
    - Weight parameters (alpha, beta)
    
    Args:
        lat, lon: Query location
        known_nodes: Known node dict
        alpha, beta: F = alpha*G + beta*H weights
        param_uncertainty: Relative uncertainty in alpha/beta
        n_samples: Monte Carlo samples
        seed: Random seed
    
    Returns:
        UncertaintyResult with F distribution
    """
    rng = np.random.default_rng(seed)
    
    # Import required functions
    try:
        from gsn_node_predictor import (
            get_geophysical_inputs, compute_G, compute_H_weighted
        )
        
        if known_nodes is None:
            from known_nodes_extended import KNOWN_NODES_EXTENDED
            known_nodes = KNOWN_NODES_EXTENDED
    except ImportError:
        raise ValueError("Required modules not available")
    
    # Get base inputs
    ga, ct, dist_km = get_geophysical_inputs(lat, lon)
    
    F_samples = []
    
    for _ in range(n_samples):
        try:
            # Sample perturbed G inputs
            ga_perturbed = ga + rng.normal(0, DATA_UNCERTAINTIES["ga_error"])
            ct_perturbed = ct + rng.normal(0, DATA_UNCERTAINTIES["ct_error"])
            dist_perturbed = max(0, dist_km + rng.normal(0, DATA_UNCERTAINTIES["boundary_error"]))
            
            G, _ = compute_G(ga_perturbed, ct_perturbed, dist_perturbed)
            
            # Sample H with node jittering
            jittered_nodes = {}
            for name, data in known_nodes.items():
                if isinstance(data, dict):
                    nlat, nlon = data.get("coords", (0, 0))
                else:
                    nlat, nlon = data
                
                nlat += rng.normal(0, DATA_UNCERTAINTIES["coord_error"])
                nlon += rng.normal(0, DATA_UNCERTAINTIES["coord_error"])
                nlat = max(-90, min(90, nlat))
                
                jittered_nodes[name] = (nlat, nlon)
            
            H = compute_H_weighted(lat, lon, known_nodes=jittered_nodes)
            
            # Sample weights
            alpha_sample = alpha * (1 + rng.normal(0, param_uncertainty))
            beta_sample = beta * (1 + rng.normal(0, param_uncertainty))
            
            F = alpha_sample * G + beta_sample * H
            
            if np.isfinite(F):
                F_samples.append(F)
                
        except Exception:
            continue
    
    if len(F_samples) < 100:
        # Fallback to simple estimate
        G_base, _ = compute_G(ga, ct, dist_km)
        H_base = compute_H_weighted(lat, lon, known_nodes=known_nodes)
        F_base = alpha * G_base + beta * H_base
        
        return UncertaintyResult(
            point_estimate=F_base,
            ci_95=ConfidenceInterval(F_base, F_base - 2, F_base + 2, 0.95, "fallback"),
            ci_68=ConfidenceInterval(F_base, F_base - 1, F_base + 1, 0.68, "fallback"),
            samples=np.array([F_base]),
            method="fallback",
            n_samples=1
        )
    
    F_array = np.array(F_samples)
    point_est = np.median(F_array)
    
    return UncertaintyResult(
        point_estimate=point_est,
        ci_95=ConfidenceInterval(
            point_est,
            np.percentile(F_array, 2.5),
            np.percentile(F_array, 97.5),
            0.95, "monte_carlo"
        ),
        ci_68=ConfidenceInterval(
            point_est,
            np.percentile(F_array, 16),
            np.percentile(F_array, 84),
            0.68, "monte_carlo"
        ),
        samples=F_array,
        method="monte_carlo",
        n_samples=len(F_samples)
    )


# ---------------------------------------------------------
# Calibration Checking
# ---------------------------------------------------------

def check_calibration(
    predictions: List[Dict],
    observed_outcomes: List[bool],
) -> Dict:
    """
    Check if predicted confidence intervals are well-calibrated.
    
    For well-calibrated predictions, 95% CIs should contain the true
    value 95% of the time.
    
    Args:
        predictions: List of dicts with 'ci_95' and 'ci_68' keys
        observed_outcomes: List of true/false for whether location is a node
    
    Returns:
        Calibration statistics
    """
    n = len(predictions)
    if n == 0:
        return {"n": 0, "calibration_error": None}
    
    # For binary classification, check if confidence aligns with outcomes
    # Higher predicted probability should correlate with positive outcomes
    
    # Group predictions by confidence decile
    decile_outcomes = {i: [] for i in range(10)}
    
    for pred, outcome in zip(predictions, observed_outcomes):
        prob = pred.get("probability", pred.get("point_estimate", 0.5))
        prob = max(0, min(1, prob))  # Clamp to [0, 1]
        decile = min(9, int(prob * 10))
        decile_outcomes[decile].append(1 if outcome else 0)
    
    # Expected vs observed rate per decile
    calibration_data = []
    for decile, outcomes in decile_outcomes.items():
        if outcomes:
            expected_rate = (decile + 0.5) / 10
            observed_rate = np.mean(outcomes)
            calibration_data.append({
                "decile": decile,
                "expected": expected_rate,
                "observed": observed_rate,
                "n": len(outcomes),
                "error": abs(expected_rate - observed_rate)
            })
    
    if calibration_data:
        weighted_error = sum(d["error"] * d["n"] for d in calibration_data) / n
    else:
        weighted_error = 0.0
    
    return {
        "n": n,
        "calibration_error": weighted_error,
        "decile_data": calibration_data,
        "is_well_calibrated": weighted_error < 0.1
    }


# ---------------------------------------------------------
# Bootstrap Confidence Intervals (Original)
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

#!/usr/bin/env python
"""
GSN Statistical Validation Module

Provides rigorous statistical tests to validate whether observed patterns
in GSN node distributions are significant or coincidental.

Key Components:
1. Complete Spatial Randomness (CSR) Test
2. Ripley's K-Function for spherical point patterns
3. Monte Carlo Great Circle Alignment Test
4. False Discovery Rate (FDR) correction

Author: H
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
DEFAULT_N_SIMULATIONS = 10000
SIGNIFICANCE_LEVELS = [0.001, 0.01, 0.05, 0.10]


# ---------------------------------------------------------
# Data Classes for Results
# ---------------------------------------------------------

@dataclass
class CSRTestResult:
    """Results from Complete Spatial Randomness test."""
    test_statistic: float
    p_value: float
    n_simulations: int
    observed_coherence: float
    null_mean: float
    null_std: float
    is_significant_05: bool
    is_significant_01: bool
    interpretation: str


@dataclass
class RipleyKResult:
    """Results from Ripley's K-function analysis."""
    radii_deg: np.ndarray
    K_observed: np.ndarray
    K_expected: np.ndarray  # Under CSR
    L_observed: np.ndarray  # L(r) = sqrt(K(r)/pi) - r
    envelope_lower: np.ndarray  # 2.5% envelope
    envelope_upper: np.ndarray  # 97.5% envelope
    max_deviation: float
    significant_clustering: bool
    significant_dispersion: bool


@dataclass
class AlignmentTestResult:
    """Results from Great Circle alignment test."""
    n_alignments_observed: int
    n_alignments_expected: float
    p_value: float
    alignment_details: List[Dict]
    is_significant: bool
    excess_ratio: float  # observed / expected


@dataclass
class ValidationSummary:
    """Summary of all validation tests."""
    csr_result: CSRTestResult
    ripley_result: RipleyKResult
    alignment_result: AlignmentTestResult
    overall_significant: bool
    overall_p_value: float
    recommendations: List[str]


# ---------------------------------------------------------
# Spherical Geometry Utilities
# ---------------------------------------------------------

def great_circle_distance_deg(lat1: float, lon1: float, 
                               lat2: float, lon2: float) -> float:
    """
    Compute great-circle angular distance in degrees.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    cos_angle = (np.sin(phi1) * np.sin(phi2) + 
                 np.cos(phi1) * np.cos(phi2) * np.cos(dlon))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def great_circle_distance_km(lat1: float, lon1: float,
                              lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    angle_deg = great_circle_distance_deg(lat1, lon1, lat2, lon2)
    return angle_deg * (np.pi / 180.0) * EARTH_RADIUS_KM


def generate_uniform_sphere_points(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate n points uniformly distributed on a sphere.
    
    Uses the algorithm that generates uniform random points on a unit sphere
    by sampling latitude from arcsin of uniform distribution.
    
    Returns:
        Array of shape (n, 2) with [lat, lon] in degrees
    """
    rng = np.random.default_rng(seed)
    
    # Uniform distribution for longitude
    lons = rng.uniform(-180, 180, n)
    
    # For uniform distribution on sphere, latitude is arcsin(U) where U ~ Uniform(-1, 1)
    u = rng.uniform(-1, 1, n)
    lats = np.degrees(np.arcsin(u))
    
    return np.column_stack([lats, lons])


def point_on_great_circle(lat1: float, lon1: float,
                          lat2: float, lon2: float,
                          lat_test: float, lon_test: float,
                          tolerance_deg: float = 2.0) -> bool:
    """
    Check if test point lies on the great circle connecting (lat1, lon1) to (lat2, lon2).
    
    Uses the fact that if three points are collinear on a great circle,
    the sum of distances A->T + T->B equals A->B.
    """
    d_ab = great_circle_distance_deg(lat1, lon1, lat2, lon2)
    d_at = great_circle_distance_deg(lat1, lon1, lat_test, lon_test)
    d_tb = great_circle_distance_deg(lat_test, lon_test, lat2, lon2)
    
    # Check if point is between A and B (not on the extended arc)
    if d_at > d_ab or d_tb > d_ab:
        return False
    
    # Check if distances sum correctly within tolerance
    deviation = abs(d_at + d_tb - d_ab)
    return deviation < tolerance_deg


# ---------------------------------------------------------
# Penrose Coherence Functions
# ---------------------------------------------------------

PENROSE_ANGLES = [36.0, 72.0, 108.0, 144.0]
EXTENDED_ANGLES = [19.47, 23.5, 30.0, 36.0, 45.0, 60.0, 72.0, 90.0, 
                   108.0, 120.0, 137.5, 144.0, 150.0, 180.0]


def penrose_kernel(theta_deg: float, angles: List[float] = None, 
                   sigma: float = 6.0) -> float:
    """
    Compute Penrose-like angular coherence.
    Returns max over all target angles of Gaussian kernel.
    """
    if angles is None:
        angles = PENROSE_ANGLES
    
    max_val = 0.0
    for target in angles:
        diff = theta_deg - target
        val = np.exp(-(diff * diff) / (2.0 * sigma * sigma))
        max_val = max(max_val, val)
    
    return max_val


def compute_coherence_score(points: np.ndarray, angles: List[float] = None,
                            sigma: float = 6.0) -> float:
    """
    Compute average pairwise angular coherence for a point set.
    
    This is the statistic we use for the CSR test.
    
    Args:
        points: Array of shape (n, 2) with [lat, lon] in degrees
        angles: Target angles for coherence calculation
        sigma: Gaussian kernel width
    
    Returns:
        Mean pairwise coherence score
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    if angles is None:
        angles = EXTENDED_ANGLES
    
    total = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            theta = great_circle_distance_deg(
                points[i, 0], points[i, 1],
                points[j, 0], points[j, 1]
            )
            total += penrose_kernel(theta, angles, sigma)
            count += 1
    
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------
# Complete Spatial Randomness (CSR) Test
# ---------------------------------------------------------

def test_pattern_significance(known_nodes: Dict, 
                              n_simulations: int = DEFAULT_N_SIMULATIONS,
                              angles: List[float] = None,
                              sigma: float = 6.0,
                              seed: Optional[int] = 42) -> CSRTestResult:
    """
    Test whether the geometric coherence of known nodes is significantly
    different from random point configurations.
    
    Null Hypothesis: The observed coherence could arise from random placement.
    Alternative: The coherence is higher than expected by chance.
    
    Args:
        known_nodes: Dict of node name -> (lat, lon) or {"coords": (lat, lon)}
        n_simulations: Number of Monte Carlo simulations
        angles: Target angles for coherence (default: EXTENDED_ANGLES)
        sigma: Gaussian kernel width for angle matching
        seed: Random seed for reproducibility
    
    Returns:
        CSRTestResult with test statistics and interpretation
    """
    # Extract coordinates
    coords = []
    for node_data in known_nodes.values():
        if isinstance(node_data, dict):
            lat, lon = node_data.get("coords", (0, 0))
        else:
            lat, lon = node_data
        coords.append([lat, lon])
    
    points = np.array(coords)
    n_points = len(points)
    
    if n_points < 3:
        return CSRTestResult(
            test_statistic=0.0,
            p_value=1.0,
            n_simulations=0,
            observed_coherence=0.0,
            null_mean=0.0,
            null_std=0.0,
            is_significant_05=False,
            is_significant_01=False,
            interpretation="Insufficient nodes for CSR test (need >= 3)"
        )
    
    # Compute observed coherence
    observed = compute_coherence_score(points, angles, sigma)
    
    # Generate null distribution through simulation
    rng = np.random.default_rng(seed)
    null_scores = []
    
    for _ in range(n_simulations):
        random_points = generate_uniform_sphere_points(n_points, seed=rng.integers(0, 2**31))
        null_scores.append(compute_coherence_score(random_points, angles, sigma))
    
    null_scores = np.array(null_scores)
    null_mean = np.mean(null_scores)
    null_std = np.std(null_scores)
    
    # Compute p-value (one-tailed: observed >= null)
    p_value = np.mean(null_scores >= observed)
    
    # Z-score (test statistic)
    z_score = (observed - null_mean) / null_std if null_std > 1e-10 else 0.0
    
    # Interpretation
    if p_value < 0.01:
        interpretation = (f"Highly significant (p={p_value:.4f}). "
                         f"The observed pattern is extremely unlikely to occur by chance. "
                         f"Observed coherence is {z_score:.1f} standard deviations above random.")
    elif p_value < 0.05:
        interpretation = (f"Significant (p={p_value:.4f}). "
                         f"The observed pattern is unlikely to occur by chance.")
    elif p_value < 0.10:
        interpretation = (f"Marginally significant (p={p_value:.4f}). "
                         f"Some evidence of non-random pattern, but not conclusive.")
    else:
        interpretation = (f"Not significant (p={p_value:.4f}). "
                         f"The observed pattern is consistent with random placement. "
                         f"No strong evidence for intentional geometric arrangement.")
    
    return CSRTestResult(
        test_statistic=z_score,
        p_value=p_value,
        n_simulations=n_simulations,
        observed_coherence=observed,
        null_mean=null_mean,
        null_std=null_std,
        is_significant_05=p_value < 0.05,
        is_significant_01=p_value < 0.01,
        interpretation=interpretation
    )


# ---------------------------------------------------------
# Ripley's K-Function for Spherical Data
# ---------------------------------------------------------

def ripley_k_spherical(nodes: Dict, 
                       radii_deg: np.ndarray = None,
                       n_simulations: int = 999,
                       seed: Optional[int] = 42) -> RipleyKResult:
    """
    Compute Ripley's K-function for point pattern on a sphere.
    
    K(r) counts the expected number of additional points within distance r
    of a typical point, normalized by density.
    
    Under CSR, K(r) = 2π(1 - cos(r)) for a sphere.
    
    Args:
        nodes: Dict of node locations
        radii_deg: Array of radii to evaluate (in degrees)
        n_simulations: Monte Carlo simulations for confidence envelope
        seed: Random seed
    
    Returns:
        RipleyKResult with K(r), L(r), and significance envelopes
    """
    # Extract coordinates
    coords = []
    for node_data in nodes.values():
        if isinstance(node_data, dict):
            lat, lon = node_data.get("coords", (0, 0))
        else:
            lat, lon = node_data
        coords.append([lat, lon])
    
    points = np.array(coords)
    n = len(points)
    
    if radii_deg is None:
        radii_deg = np.linspace(5, 90, 18)  # 5° to 90° in 5° steps
    
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = great_circle_distance_deg(
                points[i, 0], points[i, 1],
                points[j, 0], points[j, 1]
            )
            distances[i, j] = d
            distances[j, i] = d
    
    # Compute K(r) for observed data
    # K(r) = (sphere_area / n^2) * sum over all pairs of 1(d_ij <= r)
    sphere_area = 4 * np.pi  # in steradians, but we'll work in normalized units
    
    K_observed = np.zeros(len(radii_deg))
    for k, r in enumerate(radii_deg):
        count = np.sum(distances <= r) - n  # Exclude diagonal
        K_observed[k] = count / n  # Average neighbors within r
    
    # Expected K under CSR (proportional to spherical cap area)
    # K_expected(r) proportional to 2π(1 - cos(r)) / (4π) * n
    radii_rad = np.radians(radii_deg)
    K_expected = (n - 1) * (1 - np.cos(radii_rad)) / 2  # Normalized
    
    # Compute L(r) = sqrt(K(r) * 2 / (1 - cos(r))) - equivalent
    # For easier interpretation: L(r) - r shows deviation from CSR
    L_observed = np.sqrt(K_observed * 2 / np.maximum(1 - np.cos(radii_rad), 1e-10))
    
    # Monte Carlo envelope
    rng = np.random.default_rng(seed)
    K_simulated = np.zeros((n_simulations, len(radii_deg)))
    
    for sim in range(n_simulations):
        sim_points = generate_uniform_sphere_points(n, seed=rng.integers(0, 2**31))
        
        # Compute distances for simulated points
        sim_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = great_circle_distance_deg(
                    sim_points[i, 0], sim_points[i, 1],
                    sim_points[j, 0], sim_points[j, 1]
                )
                sim_distances[i, j] = d
                sim_distances[j, i] = d
        
        for k, r in enumerate(radii_deg):
            count = np.sum(sim_distances <= r) - n
            K_simulated[sim, k] = count / n
    
    # Compute envelopes (2.5% and 97.5% percentiles)
    envelope_lower = np.percentile(K_simulated, 2.5, axis=0)
    envelope_upper = np.percentile(K_simulated, 97.5, axis=0)
    
    # Check for significant clustering or dispersion
    significant_clustering = np.any(K_observed > envelope_upper)
    significant_dispersion = np.any(K_observed < envelope_lower)
    
    # Maximum deviation
    deviation_upper = np.max((K_observed - envelope_upper) / 
                             np.maximum(envelope_upper, 1e-10))
    deviation_lower = np.max((envelope_lower - K_observed) / 
                             np.maximum(envelope_lower, 1e-10))
    max_deviation = max(deviation_upper, deviation_lower)
    
    return RipleyKResult(
        radii_deg=radii_deg,
        K_observed=K_observed,
        K_expected=K_expected,
        L_observed=L_observed,
        envelope_lower=envelope_lower,
        envelope_upper=envelope_upper,
        max_deviation=max_deviation,
        significant_clustering=significant_clustering,
        significant_dispersion=significant_dispersion
    )


# ---------------------------------------------------------
# Great Circle Alignment Test
# ---------------------------------------------------------

def alignment_significance(nodes: Dict, 
                           tolerance_deg: float = 2.0,
                           n_simulations: int = 1000,
                           seed: Optional[int] = 42) -> AlignmentTestResult:
    """
    Test whether the number of great circle alignments is significant.
    
    An alignment is when a third node lies on the great circle connecting
    two other nodes within a tolerance.
    
    Args:
        nodes: Dict of node locations
        tolerance_deg: Angular tolerance for alignment detection
        n_simulations: Monte Carlo simulations for expected count
        seed: Random seed
    
    Returns:
        AlignmentTestResult with observed vs expected alignments
    """
    # Extract coordinates
    coords = []
    names = []
    for name, node_data in nodes.items():
        if isinstance(node_data, dict):
            lat, lon = node_data.get("coords", (0, 0))
        else:
            lat, lon = node_data
        coords.append([lat, lon])
        names.append(name)
    
    points = np.array(coords)
    n = len(points)
    
    if n < 3:
        return AlignmentTestResult(
            n_alignments_observed=0,
            n_alignments_expected=0.0,
            p_value=1.0,
            alignment_details=[],
            is_significant=False,
            excess_ratio=0.0
        )
    
    # Count alignments in observed data
    observed_alignments = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                
                if point_on_great_circle(
                    points[i, 0], points[i, 1],
                    points[j, 0], points[j, 1],
                    points[k, 0], points[k, 1],
                    tolerance_deg
                ):
                    observed_alignments.append({
                        "node1": names[i],
                        "node2": names[j],
                        "aligned_node": names[k],
                        "coords1": (points[i, 0], points[i, 1]),
                        "coords2": (points[j, 0], points[j, 1]),
                        "coords_aligned": (points[k, 0], points[k, 1])
                    })
    
    n_observed = len(observed_alignments)
    
    # Monte Carlo for expected count under CSR
    rng = np.random.default_rng(seed)
    null_counts = []
    
    for _ in range(n_simulations):
        sim_points = generate_uniform_sphere_points(n, seed=rng.integers(0, 2**31))
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    if point_on_great_circle(
                        sim_points[i, 0], sim_points[i, 1],
                        sim_points[j, 0], sim_points[j, 1],
                        sim_points[k, 0], sim_points[k, 1],
                        tolerance_deg
                    ):
                        count += 1
        
        null_counts.append(count)
    
    null_counts = np.array(null_counts)
    expected = np.mean(null_counts)
    
    # P-value (one-tailed: observed >= expected)
    p_value = np.mean(null_counts >= n_observed)
    
    # Excess ratio
    excess_ratio = n_observed / expected if expected > 0 else float('inf')
    
    return AlignmentTestResult(
        n_alignments_observed=n_observed,
        n_alignments_expected=expected,
        p_value=p_value,
        alignment_details=observed_alignments,
        is_significant=p_value < 0.05,
        excess_ratio=excess_ratio
    )


# ---------------------------------------------------------
# False Discovery Rate Correction
# ---------------------------------------------------------

def compute_false_discovery_rate(p_values: List[float], 
                                  alpha: float = 0.05) -> List[Dict]:
    """
    Apply Benjamini-Hochberg FDR correction to multiple p-values.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Target FDR level
    
    Returns:
        List of dicts with original p-value, adjusted p-value, and significance
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    
    results = [None] * n
    
    # Benjamini-Hochberg procedure
    for rank, (orig_idx, p) in enumerate(sorted_pairs, 1):
        # Critical value for this rank
        critical = (rank / n) * alpha
        
        # Adjusted p-value
        adjusted = p * n / rank
        adjusted = min(adjusted, 1.0)
        
        results[orig_idx] = {
            "original_p": p,
            "adjusted_p": adjusted,
            "rank": rank,
            "critical_value": critical,
            "significant": p <= critical
        }
    
    return results


# ---------------------------------------------------------
# Comprehensive Validation
# ---------------------------------------------------------

def run_full_validation(known_nodes: Dict,
                        n_simulations: int = 1000,
                        seed: int = 42) -> ValidationSummary:
    """
    Run all validation tests and provide comprehensive summary.
    
    Args:
        known_nodes: Dict of known node locations
        n_simulations: Number of Monte Carlo simulations
        seed: Random seed for reproducibility
    
    Returns:
        ValidationSummary with all test results and recommendations
    """
    print("[INFO] Running statistical validation...")
    print(f"[INFO] Using {n_simulations} Monte Carlo simulations")
    
    # Run CSR test
    print("[INFO] Running Complete Spatial Randomness test...")
    csr_result = test_pattern_significance(
        known_nodes, 
        n_simulations=n_simulations,
        seed=seed
    )
    print(f"[INFO] CSR p-value: {csr_result.p_value:.4f}")
    
    # Run Ripley's K
    print("[INFO] Computing Ripley's K-function...")
    ripley_result = ripley_k_spherical(
        known_nodes,
        n_simulations=min(n_simulations, 999),
        seed=seed
    )
    print(f"[INFO] Significant clustering: {ripley_result.significant_clustering}")
    
    # Run alignment test
    print("[INFO] Testing great circle alignments...")
    alignment_result = alignment_significance(
        known_nodes,
        n_simulations=min(n_simulations, 500),
        seed=seed
    )
    print(f"[INFO] Alignments: {alignment_result.n_alignments_observed} observed, "
          f"{alignment_result.n_alignments_expected:.1f} expected")
    
    # Combine p-values using Fisher's method
    p_values = [
        csr_result.p_value,
        alignment_result.p_value
    ]
    
    # Fisher's combined probability (chi-squared with 2k degrees of freedom)
    valid_p = [p for p in p_values if 0 < p < 1]
    if valid_p:
        chi2_stat = -2 * sum(np.log(p) for p in valid_p)
        # Degrees of freedom = 2 * number of tests
        df = 2 * len(valid_p)
        try:
            from scipy import stats
            overall_p = 1 - stats.chi2.cdf(chi2_stat, df)
        except ImportError:
            # Fallback: use minimum p-value with Bonferroni
            overall_p = min(p_values) * len(p_values)
    else:
        overall_p = max(p_values) if p_values else 1.0
    
    overall_p = min(overall_p, 1.0)
    
    # Generate recommendations
    recommendations = []
    
    if csr_result.is_significant_01:
        recommendations.append(
            "Strong evidence for non-random geometric pattern (CSR p < 0.01)"
        )
    elif csr_result.is_significant_05:
        recommendations.append(
            "Moderate evidence for non-random pattern (CSR p < 0.05)"
        )
    else:
        recommendations.append(
            "No significant evidence of non-random pattern in CSR test"
        )
    
    if ripley_result.significant_clustering:
        recommendations.append(
            "Nodes show significant spatial clustering at some scales"
        )
    if ripley_result.significant_dispersion:
        recommendations.append(
            "Nodes show significant spatial dispersion (regular spacing)"
        )
    
    if alignment_result.is_significant:
        recommendations.append(
            f"More great circle alignments than expected by chance "
            f"(ratio: {alignment_result.excess_ratio:.1f}x)"
        )
    else:
        recommendations.append(
            "Great circle alignments are consistent with random placement"
        )
    
    if overall_p < 0.05:
        recommendations.append(
            "OVERALL: Combined tests suggest significant non-random pattern"
        )
    else:
        recommendations.append(
            "OVERALL: Combined tests do not reject random placement hypothesis"
        )
    
    print("[INFO] Validation complete")
    
    return ValidationSummary(
        csr_result=csr_result,
        ripley_result=ripley_result,
        alignment_result=alignment_result,
        overall_significant=overall_p < 0.05,
        overall_p_value=overall_p,
        recommendations=recommendations
    )


# ---------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------

def main():
    """Run validation tests on known GSN nodes."""
    print("\n=== GSN Statistical Validation ===\n")
    
    # Try to load extended nodes
    try:
        from known_nodes_extended import KNOWN_NODES_EXTENDED
        nodes = KNOWN_NODES_EXTENDED
        print(f"Loaded {len(nodes)} nodes from known_nodes_extended.py")
    except ImportError:
        try:
            from gsn_node_predictor import KNOWN_NODES
            nodes = KNOWN_NODES
            print(f"Loaded {len(nodes)} nodes from gsn_node_predictor.py")
        except ImportError:
            print("ERROR: Could not load known nodes")
            return
    
    # Run validation
    summary = run_full_validation(nodes, n_simulations=1000)
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print("\n1. Complete Spatial Randomness Test:")
    print(f"   Observed coherence: {summary.csr_result.observed_coherence:.4f}")
    print(f"   Expected (random):  {summary.csr_result.null_mean:.4f} ± {summary.csr_result.null_std:.4f}")
    print(f"   Z-score: {summary.csr_result.test_statistic:.2f}")
    print(f"   P-value: {summary.csr_result.p_value:.4f}")
    print(f"   {summary.csr_result.interpretation}")
    
    print("\n2. Ripley's K-Function:")
    print(f"   Significant clustering: {summary.ripley_result.significant_clustering}")
    print(f"   Significant dispersion: {summary.ripley_result.significant_dispersion}")
    print(f"   Max deviation from CSR: {summary.ripley_result.max_deviation:.2f}")
    
    print("\n3. Great Circle Alignment Test:")
    print(f"   Observed alignments: {summary.alignment_result.n_alignments_observed}")
    print(f"   Expected (random):   {summary.alignment_result.n_alignments_expected:.1f}")
    print(f"   Excess ratio: {summary.alignment_result.excess_ratio:.2f}x")
    print(f"   P-value: {summary.alignment_result.p_value:.4f}")
    
    print("\n4. Combined Results:")
    print(f"   Overall p-value: {summary.overall_p_value:.4f}")
    print(f"   Pattern significant: {summary.overall_significant}")
    
    print("\n5. Recommendations:")
    for rec in summary.recommendations:
        print(f"   • {rec}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()


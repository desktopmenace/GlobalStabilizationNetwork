#!/usr/bin/env python
"""
GSN Bayesian Scoring Framework

Replaces ad-hoc F scores with proper Bayesian probability estimates.

Key Features:
1. Prior distributions for model parameters
2. Likelihood functions derived from known nodes
3. Posterior probability P(is_node | features)
4. Log-likelihood ratios for interpretability
5. Leave-one-out likelihood estimation

Author: H
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Prior probability that any random location is a GSN node
# Estimate: ~15,000 nodes on Earth / (510 million km² / 100 km² per node area)
DEFAULT_PRIOR_IS_NODE = 0.0001

# Expected number of total GSN nodes (rough estimate)
ESTIMATED_TOTAL_NODES = 15000


# ---------------------------------------------------------
# Data Classes
# ---------------------------------------------------------

@dataclass
class PriorConfig:
    """Configuration for prior distributions."""
    
    # Prior probability of being a node
    p_node: float = DEFAULT_PRIOR_IS_NODE
    
    # Sigma prior: Gamma(shape, rate) -> mean = shape/rate
    sigma_shape: float = 3.0
    sigma_rate: float = 0.5  # Mean sigma = 6.0
    
    # Distance scale prior: Gamma(shape, rate)
    distance_scale_shape: float = 3.0
    distance_scale_rate: float = 0.1  # Mean = 30.0


@dataclass
class LikelihoodResult:
    """Result of likelihood computation."""
    log_likelihood_node: float
    log_likelihood_not_node: float
    log_likelihood_ratio: float
    
    @property
    def likelihood_ratio(self) -> float:
        """Bayes factor: L(node) / L(not_node)"""
        return math.exp(min(self.log_likelihood_ratio, 700))  # Prevent overflow


@dataclass
class PosteriorResult:
    """Result of posterior computation."""
    probability: float  # P(is_node | features)
    log_odds: float
    credible_interval_95: Tuple[float, float]
    credible_interval_68: Tuple[float, float]
    prior_probability: float
    likelihood_ratio: float
    
    def __str__(self) -> str:
        return (f"P(node|data) = {self.probability:.4f} "
                f"(95% CI: {self.credible_interval_95[0]:.4f}-{self.credible_interval_95[1]:.4f})")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def great_circle_angle(lat1: float, lon1: float, 
                       lat2: float, lon2: float) -> float:
    """Compute great-circle angular distance in degrees."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    cos_angle = (np.sin(phi1) * np.sin(phi2) + 
                 np.cos(phi1) * np.cos(phi2) * np.cos(dlon))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def log_gaussian(x: float, mu: float, sigma: float) -> float:
    """Log of Gaussian PDF."""
    if sigma <= 0:
        return -np.inf
    return -0.5 * ((x - mu) ** 2 / sigma ** 2) - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def log_sum_exp(log_values: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_values)
    if not np.isfinite(max_val):
        return -np.inf
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


# ---------------------------------------------------------
# Sacred Angles (for angle-based likelihood)
# ---------------------------------------------------------

SACRED_ANGLES = [
    19.47,  # Tetrahedral
    23.5,   # Axial tilt
    30.0,   # Dodecagon
    36.0,   # Penrose
    45.0,   # Octagon
    60.0,   # Hexagon
    72.0,   # Pentagon
    90.0,   # Quadrant
    108.0,  # Pentagon interior
    120.0,  # Hexagon interior
    137.5,  # Golden angle
    144.0,  # Penrose
    150.0,  # Dodecagon
    180.0,  # Antipodal
]


# ---------------------------------------------------------
# Bayesian Scorer Class
# ---------------------------------------------------------

class BayesianScorer:
    """
    Bayesian framework for GSN node scoring.
    
    Replaces arbitrary F = αG + βH with proper posterior probability.
    """
    
    def __init__(self, 
                 known_nodes: Dict = None,
                 prior_config: PriorConfig = None,
                 angle_sigma: float = 6.0):
        """
        Initialize Bayesian scorer.
        
        Args:
            known_nodes: Dict of known node locations
            prior_config: Prior distribution configuration
            angle_sigma: Gaussian width for angle matching
        """
        self.prior_config = prior_config or PriorConfig()
        self.angle_sigma = angle_sigma
        
        # Load known nodes
        if known_nodes is None:
            try:
                from known_nodes_extended import KNOWN_NODES_EXTENDED
                self.known_nodes = KNOWN_NODES_EXTENDED
            except ImportError:
                try:
                    from gsn_node_predictor import KNOWN_NODES
                    self.known_nodes = KNOWN_NODES
                except ImportError:
                    self.known_nodes = {}
        else:
            self.known_nodes = known_nodes
        
        # Extract coordinates
        self.node_coords = []
        for name, data in self.known_nodes.items():
            if isinstance(data, dict):
                lat, lon = data.get("coords", (0, 0))
            else:
                lat, lon = data
            self.node_coords.append((name, lat, lon))
        
        # Fit likelihood parameters from known nodes
        self._fit_likelihoods()
    
    def _fit_likelihoods(self):
        """
        Fit likelihood distributions from known node data.
        
        Uses leave-one-out to estimate P(features | is_node).
        """
        if len(self.node_coords) < 3:
            # Not enough data - use defaults
            self.angle_dist_node = {"angles": SACRED_ANGLES, "sigma": 6.0}
            self.angle_dist_not_node = {"uniform": True, "range": (0, 180)}
            return
        
        # Compute pairwise angles between known nodes
        node_angles = []
        for i, (name1, lat1, lon1) in enumerate(self.node_coords):
            for j, (name2, lat2, lon2) in enumerate(self.node_coords):
                if i >= j:
                    continue
                angle = great_circle_angle(lat1, lon1, lat2, lon2)
                node_angles.append(angle)
        
        self.node_angles = np.array(node_angles)
        
        # Fit angle distribution for nodes
        # Mixture of Gaussians centered on sacred angles
        self.angle_dist_node = {
            "angles": SACRED_ANGLES,
            "sigma": self.angle_sigma,
            "weights": self._estimate_angle_weights(self.node_angles)
        }
        
        # Non-node distribution: uniform on [0, 180]
        self.angle_dist_not_node = {"uniform": True, "range": (0, 180)}
    
    def _estimate_angle_weights(self, observed_angles: np.ndarray) -> np.ndarray:
        """Estimate mixture weights for sacred angles."""
        n_angles = len(SACRED_ANGLES)
        counts = np.zeros(n_angles)
        
        for angle in observed_angles:
            # Find nearest sacred angle
            diffs = np.abs(np.array(SACRED_ANGLES) - angle)
            nearest = np.argmin(diffs)
            if diffs[nearest] < 2 * self.angle_sigma:
                counts[nearest] += 1
        
        # Add pseudocount for smoothing
        counts += 0.5
        weights = counts / counts.sum()
        
        return weights
    
    def _log_likelihood_angle_node(self, angle: float) -> float:
        """Log-likelihood of angle given location IS a node."""
        # Mixture of Gaussians at sacred angles
        weights = self.angle_dist_node.get("weights", 
                                           np.ones(len(SACRED_ANGLES)) / len(SACRED_ANGLES))
        sigma = self.angle_dist_node["sigma"]
        
        log_probs = []
        for i, sacred_angle in enumerate(SACRED_ANGLES):
            log_weight = np.log(weights[i] + 1e-10)
            log_gaussian_val = log_gaussian(angle, sacred_angle, sigma)
            log_probs.append(log_weight + log_gaussian_val)
        
        return log_sum_exp(np.array(log_probs))
    
    def _log_likelihood_angle_not_node(self, angle: float) -> float:
        """Log-likelihood of angle given location is NOT a node."""
        # Uniform distribution on [0, 180]
        if 0 <= angle <= 180:
            return -np.log(180)  # log(1/180)
        return -np.inf
    
    def compute_likelihood(self, lat: float, lon: float) -> LikelihoodResult:
        """
        Compute likelihood ratio for a location.
        
        Uses angles to all known nodes as features.
        
        Args:
            lat, lon: Query location
            
        Returns:
            LikelihoodResult with log-likelihoods and ratio
        """
        if not self.node_coords:
            return LikelihoodResult(
                log_likelihood_node=0.0,
                log_likelihood_not_node=0.0,
                log_likelihood_ratio=0.0
            )
        
        log_L_node = 0.0
        log_L_not_node = 0.0
        
        for name, nlat, nlon in self.node_coords:
            angle = great_circle_angle(lat, lon, nlat, nlon)
            
            log_L_node += self._log_likelihood_angle_node(angle)
            log_L_not_node += self._log_likelihood_angle_not_node(angle)
        
        log_ratio = log_L_node - log_L_not_node
        
        return LikelihoodResult(
            log_likelihood_node=log_L_node,
            log_likelihood_not_node=log_L_not_node,
            log_likelihood_ratio=log_ratio
        )
    
    def compute_posterior(self, lat: float, lon: float,
                          prior: float = None) -> PosteriorResult:
        """
        Compute posterior probability P(is_node | features).
        
        Uses Bayes' theorem:
        P(node|data) = P(data|node) * P(node) / P(data)
        
        Args:
            lat, lon: Query location
            prior: Prior probability (default from config)
            
        Returns:
            PosteriorResult with probability and credible intervals
        """
        if prior is None:
            prior = self.prior_config.p_node
        
        likelihood_result = self.compute_likelihood(lat, lon)
        
        # Log-odds form of Bayes' theorem
        # log_odds_posterior = log_odds_prior + log_likelihood_ratio
        prior_odds = prior / (1 - prior) if prior < 1 else 1e10
        log_prior_odds = np.log(prior_odds + 1e-300)
        
        log_posterior_odds = log_prior_odds + likelihood_result.log_likelihood_ratio
        
        # Convert back to probability
        # P = odds / (1 + odds) = 1 / (1 + exp(-log_odds))
        if log_posterior_odds > 700:
            posterior_prob = 1.0
        elif log_posterior_odds < -700:
            posterior_prob = 0.0
        else:
            posterior_prob = 1.0 / (1.0 + np.exp(-log_posterior_odds))
        
        # Compute credible intervals via uncertainty propagation
        # Approximate using likelihood ratio uncertainty
        ci_95 = self._compute_credible_interval(posterior_prob, 0.95)
        ci_68 = self._compute_credible_interval(posterior_prob, 0.68)
        
        return PosteriorResult(
            probability=posterior_prob,
            log_odds=log_posterior_odds,
            credible_interval_95=ci_95,
            credible_interval_68=ci_68,
            prior_probability=prior,
            likelihood_ratio=likelihood_result.likelihood_ratio
        )
    
    def _compute_credible_interval(self, prob: float, 
                                    level: float) -> Tuple[float, float]:
        """
        Compute approximate credible interval.
        
        Uses beta distribution approximation based on effective sample size.
        """
        # Effective sample size from likelihood
        n_eff = len(self.node_coords)
        
        if n_eff < 2:
            # Wide uncertainty with little data
            half_width = (1 - level) / 2 + 0.2
            return (max(0, prob - half_width), min(1, prob + half_width))
        
        # Beta distribution parameters
        # Match mean to posterior probability
        alpha = prob * n_eff + 1
        beta = (1 - prob) * n_eff + 1
        
        try:
            from scipy import stats
            lower = stats.beta.ppf((1 - level) / 2, alpha, beta)
            upper = stats.beta.ppf(1 - (1 - level) / 2, alpha, beta)
        except ImportError:
            # Approximate with normal
            std = np.sqrt(prob * (1 - prob) / n_eff)
            z = 1.96 if level >= 0.95 else 1.0
            lower = max(0, prob - z * std)
            upper = min(1, prob + z * std)
        
        return (float(lower), float(upper))
    
    def leave_one_out_validation(self) -> Dict:
        """
        Perform leave-one-out cross-validation.
        
        For each known node, compute posterior using all other nodes.
        Evaluates how well the model predicts known nodes.
        
        Returns:
            Dict with validation metrics
        """
        if len(self.node_coords) < 3:
            return {"error": "Insufficient nodes for LOOCV"}
        
        predictions = []
        
        for i, (name, lat, lon) in enumerate(self.node_coords):
            # Create scorer without this node
            remaining_nodes = {
                n: self.known_nodes[n] 
                for n in self.known_nodes if n != name
            }
            
            loo_scorer = BayesianScorer(
                known_nodes=remaining_nodes,
                prior_config=self.prior_config,
                angle_sigma=self.angle_sigma
            )
            
            posterior = loo_scorer.compute_posterior(lat, lon)
            
            predictions.append({
                "node": name,
                "lat": lat,
                "lon": lon,
                "posterior_probability": posterior.probability,
                "log_odds": posterior.log_odds,
                "likelihood_ratio": posterior.likelihood_ratio
            })
        
        # Compute metrics
        probs = np.array([p["posterior_probability"] for p in predictions])
        
        # For true positives, we want high probabilities
        mean_prob = np.mean(probs)
        min_prob = np.min(probs)
        max_prob = np.max(probs)
        
        # AUC would require negative examples - we just check that
        # known nodes get higher scores
        
        return {
            "n_nodes": len(predictions),
            "mean_posterior": float(mean_prob),
            "min_posterior": float(min_prob),
            "max_posterior": float(max_prob),
            "std_posterior": float(np.std(probs)),
            "predictions": predictions,
            "interpretation": self._interpret_loocv(probs)
        }
    
    def _interpret_loocv(self, probs: np.ndarray) -> str:
        """Interpret LOOCV results."""
        mean_p = np.mean(probs)
        min_p = np.min(probs)
        
        if mean_p > 0.5 and min_p > 0.1:
            return ("Good: Known nodes receive high posterior probabilities. "
                   "Model captures node characteristics well.")
        elif mean_p > 0.2:
            return ("Moderate: Known nodes receive elevated but not high probabilities. "
                   "Model partially captures node characteristics.")
        else:
            return ("Poor: Known nodes don't receive notably higher probabilities. "
                   "Model may not capture relevant features or data is insufficient.")
    
    def score_grid(self, lats: np.ndarray, lons: np.ndarray,
                   prior: float = None) -> np.ndarray:
        """
        Compute posterior probabilities for a lat/lon grid.
        
        Args:
            lats: 1D array of latitudes
            lons: 1D array of longitudes
            prior: Prior probability
            
        Returns:
            2D array (nlat, nlon) of posterior probabilities
        """
        nlat, nlon = len(lats), len(lons)
        posterior_grid = np.zeros((nlat, nlon))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                result = self.compute_posterior(lat, lon, prior)
                posterior_grid[i, j] = result.probability
        
        return posterior_grid


# ---------------------------------------------------------
# Geophysical Bayesian Component
# ---------------------------------------------------------

class GeophysicalBayesianScorer:
    """
    Bayesian scorer for geophysical features (G component).
    
    Models P(geophysical_features | is_node) using empirical distributions
    from known node locations.
    """
    
    def __init__(self, known_nodes: Dict = None):
        """Initialize with known node locations."""
        if known_nodes is None:
            try:
                from known_nodes_extended import KNOWN_NODES_EXTENDED
                known_nodes = KNOWN_NODES_EXTENDED
            except ImportError:
                known_nodes = {}
        
        self.known_nodes = known_nodes
        self._fit_distributions()
    
    def _fit_distributions(self):
        """Fit feature distributions from known nodes."""
        try:
            from gsn_node_predictor import get_geophysical_inputs
        except ImportError:
            self.fitted = False
            return
        
        # Collect features from known nodes
        ga_values = []
        ct_values = []
        dist_values = []
        
        for name, data in self.known_nodes.items():
            if isinstance(data, dict):
                lat, lon = data.get("coords", (0, 0))
            else:
                lat, lon = data
            
            try:
                ga, ct, dist = get_geophysical_inputs(lat, lon)
                if np.isfinite(ga):
                    ga_values.append(ga)
                if np.isfinite(ct):
                    ct_values.append(ct)
                if np.isfinite(dist):
                    dist_values.append(dist)
            except Exception:
                continue
        
        if len(ga_values) < 5:
            self.fitted = False
            return
        
        # Fit Gaussian distributions
        self.ga_dist = {
            "mean": np.mean(ga_values),
            "std": max(np.std(ga_values), 5.0)  # Minimum std
        }
        
        self.ct_dist = {
            "mean": np.mean(ct_values),
            "std": max(np.std(ct_values), 3.0)
        }
        
        # Log-normal for distance (always positive)
        log_dist = np.log(np.array(dist_values) + 1)
        self.dist_dist = {
            "log_mean": np.mean(log_dist),
            "log_std": max(np.std(log_dist), 0.5)
        }
        
        self.fitted = True
    
    def compute_log_likelihood(self, ga: float, ct: float, 
                                dist_km: float) -> Tuple[float, float]:
        """
        Compute log-likelihoods for geophysical features.
        
        Returns:
            (log_L_node, log_L_not_node)
        """
        if not self.fitted:
            return (0.0, 0.0)
        
        # P(features | node) - use fitted distributions
        log_L_node = 0.0
        
        if np.isfinite(ga):
            log_L_node += log_gaussian(ga, self.ga_dist["mean"], self.ga_dist["std"])
        
        if np.isfinite(ct):
            log_L_node += log_gaussian(ct, self.ct_dist["mean"], self.ct_dist["std"])
        
        if np.isfinite(dist_km) and dist_km > 0:
            log_dist = np.log(dist_km + 1)
            log_L_node += log_gaussian(log_dist, 
                                       self.dist_dist["log_mean"],
                                       self.dist_dist["log_std"])
        
        # P(features | not_node) - use broader distributions
        # Assume non-nodes have more variable features
        log_L_not_node = 0.0
        
        # Use global distributions (wider)
        if np.isfinite(ga):
            log_L_not_node += log_gaussian(ga, 0, 100)  # Wide prior
        
        if np.isfinite(ct):
            log_L_not_node += log_gaussian(ct, 35, 15)  # Continental mean
        
        if np.isfinite(dist_km) and dist_km > 0:
            # Uniform on log scale
            log_L_not_node += -np.log(10)  # Rough approximation
        
        return (log_L_node, log_L_not_node)


# ---------------------------------------------------------
# Combined Bayesian Scorer
# ---------------------------------------------------------

class CombinedBayesianScorer:
    """
    Combined Bayesian scorer using both geometric and geophysical features.
    """
    
    def __init__(self, known_nodes: Dict = None, prior: float = None):
        """Initialize combined scorer."""
        self.geometric_scorer = BayesianScorer(known_nodes)
        self.geophysical_scorer = GeophysicalBayesianScorer(known_nodes)
        self.prior = prior or DEFAULT_PRIOR_IS_NODE
    
    def compute_posterior(self, lat: float, lon: float,
                          ga: float = None, ct: float = None,
                          dist_km: float = None) -> PosteriorResult:
        """
        Compute combined posterior probability.
        
        Combines geometric and geophysical likelihoods.
        """
        # Get geophysical inputs if not provided
        if ga is None or ct is None or dist_km is None:
            try:
                from gsn_node_predictor import get_geophysical_inputs
                ga, ct, dist_km = get_geophysical_inputs(lat, lon)
            except Exception:
                ga, ct, dist_km = 0, 35, 500
        
        # Geometric likelihood
        geo_result = self.geometric_scorer.compute_likelihood(lat, lon)
        
        # Geophysical likelihood
        log_L_geo_node, log_L_geo_not = self.geophysical_scorer.compute_log_likelihood(
            ga, ct, dist_km
        )
        
        # Combined log-likelihood ratio
        log_lr_combined = (geo_result.log_likelihood_ratio + 
                          (log_L_geo_node - log_L_geo_not))
        
        # Posterior from combined likelihood
        prior_odds = self.prior / (1 - self.prior)
        log_prior_odds = np.log(prior_odds + 1e-300)
        log_posterior_odds = log_prior_odds + log_lr_combined
        
        if log_posterior_odds > 700:
            posterior_prob = 1.0
        elif log_posterior_odds < -700:
            posterior_prob = 0.0
        else:
            posterior_prob = 1.0 / (1.0 + np.exp(-log_posterior_odds))
        
        # Credible intervals (approximate)
        n_eff = len(self.geometric_scorer.node_coords) + 3
        std = np.sqrt(posterior_prob * (1 - posterior_prob) / n_eff)
        
        return PosteriorResult(
            probability=posterior_prob,
            log_odds=log_posterior_odds,
            credible_interval_95=(max(0, posterior_prob - 2*std), 
                                  min(1, posterior_prob + 2*std)),
            credible_interval_68=(max(0, posterior_prob - std),
                                  min(1, posterior_prob + std)),
            prior_probability=self.prior,
            likelihood_ratio=np.exp(min(log_lr_combined, 700))
        )


# ---------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------

def main():
    """Demonstrate Bayesian scoring."""
    print("\n=== GSN Bayesian Scorer Demo ===\n")
    
    # Initialize scorer
    scorer = BayesianScorer()
    print(f"Loaded {len(scorer.node_coords)} known nodes")
    
    # Test on a known node (Giza)
    print("\n--- Testing Known Node (Giza) ---")
    result = scorer.compute_posterior(29.9792, 31.1342)
    print(f"Posterior: {result}")
    print(f"Likelihood ratio: {result.likelihood_ratio:.2f}")
    
    # Test on a random location
    print("\n--- Testing Random Location ---")
    result = scorer.compute_posterior(45.0, -93.0)  # Minneapolis
    print(f"Posterior: {result}")
    print(f"Likelihood ratio: {result.likelihood_ratio:.4f}")
    
    # Leave-one-out validation
    print("\n--- Leave-One-Out Validation ---")
    loocv = scorer.leave_one_out_validation()
    print(f"Mean posterior for known nodes: {loocv['mean_posterior']:.4f}")
    print(f"Min posterior: {loocv['min_posterior']:.4f}")
    print(f"Interpretation: {loocv['interpretation']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()


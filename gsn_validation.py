#!/usr/bin/env python
"""
GSN Validation Framework

Provides metrics and cross-validation tools to evaluate the accuracy
of GSN node predictions against known reference sites.

Key features:
- Recall@K metrics: How many known nodes are captured in top-K predictions
- Leave-one-out cross-validation: Test generalization on held-out nodes
- Distance-based matching: Match predictions within threshold distance
- Statistical significance testing
"""

import math
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from known_nodes_extended import KNOWN_NODES_EXTENDED, get_coords_dict


# ---------------------------------------------------------
# Distance Functions
# ---------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in kilometers.
    """
    R = 6371.0  # Earth radius in km
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    c = 2 * math.asin(math.sqrt(min(1.0, a)))
    
    return R * c


def find_nearest_candidate(
    lat: float, 
    lon: float, 
    candidates: List[Dict]
) -> Tuple[Optional[Dict], float]:
    """
    Find the nearest candidate to a given point.
    
    Returns:
        (nearest_candidate, distance_km)
    """
    min_dist = float("inf")
    nearest = None
    
    for cand in candidates:
        d = haversine_km(lat, lon, cand["lat"], cand["lon"])
        if d < min_dist:
            min_dist = d
            nearest = cand
    
    return nearest, min_dist


# ---------------------------------------------------------
# Core Validation Metrics
# ---------------------------------------------------------

def compute_recall_at_k(
    candidates: List[Dict],
    known_nodes: Dict[str, Tuple[float, float]],
    threshold_km: float = 100.0,
    k: int = None,
) -> Dict[str, any]:
    """
    Compute recall@K: fraction of known nodes matched by top-K predictions.
    
    A known node is "matched" if any candidate within top-K is within
    threshold_km distance of it.
    
    Args:
        candidates: List of candidate dicts with 'lat', 'lon', 'F' keys
        known_nodes: Dict of {name: (lat, lon)} for ground truth
        threshold_km: Distance threshold for a match
        k: Number of top candidates to consider (None = all)
    
    Returns:
        Dict with recall, hits, misses, and per-node details
    """
    if k is not None:
        candidates = candidates[:k]
    
    hits = []
    misses = []
    details = {}
    
    for name, coords in known_nodes.items():
        if isinstance(coords, dict):
            lat, lon = coords["coords"]
        else:
            lat, lon = coords
        
        nearest, dist = find_nearest_candidate(lat, lon, candidates)
        
        matched = dist <= threshold_km
        details[name] = {
            "lat": lat,
            "lon": lon,
            "nearest_dist_km": dist,
            "matched": matched,
            "nearest_candidate": nearest,
        }
        
        if matched:
            hits.append(name)
        else:
            misses.append(name)
    
    n_nodes = len(known_nodes)
    recall = len(hits) / n_nodes if n_nodes > 0 else 0.0
    
    return {
        "recall": recall,
        "n_hits": len(hits),
        "n_misses": len(misses),
        "n_nodes": n_nodes,
        "n_candidates": len(candidates),
        "threshold_km": threshold_km,
        "hits": hits,
        "misses": misses,
        "details": details,
    }


def compute_precision_at_k(
    candidates: List[Dict],
    known_nodes: Dict[str, Tuple[float, float]],
    threshold_km: float = 100.0,
    k: int = None,
) -> Dict[str, any]:
    """
    Compute precision@K: fraction of top-K candidates that match known nodes.
    
    Args:
        candidates: List of candidate dicts
        known_nodes: Ground truth nodes
        threshold_km: Distance threshold for a match
        k: Number of top candidates
    
    Returns:
        Dict with precision and details
    """
    if k is not None:
        candidates = candidates[:k]
    
    # For each candidate, check if it matches any known node
    matched_candidates = []
    unmatched_candidates = []
    
    node_coords = []
    for name, coords in known_nodes.items():
        if isinstance(coords, dict):
            node_coords.append((name, coords["coords"][0], coords["coords"][1]))
        else:
            node_coords.append((name, coords[0], coords[1]))
    
    for cand in candidates:
        is_matched = False
        matched_node = None
        min_dist = float("inf")
        
        for name, nlat, nlon in node_coords:
            d = haversine_km(cand["lat"], cand["lon"], nlat, nlon)
            if d < min_dist:
                min_dist = d
                if d <= threshold_km:
                    is_matched = True
                    matched_node = name
        
        if is_matched:
            matched_candidates.append((cand, matched_node, min_dist))
        else:
            unmatched_candidates.append((cand, min_dist))
    
    n_candidates = len(candidates)
    precision = len(matched_candidates) / n_candidates if n_candidates > 0 else 0.0
    
    return {
        "precision": precision,
        "n_matched": len(matched_candidates),
        "n_unmatched": len(unmatched_candidates),
        "n_candidates": n_candidates,
        "threshold_km": threshold_km,
    }


def compute_f1_score(recall: float, precision: float) -> float:
    """Compute F1 score from recall and precision."""
    if recall + precision == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ---------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------

def leave_one_out_cv(
    known_nodes: Dict,
    predict_func: Callable,
    threshold_km: float = 100.0,
    top_k: int = 50,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Leave-one-out cross-validation for GSN predictions.
    
    For each known node:
    1. Remove it from the training set
    2. Run prediction using remaining nodes
    3. Check if the held-out node is recovered in top-K
    
    Args:
        known_nodes: Dict of known nodes
        predict_func: Function(training_nodes) -> candidates list
        threshold_km: Match threshold
        top_k: Number of candidates to consider
        verbose: Print progress
    
    Returns:
        Dict with mean recall, per-node results
    """
    results = {}
    recalls = []
    
    node_names = list(known_nodes.keys())
    
    for i, held_out_name in enumerate(node_names):
        if verbose:
            print(f"[CV] Fold {i+1}/{len(node_names)}: holding out {held_out_name}")
        
        # Create training set without held-out node
        training_nodes = {
            k: v for k, v in known_nodes.items() 
            if k != held_out_name
        }
        
        # Run prediction
        try:
            candidates = predict_func(training_nodes)
        except Exception as e:
            print(f"[ERROR] Prediction failed for fold {i+1}: {e}")
            results[held_out_name] = {"error": str(e)}
            continue
        
        # Check if held-out node is recovered
        held_out_coords = known_nodes[held_out_name]
        if isinstance(held_out_coords, dict):
            held_out_lat, held_out_lon = held_out_coords["coords"]
        else:
            held_out_lat, held_out_lon = held_out_coords
        
        recall_result = compute_recall_at_k(
            candidates[:top_k],
            {held_out_name: (held_out_lat, held_out_lon)},
            threshold_km=threshold_km,
        )
        
        results[held_out_name] = {
            "recovered": recall_result["recall"] > 0,
            "nearest_dist_km": recall_result["details"][held_out_name]["nearest_dist_km"],
        }
        recalls.append(recall_result["recall"])
    
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "mean_recall": mean_recall,
        "n_recovered": sum(1 for r in results.values() if r.get("recovered", False)),
        "n_total": len(node_names),
        "threshold_km": threshold_km,
        "top_k": top_k,
        "per_node": results,
    }


def k_fold_cv(
    known_nodes: Dict,
    predict_func: Callable,
    k_folds: int = 5,
    threshold_km: float = 100.0,
    top_k: int = 50,
    seed: int = 42,
) -> Dict[str, any]:
    """
    K-fold cross-validation for GSN predictions.
    
    Args:
        known_nodes: Dict of known nodes
        predict_func: Function(training_nodes) -> candidates list
        k_folds: Number of folds
        threshold_km: Match threshold
        top_k: Number of candidates
        seed: Random seed for fold assignment
    
    Returns:
        Dict with fold results and aggregate metrics
    """
    np.random.seed(seed)
    
    node_names = list(known_nodes.keys())
    np.random.shuffle(node_names)
    
    # Split into folds
    folds = [[] for _ in range(k_folds)]
    for i, name in enumerate(node_names):
        folds[i % k_folds].append(name)
    
    fold_results = []
    
    for fold_idx in range(k_folds):
        test_names = folds[fold_idx]
        train_names = [n for n in node_names if n not in test_names]
        
        training_nodes = {k: known_nodes[k] for k in train_names}
        test_nodes = {k: known_nodes[k] for k in test_names}
        
        # Convert test_nodes for recall computation
        test_coords = {}
        for name, data in test_nodes.items():
            if isinstance(data, dict):
                test_coords[name] = data["coords"]
            else:
                test_coords[name] = data
        
        try:
            candidates = predict_func(training_nodes)
            recall_result = compute_recall_at_k(
                candidates[:top_k],
                test_coords,
                threshold_km=threshold_km,
            )
            fold_results.append({
                "fold": fold_idx,
                "recall": recall_result["recall"],
                "n_test": len(test_names),
                "n_train": len(train_names),
            })
        except Exception as e:
            fold_results.append({
                "fold": fold_idx,
                "error": str(e),
            })
    
    valid_recalls = [f["recall"] for f in fold_results if "recall" in f]
    
    return {
        "mean_recall": np.mean(valid_recalls) if valid_recalls else 0.0,
        "std_recall": np.std(valid_recalls) if valid_recalls else 0.0,
        "fold_results": fold_results,
        "k_folds": k_folds,
    }


# ---------------------------------------------------------
# Geographic Stratified Cross-Validation
# ---------------------------------------------------------

# Geographic regions for stratification
GEOGRAPHIC_REGIONS = {
    "americas_north": lambda lat, lon: lat >= 15 and -170 <= lon < -30,
    "americas_central": lambda lat, lon: -15 < lat < 15 and -120 <= lon < -30,
    "americas_south": lambda lat, lon: lat <= -15 and -90 <= lon < -30,
    "europe": lambda lat, lon: lat >= 35 and -15 <= lon < 40,
    "africa": lambda lat, lon: -35 <= lat < 35 and -20 <= lon < 55,
    "middle_east": lambda lat, lon: 10 <= lat < 45 and 25 <= lon < 65,
    "asia_central": lambda lat, lon: 25 <= lat < 55 and 60 <= lon < 100,
    "asia_east": lambda lat, lon: 15 <= lat < 55 and 100 <= lon <= 145,
    "asia_south": lambda lat, lon: -10 <= lat < 35 and 65 <= lon < 100,
    "oceania": lambda lat, lon: lat < 0 and 100 <= lon <= 180,
}


def assign_region(lat: float, lon: float) -> str:
    """Assign a coordinate to a geographic region."""
    for region, check in GEOGRAPHIC_REGIONS.items():
        if check(lat, lon):
            return region
    return "other"


def geographic_stratified_cv(
    known_nodes: Dict,
    predict_func: Callable,
    n_folds: int = 5,
    threshold_km: float = 100.0,
    top_k: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Cross-validation with geographic stratification.
    
    Ensures each fold contains proportional representation from different
    geographic regions, preventing overfitting to spatial clusters.
    
    Args:
        known_nodes: Dict of known nodes
        predict_func: Function(training_nodes) -> candidates list
        n_folds: Number of folds
        threshold_km: Match threshold
        top_k: Number of candidates
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with fold results and aggregate metrics
    """
    np.random.seed(seed)
    
    # Assign each node to a region
    node_regions = {}
    region_nodes = {r: [] for r in list(GEOGRAPHIC_REGIONS.keys()) + ["other"]}
    
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            lat, lon = data["coords"]
        else:
            lat, lon = data
        
        region = assign_region(lat, lon)
        node_regions[name] = region
        region_nodes[region].append(name)
    
    if verbose:
        print("[GeoCV] Region distribution:")
        for region, nodes in region_nodes.items():
            if nodes:
                print(f"  {region}: {len(nodes)} nodes")
    
    # Shuffle nodes within each region
    for region in region_nodes:
        np.random.shuffle(region_nodes[region])
    
    # Create stratified folds
    folds = [[] for _ in range(n_folds)]
    
    for region, nodes in region_nodes.items():
        for i, node_name in enumerate(nodes):
            fold_idx = i % n_folds
            folds[fold_idx].append(node_name)
    
    # Run cross-validation
    fold_results = []
    
    for fold_idx in range(n_folds):
        test_names = folds[fold_idx]
        train_names = [n for n in known_nodes.keys() if n not in test_names]
        
        if verbose:
            print(f"[GeoCV] Fold {fold_idx + 1}/{n_folds}: {len(train_names)} train, {len(test_names)} test")
        
        training_nodes = {k: known_nodes[k] for k in train_names}
        test_nodes = {k: known_nodes[k] for k in test_names}
        
        # Convert test_nodes for recall computation
        test_coords = {}
        for name, data in test_nodes.items():
            if isinstance(data, dict):
                test_coords[name] = data["coords"]
            else:
                test_coords[name] = data
        
        try:
            candidates = predict_func(training_nodes)
            recall_result = compute_recall_at_k(
                candidates[:top_k],
                test_coords,
                threshold_km=threshold_km,
            )
            
            # Track which regions were recovered
            regions_in_fold = set(node_regions[n] for n in test_names)
            
            fold_results.append({
                "fold": fold_idx,
                "recall": recall_result["recall"],
                "n_test": len(test_names),
                "n_train": len(train_names),
                "n_hits": recall_result["n_hits"],
                "regions": list(regions_in_fold),
            })
        except Exception as e:
            fold_results.append({
                "fold": fold_idx,
                "error": str(e),
            })
    
    valid_recalls = [f["recall"] for f in fold_results if "recall" in f]
    
    return {
        "mean_recall": np.mean(valid_recalls) if valid_recalls else 0.0,
        "std_recall": np.std(valid_recalls) if valid_recalls else 0.0,
        "fold_results": fold_results,
        "n_folds": n_folds,
        "region_distribution": {r: len(nodes) for r, nodes in region_nodes.items() if nodes},
    }


def spatial_buffer_cv(
    known_nodes: Dict,
    predict_func: Callable,
    buffer_km: float = 500.0,
    threshold_km: float = 100.0,
    top_k: int = 50,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Leave-one-out CV with spatial buffer to prevent spatial leakage.
    
    When testing on a held-out node, also excludes all nearby nodes
    from training to ensure true generalization.
    
    Args:
        known_nodes: Dict of known nodes
        predict_func: Function(training_nodes) -> candidates list
        buffer_km: Radius to exclude around held-out node
        threshold_km: Match threshold
        top_k: Number of candidates
        verbose: Print progress
    
    Returns:
        Dict with results per node and aggregate metrics
    """
    results = {}
    recalls = []
    excluded_counts = []
    
    node_names = list(known_nodes.keys())
    
    # Pre-compute node coordinates
    node_coords = {}
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            node_coords[name] = data["coords"]
        else:
            node_coords[name] = data
    
    for i, held_out_name in enumerate(node_names):
        held_out_lat, held_out_lon = node_coords[held_out_name]
        
        # Find nodes within buffer distance
        excluded = [held_out_name]
        for other_name, (other_lat, other_lon) in node_coords.items():
            if other_name == held_out_name:
                continue
            dist = haversine_km(held_out_lat, held_out_lon, other_lat, other_lon)
            if dist < buffer_km:
                excluded.append(other_name)
        
        excluded_counts.append(len(excluded))
        
        if verbose:
            print(f"[BufferCV] {i+1}/{len(node_names)}: {held_out_name} "
                  f"(excluding {len(excluded)} nodes within {buffer_km}km)")
        
        # Create training set excluding buffered nodes
        training_nodes = {
            k: v for k, v in known_nodes.items() 
            if k not in excluded
        }
        
        if len(training_nodes) < 3:
            if verbose:
                print(f"  [WARN] Too few training nodes, skipping")
            results[held_out_name] = {"error": "insufficient_training_data"}
            continue
        
        try:
            candidates = predict_func(training_nodes)
            recall_result = compute_recall_at_k(
                candidates[:top_k],
                {held_out_name: node_coords[held_out_name]},
                threshold_km=threshold_km,
            )
            
            results[held_out_name] = {
                "recovered": recall_result["recall"] > 0,
                "nearest_dist_km": recall_result["details"][held_out_name]["nearest_dist_km"],
                "n_excluded": len(excluded),
                "n_training": len(training_nodes),
            }
            recalls.append(recall_result["recall"])
            
        except Exception as e:
            results[held_out_name] = {"error": str(e)}
    
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "mean_recall": mean_recall,
        "n_recovered": sum(1 for r in results.values() if r.get("recovered", False)),
        "n_total": len(node_names),
        "buffer_km": buffer_km,
        "threshold_km": threshold_km,
        "top_k": top_k,
        "avg_excluded": np.mean(excluded_counts) if excluded_counts else 0,
        "per_node": results,
    }


def spatial_block_cv(
    known_nodes: Dict,
    predict_func: Callable,
    n_blocks: int = 5,
    block_size_km: float = 500.0,
    threshold_km: float = 100.0,
    top_k: int = 50,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Cross-validation with spatial blocking using hierarchical clustering.
    
    Groups nearby nodes into blocks, then uses blocks as CV folds.
    Nodes within a block are never split across train/test sets,
    preventing spatial leakage.
    
    Args:
        known_nodes: Dict of known nodes
        predict_func: Function(training_nodes) -> candidates list
        n_blocks: Desired number of blocks (actual may vary)
        block_size_km: Distance threshold for clustering
        threshold_km: Match threshold
        top_k: Number of candidates
        verbose: Print progress
    
    Returns:
        Dict with block info, fold results, and aggregate metrics
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        HAS_CLUSTERING = True
    except ImportError:
        HAS_CLUSTERING = False
    
    # Extract coordinates
    node_names = list(known_nodes.keys())
    coords = []
    for name in node_names:
        data = known_nodes[name]
        if isinstance(data, dict):
            lat, lon = data["coords"]
        else:
            lat, lon = data
        coords.append([lat, lon])
    
    coords = np.array(coords)
    
    # Cluster nodes spatially
    if HAS_CLUSTERING:
        # Convert block_size_km to approximate degrees (rough conversion)
        # 1 degree ≈ 111 km at equator
        distance_threshold = block_size_km / 111.0
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="complete",
        )
        block_labels = clustering.fit_predict(coords)
    else:
        # Fallback: simple grid-based blocking
        lat_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_blocks + 1)
        lon_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_blocks + 1)
        
        block_labels = []
        for lat, lon in coords:
            lat_idx = min(n_blocks - 1, np.searchsorted(lat_bins[1:], lat))
            lon_idx = min(n_blocks - 1, np.searchsorted(lon_bins[1:], lon))
            block_labels.append(lat_idx * n_blocks + lon_idx)
        block_labels = np.array(block_labels)
    
    # Group nodes by block
    unique_blocks = np.unique(block_labels)
    n_actual_blocks = len(unique_blocks)
    
    block_nodes = {b: [] for b in unique_blocks}
    for i, name in enumerate(node_names):
        block_nodes[block_labels[i]].append(name)
    
    if verbose:
        print(f"[SpatialBlockCV] Created {n_actual_blocks} spatial blocks:")
        for block, nodes in block_nodes.items():
            print(f"  Block {block}: {len(nodes)} nodes")
    
    # Use blocks as CV folds (merge small blocks if needed)
    # Assign blocks to folds in round-robin fashion
    block_to_fold = {}
    sorted_blocks = sorted(unique_blocks, key=lambda b: len(block_nodes[b]), reverse=True)
    
    target_folds = min(n_blocks, n_actual_blocks)
    for i, block in enumerate(sorted_blocks):
        block_to_fold[block] = i % target_folds
    
    # Create folds
    folds = [[] for _ in range(target_folds)]
    for block, fold_idx in block_to_fold.items():
        folds[fold_idx].extend(block_nodes[block])
    
    # Run cross-validation
    fold_results = []
    
    for fold_idx in range(target_folds):
        test_names = folds[fold_idx]
        train_names = [n for n in node_names if n not in test_names]
        
        if verbose:
            print(f"[SpatialBlockCV] Fold {fold_idx + 1}/{target_folds}: "
                  f"{len(train_names)} train, {len(test_names)} test")
        
        training_nodes = {k: known_nodes[k] for k in train_names}
        
        # Convert test nodes for recall computation
        test_coords = {}
        for name in test_names:
            data = known_nodes[name]
            if isinstance(data, dict):
                test_coords[name] = data["coords"]
            else:
                test_coords[name] = data
        
        try:
            candidates = predict_func(training_nodes)
            recall_result = compute_recall_at_k(
                candidates[:top_k],
                test_coords,
                threshold_km=threshold_km,
            )
            
            fold_results.append({
                "fold": fold_idx,
                "recall": recall_result["recall"],
                "n_test": len(test_names),
                "n_train": len(train_names),
                "n_hits": recall_result["n_hits"],
            })
        except Exception as e:
            fold_results.append({
                "fold": fold_idx,
                "error": str(e),
            })
    
    valid_recalls = [f["recall"] for f in fold_results if "recall" in f]
    
    return {
        "mean_recall": np.mean(valid_recalls) if valid_recalls else 0.0,
        "std_recall": np.std(valid_recalls) if valid_recalls else 0.0,
        "n_blocks": n_actual_blocks,
        "n_folds": target_folds,
        "block_sizes": {b: len(nodes) for b, nodes in block_nodes.items()},
        "fold_results": fold_results,
        "method": "spatial_block",
    }


# ---------------------------------------------------------
# Ranking Metrics
# ---------------------------------------------------------

def compute_mean_reciprocal_rank(
    candidates: List[Dict],
    known_nodes: Dict,
    threshold_km: float = 100.0,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for known node recovery.
    
    For each known node, find the rank of the first matching candidate.
    MRR = average of 1/rank across all nodes.
    """
    reciprocal_ranks = []
    
    for name, coords in known_nodes.items():
        if isinstance(coords, dict):
            lat, lon = coords["coords"]
        else:
            lat, lon = coords
        
        # Find rank of first matching candidate
        for rank, cand in enumerate(candidates, start=1):
            d = haversine_km(lat, lon, cand["lat"], cand["lon"])
            if d <= threshold_km:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No match found
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_average_distance_to_nearest(
    candidates: List[Dict],
    known_nodes: Dict,
) -> Dict[str, float]:
    """
    Compute average distance from each known node to nearest candidate.
    """
    distances = []
    
    for name, coords in known_nodes.items():
        if isinstance(coords, dict):
            lat, lon = coords["coords"]
        else:
            lat, lon = coords
        
        _, dist = find_nearest_candidate(lat, lon, candidates)
        distances.append(dist)
    
    return {
        "mean_dist_km": np.mean(distances),
        "median_dist_km": np.median(distances),
        "min_dist_km": np.min(distances),
        "max_dist_km": np.max(distances),
        "std_dist_km": np.std(distances),
    }


# ---------------------------------------------------------
# Validation Report
# ---------------------------------------------------------

def generate_validation_report(
    candidates: List[Dict],
    known_nodes: Dict = None,
    threshold_km: float = 100.0,
    k_values: List[int] = [10, 20, 50, 100],
) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        candidates: Prediction candidates
        known_nodes: Ground truth (defaults to KNOWN_NODES_EXTENDED)
        threshold_km: Distance threshold for matching
        k_values: K values for recall@K computation
    
    Returns:
        Formatted report string
    """
    if known_nodes is None:
        known_nodes = get_coords_dict()
    
    lines = []
    lines.append("=" * 60)
    lines.append("GSN VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total candidates: {len(candidates)}")
    lines.append(f"Known nodes: {len(known_nodes)}")
    lines.append(f"Match threshold: {threshold_km} km")
    lines.append("")
    
    # Recall at various K
    lines.append("RECALL @ K")
    lines.append("-" * 40)
    for k in k_values:
        if k <= len(candidates):
            result = compute_recall_at_k(candidates, known_nodes, threshold_km, k)
            lines.append(f"  Recall@{k:3d}: {result['recall']:.3f} ({result['n_hits']}/{result['n_nodes']} nodes)")
    lines.append("")
    
    # Precision at K
    lines.append("PRECISION @ K")
    lines.append("-" * 40)
    for k in k_values:
        if k <= len(candidates):
            result = compute_precision_at_k(candidates, known_nodes, threshold_km, k)
            lines.append(f"  Precision@{k:3d}: {result['precision']:.3f} ({result['n_matched']}/{k} candidates)")
    lines.append("")
    
    # MRR
    mrr = compute_mean_reciprocal_rank(candidates, known_nodes, threshold_km)
    lines.append(f"Mean Reciprocal Rank: {mrr:.4f}")
    lines.append("")
    
    # Distance stats
    dist_stats = compute_average_distance_to_nearest(candidates, known_nodes)
    lines.append("DISTANCE TO NEAREST CANDIDATE")
    lines.append("-" * 40)
    lines.append(f"  Mean:   {dist_stats['mean_dist_km']:.1f} km")
    lines.append(f"  Median: {dist_stats['median_dist_km']:.1f} km")
    lines.append(f"  Min:    {dist_stats['min_dist_km']:.1f} km")
    lines.append(f"  Max:    {dist_stats['max_dist_km']:.1f} km")
    lines.append("")
    
    # Missed nodes
    full_recall = compute_recall_at_k(candidates, known_nodes, threshold_km)
    if full_recall["misses"]:
        lines.append("MISSED NODES (no candidate within threshold)")
        lines.append("-" * 40)
        for name in full_recall["misses"]:
            detail = full_recall["details"][name]
            lines.append(f"  {name}: nearest candidate at {detail['nearest_dist_km']:.1f} km")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    # Demo with synthetic candidates
    print("GSN Validation Framework Demo")
    print("-" * 40)
    
    # Create some dummy candidates near known nodes
    known = get_coords_dict()
    
    candidates = []
    for i, (name, (lat, lon)) in enumerate(list(known.items())[:20]):
        # Add some noise
        candidates.append({
            "lat": lat + np.random.uniform(-0.5, 0.5),
            "lon": lon + np.random.uniform(-0.5, 0.5),
            "F": 2.0 - i * 0.05,
        })
    
    # Add some random candidates
    for i in range(30):
        candidates.append({
            "lat": np.random.uniform(-60, 60),
            "lon": np.random.uniform(-180, 180),
            "F": 1.0 - i * 0.02,
        })
    
    # Sort by F score
    candidates.sort(key=lambda x: x["F"], reverse=True)
    
    # Generate report
    report = generate_validation_report(candidates, known)
    print(report)

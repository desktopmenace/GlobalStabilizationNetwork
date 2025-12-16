#!/usr/bin/env python
"""
GSN World Stress Map Module

Loads and processes crustal stress data from the World Stress Map (WSM) project.

Stress patterns indicate tectonic activity zones and may correlate with 
ancient site placement patterns.

Data Source:
- World Stress Map Project: https://www.world-stress-map.org/
- Download WSM database from: https://www.world-stress-map.org/download

Expected CSV columns:
- ID, LAT, LON, DEPTH, TYPE, REGIME, AZI, QUALITY, REFERENCE

Stress Regimes:
- NF: Normal Faulting (extensional)
- SS: Strike-Slip (transform)
- TF: Thrust Faulting (compressional)
- NS: Normal/Strike-Slip
- TS: Thrust/Strike-Slip
- U: Undefined
"""

import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Expected data file - download from WSM website
DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wsm_data.csv"
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Stress regime codes and weights
# Higher weight = more tectonically active = potentially more interesting for GSN
REGIME_WEIGHTS = {
    "TF": 0.9,   # Thrust faulting - compressional, active
    "TS": 0.8,   # Thrust/Strike-slip
    "SS": 0.7,   # Strike-slip - transform
    "NS": 0.6,   # Normal/Strike-slip
    "NF": 0.5,   # Normal faulting - extensional
    "U": 0.3,    # Undefined
}

# Quality rankings (A is best, E is worst)
QUALITY_WEIGHTS = {
    "A": 1.0,
    "B": 0.8,
    "C": 0.6,
    "D": 0.4,
    "E": 0.2,
}


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

_wsm_cache = None
_kdtree_cache = None


def load_wsm_data(use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load World Stress Map data.
    
    Returns:
        DataFrame with stress measurements
    """
    global _wsm_cache
    
    if _wsm_cache is not None and use_cache:
        return _wsm_cache
    
    if not HAS_PANDAS:
        print("[ERROR] pandas required for WSM data")
        return None
    
    if not os.path.exists(DATA_FILE):
        print(f"[INFO] World Stress Map data not found: {DATA_FILE}")
        print("[INFO] Download from: https://www.world-stress-map.org/download")
        print("[INFO] Save as 'wsm_data.csv' in the GSN directory")
        return None
    
    try:
        # Try to read CSV with various common formats
        df = pd.read_csv(DATA_FILE)
        
        # Normalize column names
        df.columns = [c.upper().strip() for c in df.columns]
        
        # Check for required columns
        required = ["LAT", "LON"]
        if not all(c in df.columns for c in required):
            print(f"[ERROR] WSM data missing required columns. Found: {df.columns.tolist()}")
            return None
        
        # Convert to numeric
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        
        # Remove invalid rows
        df = df.dropna(subset=["LAT", "LON"])
        
        print(f"[INFO] Loaded {len(df)} WSM stress measurements")
        
        _wsm_cache = df
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load WSM data: {e}")
        return None


def is_available() -> bool:
    """Check if WSM data is available."""
    return os.path.exists(DATA_FILE)


# ---------------------------------------------------------
# Stress Queries
# ---------------------------------------------------------

def build_wsm_tree(data: pd.DataFrame = None):
    """Build KD-tree for fast stress data queries."""
    global _kdtree_cache
    
    if _kdtree_cache is not None:
        return _kdtree_cache
    
    if data is None:
        data = load_wsm_data()
    
    if data is None or len(data) == 0:
        return None
    
    if not HAS_SCIPY:
        print("[WARN] scipy required for spatial queries")
        return None
    
    # Convert to 3D Cartesian
    lats_rad = np.radians(data["LAT"].values)
    lons_rad = np.radians(data["LON"].values)
    
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    _kdtree_cache = (tree, data)
    
    return tree, data


def get_stress_regime(lat: float, lon: float, max_dist_km: float = 300.0) -> Optional[str]:
    """
    Get the dominant stress regime near a location.
    
    Args:
        lat, lon: Query coordinates
        max_dist_km: Maximum distance to search
    
    Returns:
        Stress regime code (NF, SS, TF, etc.) or None
    """
    result = build_wsm_tree()
    
    if result is None:
        return None
    
    tree, data = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([[qx, qy, qz]])
    
    # Convert max distance to chord
    angle_rad = max_dist_km / 6371
    chord_radius = 2 * np.sin(angle_rad / 2)
    
    # Query ball
    indices = tree.query_ball_point(query[0], chord_radius)
    
    if not indices:
        return None
    
    # Count regimes
    if "REGIME" not in data.columns:
        return "U"  # Undefined if no regime column
    
    regime_counts = {}
    for idx in indices:
        regime = str(data.iloc[idx].get("REGIME", "U"))
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    # Return most common regime
    if regime_counts:
        return max(regime_counts.items(), key=lambda x: x[1])[0]
    
    return None


def get_stress_score(lat: float, lon: float, max_dist_km: float = 300.0) -> float:
    """
    Get a tectonic stress activity score for a location.
    
    Higher scores indicate more active tectonic regions.
    
    Args:
        lat, lon: Query coordinates
        max_dist_km: Maximum distance to search
    
    Returns:
        Stress score (0-1)
    """
    result = build_wsm_tree()
    
    if result is None:
        return 0.5  # Default neutral score
    
    tree, data = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([[qx, qy, qz]])
    
    # Convert max distance to chord
    angle_rad = max_dist_km / 6371
    chord_radius = 2 * np.sin(angle_rad / 2)
    
    # Query ball
    indices = tree.query_ball_point(query[0], chord_radius)
    
    if not indices:
        return 0.3  # Low score if no nearby data
    
    # Calculate weighted score
    scores = []
    for idx in indices:
        row = data.iloc[idx]
        
        regime = str(row.get("REGIME", "U"))
        quality = str(row.get("QUALITY", "C"))
        
        regime_weight = REGIME_WEIGHTS.get(regime, 0.3)
        quality_weight = QUALITY_WEIGHTS.get(quality, 0.5)
        
        scores.append(regime_weight * quality_weight)
    
    return np.mean(scores) if scores else 0.3


def get_stress_density(lat: float, lon: float, radius_km: float = 200.0) -> int:
    """
    Get the density of stress measurements near a location.
    
    Higher density indicates better characterized tectonic region.
    """
    result = build_wsm_tree()
    
    if result is None:
        return 0
    
    tree, data = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([qx, qy, qz])
    
    # Convert radius to chord
    angle_rad = radius_km / 6371
    chord_radius = 2 * np.sin(angle_rad / 2)
    
    indices = tree.query_ball_point(query, chord_radius)
    
    return len(indices)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def get_wsm_stats() -> Dict:
    """Get statistics about the WSM dataset."""
    data = load_wsm_data()
    
    if data is None:
        return {"available": False}
    
    # Count by regime
    regime_counts = {}
    if "REGIME" in data.columns:
        for regime in data["REGIME"].dropna():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    return {
        "available": True,
        "n_measurements": len(data),
        "lat_range": (float(data["LAT"].min()), float(data["LAT"].max())),
        "lon_range": (float(data["LON"].min()), float(data["LON"].max())),
        "regime_counts": regime_counts,
    }


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN World Stress Map Module Demo")
    print("-" * 40)
    
    if not is_available():
        print("\n[INFO] WSM data not available.")
        print("To use this module:")
        print("1. Visit https://www.world-stress-map.org/download")
        print("2. Download the WSM database (CSV format)")
        print("3. Save as 'wsm_data.csv' in the GSN directory")
    else:
        stats = get_wsm_stats()
        print(f"\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Test queries
        test_points = [
            (35.6762, 139.6503, "Tokyo"),
            (37.7749, -122.4194, "San Francisco"),
            (51.5074, -0.1278, "London"),
        ]
        
        print("\nStress scores at test locations:")
        for lat, lon, name in test_points:
            score = get_stress_score(lat, lon)
            regime = get_stress_regime(lat, lon)
            print(f"  {name}: score={score:.2f}, regime={regime}")

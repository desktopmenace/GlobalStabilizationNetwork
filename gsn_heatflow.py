#!/usr/bin/env python
"""
GSN Heat Flow Data Module

Loads and processes global heat flow data from the International Heat Flow
Commission (IHFC) Global Heat Flow Database.

Heat flow anomalies correlate with geological features (rifts, hotspots, 
ancient cratons) that may influence GSN node placement.
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
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "GHFDB-R2024",
    "IHFC_2024_GHFDB.xlsx"
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
HEATFLOW_GRID_CACHE = os.path.join(CACHE_DIR, "heatflow_grid.npz")

# Column names in the Excel file (row 5 is the header)
COL_HEATFLOW = "q"
COL_LAT = "lat_NS"
COL_LON = "long_EW"

# Grid parameters for interpolation
DEFAULT_GRID_RESOLUTION = 1.0  # degrees


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

_data_cache = None


def load_heatflow_data(use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load heat flow measurements from IHFC database.
    
    Returns:
        DataFrame with columns: lat, lon, heatflow (mW/m²)
    """
    global _data_cache
    
    if _data_cache is not None and use_cache:
        return _data_cache
    
    if not HAS_PANDAS:
        print("[ERROR] pandas required for heat flow data. Install with: pip install pandas openpyxl")
        return None
    
    if not os.path.exists(DATA_FILE):
        print(f"[WARN] Heat flow data file not found: {DATA_FILE}")
        return None
    
    print(f"[INFO] Loading heat flow data from {DATA_FILE}...")
    
    try:
        # Read Excel file, header is at row 5 (0-indexed)
        df = pd.read_excel(DATA_FILE, header=5)
        
        # Extract relevant columns
        if COL_HEATFLOW not in df.columns or COL_LAT not in df.columns or COL_LON not in df.columns:
            print(f"[ERROR] Required columns not found. Available: {df.columns.tolist()[:10]}")
            return None
        
        # Create clean dataframe
        result = pd.DataFrame({
            "lat": pd.to_numeric(df[COL_LAT], errors="coerce"),
            "lon": pd.to_numeric(df[COL_LON], errors="coerce"),
            "heatflow": pd.to_numeric(df[COL_HEATFLOW], errors="coerce"),
        })
        
        # Remove invalid rows
        result = result.dropna()
        
        # Filter out unrealistic values
        result = result[(result["heatflow"] > 0) & (result["heatflow"] < 1000)]
        result = result[(result["lat"] >= -90) & (result["lat"] <= 90)]
        result = result[(result["lon"] >= -180) & (result["lon"] <= 180)]
        
        print(f"[INFO] Loaded {len(result)} valid heat flow measurements")
        print(f"[INFO] Heat flow range: {result['heatflow'].min():.1f} - {result['heatflow'].max():.1f} mW/m²")
        
        _data_cache = result
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to load heat flow data: {e}")
        return None


# ---------------------------------------------------------
# Interpolation
# ---------------------------------------------------------

_interpolator_cache = None
_kdtree_cache = None


def build_interpolator(data: pd.DataFrame = None):
    """Build KD-tree and data arrays for fast nearest-neighbor interpolation."""
    global _interpolator_cache, _kdtree_cache
    
    if _kdtree_cache is not None:
        return _kdtree_cache, _interpolator_cache
    
    if data is None:
        data = load_heatflow_data()
    
    if data is None or len(data) == 0:
        return None, None
    
    if not HAS_SCIPY:
        print("[WARN] scipy required for interpolation")
        return None, None
    
    # Build KD-tree in radians for great-circle queries
    lats_rad = np.radians(data["lat"].values)
    lons_rad = np.radians(data["lon"].values)
    
    # Convert to 3D Cartesian for proper spherical distance
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    values = data["heatflow"].values
    
    _kdtree_cache = tree
    _interpolator_cache = (coords, values, data)
    
    print(f"[INFO] Built heat flow interpolator with {len(data)} points")
    
    return tree, (coords, values, data)


def get_heatflow(lat: float, lon: float, k: int = 5, max_dist_km: float = 500.0) -> Optional[float]:
    """
    Get interpolated heat flow value at a location.
    
    Uses inverse-distance weighted average of k nearest measurements.
    
    Args:
        lat, lon: Query coordinates
        k: Number of nearest neighbors to use
        max_dist_km: Maximum distance to consider (returns None if all points farther)
    
    Returns:
        Heat flow in mW/m² or None if no nearby data
    """
    tree, interp_data = build_interpolator()
    
    if tree is None or interp_data is None:
        return None
    
    coords, values, _ = interp_data
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([[qx, qy, qz]])
    
    # Query k nearest neighbors
    distances, indices = tree.query(query, k=min(k, len(values)))
    
    # Convert chord distance to approximate km (Earth radius ~6371 km)
    # chord = 2 * sin(angle/2), so angle = 2 * arcsin(chord/2)
    dist_km = 2 * 6371 * np.arcsin(distances[0] / 2)
    
    # Filter by max distance
    valid = dist_km < max_dist_km
    if not np.any(valid):
        return None
    
    valid_indices = indices[0][valid]
    valid_distances = dist_km[valid]
    valid_values = values[valid_indices]
    
    # Inverse distance weighting
    if len(valid_distances) == 1:
        return float(valid_values[0])
    
    # Avoid division by zero
    weights = 1.0 / (valid_distances + 0.1)
    weighted_avg = np.sum(weights * valid_values) / np.sum(weights)
    
    return float(weighted_avg)


# ---------------------------------------------------------
# Grid Computation
# ---------------------------------------------------------

def compute_heatflow_grid(
    resolution_deg: float = DEFAULT_GRID_RESOLUTION,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a global heat flow grid via interpolation.
    
    Args:
        resolution_deg: Grid resolution in degrees
        use_cache: Use cached grid if available
    
    Returns:
        Tuple of (heatflow_grid, lats, lons)
    """
    # Check cache
    if use_cache and os.path.exists(HEATFLOW_GRID_CACHE):
        try:
            data = np.load(HEATFLOW_GRID_CACHE)
            cached_res = data.get("resolution", resolution_deg)
            if abs(cached_res - resolution_deg) < 0.01:
                print(f"[INFO] Loaded heat flow grid from cache")
                return data["grid"], data["lats"], data["lons"]
        except Exception as e:
            print(f"[WARN] Could not load cache: {e}")
    
    print(f"[INFO] Computing heat flow grid at {resolution_deg}° resolution...")
    
    lats = np.arange(-90 + resolution_deg/2, 90, resolution_deg)
    lons = np.arange(-180 + resolution_deg/2, 180, resolution_deg)
    
    grid = np.zeros((len(lats), len(lons)))
    
    total = len(lats) * len(lons)
    count = 0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            hf = get_heatflow(lat, lon)
            grid[i, j] = hf if hf is not None else np.nan
            count += 1
        
        if (i + 1) % 20 == 0:
            print(f"[INFO] Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    # Fill NaN values with global mean
    valid = ~np.isnan(grid)
    if np.any(valid):
        global_mean = np.nanmean(grid)
        grid[~valid] = global_mean
        print(f"[INFO] Filled {np.sum(~valid)} NaN cells with mean ({global_mean:.1f} mW/m²)")
    
    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        np.savez(HEATFLOW_GRID_CACHE, grid=grid, lats=lats, lons=lons, resolution=resolution_deg)
        print(f"[INFO] Cached heat flow grid")
    except Exception as e:
        print(f"[WARN] Could not save cache: {e}")
    
    return grid, lats, lons


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def get_heatflow_stats() -> Dict:
    """Get statistics about the heat flow dataset."""
    data = load_heatflow_data()
    
    if data is None:
        return {"available": False}
    
    return {
        "available": True,
        "n_measurements": len(data),
        "mean_heatflow": float(data["heatflow"].mean()),
        "median_heatflow": float(data["heatflow"].median()),
        "std_heatflow": float(data["heatflow"].std()),
        "min_heatflow": float(data["heatflow"].min()),
        "max_heatflow": float(data["heatflow"].max()),
        "lat_range": (float(data["lat"].min()), float(data["lat"].max())),
        "lon_range": (float(data["lon"].min()), float(data["lon"].max())),
    }


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Heat Flow Data Module Demo")
    print("-" * 40)
    
    # Load data
    data = load_heatflow_data()
    
    if data is not None:
        stats = get_heatflow_stats()
        print(f"\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Test interpolation at a few locations
        test_points = [
            (29.9792, 31.1342, "Giza"),
            (51.1789, -1.8262, "Stonehenge"),
            (35.6762, 139.6503, "Tokyo"),
            (-13.1631, -72.5450, "Machu Picchu"),
        ]
        
        print("\nHeat flow at test locations:")
        for lat, lon, name in test_points:
            hf = get_heatflow(lat, lon)
            print(f"  {name}: {hf:.1f} mW/m²" if hf else f"  {name}: No data")

#!/usr/bin/env python
"""
GSN Seismic Data Module

Fetches earthquake data from USGS Earthquake Catalog API and computes
seismic density grids for use in GSN node prediction.

Seismic activity correlates with tectonic boundaries and may indicate
geologically significant locations relevant to GSN node placement.
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Cache file for earthquake data
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
EARTHQUAKE_CACHE_FILE = os.path.join(CACHE_DIR, "earthquakes.json")
SEISMIC_GRID_CACHE_FILE = os.path.join(CACHE_DIR, "seismic_density_grid.npz")

# Default parameters
DEFAULT_MIN_MAGNITUDE = 4.0
DEFAULT_START_DATE = "1970-01-01"
DEFAULT_CELL_SIZE_DEG = 1.0


# ---------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------

def fetch_earthquakes(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = None,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    max_results: int = 20000,
    use_cache: bool = True,
) -> List[Dict]:
    """
    Fetch historical earthquakes from USGS Earthquake Catalog API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date (defaults to today)
        min_magnitude: Minimum earthquake magnitude
        max_results: Maximum number of results per request
        use_cache: Whether to use cached data if available
    
    Returns:
        List of earthquake dicts with 'lat', 'lon', 'magnitude', 'depth', 'time'
    """
    if not HAS_REQUESTS:
        print("[ERROR] requests library required. Install with: pip install requests")
        return []
    
    # Check cache
    if use_cache and os.path.exists(EARTHQUAKE_CACHE_FILE):
        try:
            with open(EARTHQUAKE_CACHE_FILE, 'r') as f:
                cached = json.load(f)
            print(f"[INFO] Loaded {len(cached)} earthquakes from cache")
            return cached
        except Exception as e:
            print(f"[WARN] Could not load cache: {e}")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"[INFO] Fetching earthquakes from USGS API...")
    print(f"[INFO] Date range: {start_date} to {end_date}, min magnitude: {min_magnitude}")
    
    all_earthquakes = []
    
    # USGS API has limits, so we may need to paginate by time
    # Split into yearly chunks for large date ranges
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=365), end)
        
        params = {
            "format": "geojson",
            "starttime": current_start.strftime("%Y-%m-%d"),
            "endtime": current_end.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "limit": max_results,
            "orderby": "time",
        }
        
        try:
            response = requests.get(USGS_API_BASE, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            features = data.get("features", [])
            for feature in features:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates", [])
                
                if len(coords) >= 2:
                    all_earthquakes.append({
                        "lon": coords[0],
                        "lat": coords[1],
                        "depth": coords[2] if len(coords) > 2 else 0,
                        "magnitude": props.get("mag", 0),
                        "time": props.get("time"),
                        "place": props.get("place", ""),
                    })
            
            print(f"[INFO] Fetched {len(features)} earthquakes for {current_start.year}")
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch earthquakes: {e}")
        
        current_start = current_end
    
    print(f"[INFO] Total earthquakes fetched: {len(all_earthquakes)}")
    
    # Save to cache
    if all_earthquakes:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(EARTHQUAKE_CACHE_FILE, 'w') as f:
                json.dump(all_earthquakes, f)
            print(f"[INFO] Cached earthquake data to {EARTHQUAKE_CACHE_FILE}")
        except Exception as e:
            print(f"[WARN] Could not save cache: {e}")
    
    return all_earthquakes


# ---------------------------------------------------------
# Seismic Density Grid Computation
# ---------------------------------------------------------

def compute_seismic_density_grid(
    earthquakes: List[Dict] = None,
    cell_size_deg: float = DEFAULT_CELL_SIZE_DEG,
    weight_by_magnitude: bool = True,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute seismic density grid from earthquake data.
    
    Args:
        earthquakes: List of earthquake dicts (fetched if None)
        cell_size_deg: Grid cell size in degrees
        weight_by_magnitude: Weight counts by earthquake magnitude
        use_cache: Use cached grid if available
    
    Returns:
        Tuple of (density_grid, lats, lons)
        - density_grid: 2D array of seismic density values
        - lats: 1D array of latitude centers
        - lons: 1D array of longitude centers
    """
    # Check cache
    if use_cache and os.path.exists(SEISMIC_GRID_CACHE_FILE):
        try:
            data = np.load(SEISMIC_GRID_CACHE_FILE)
            print(f"[INFO] Loaded seismic density grid from cache")
            return data['grid'], data['lats'], data['lons']
        except Exception as e:
            print(f"[WARN] Could not load grid cache: {e}")
    
    # Fetch earthquakes if not provided
    if earthquakes is None:
        earthquakes = fetch_earthquakes()
    
    if not earthquakes:
        print("[WARN] No earthquake data available")
        # Return empty grid
        lats = np.arange(-90 + cell_size_deg/2, 90, cell_size_deg)
        lons = np.arange(-180 + cell_size_deg/2, 180, cell_size_deg)
        return np.zeros((len(lats), len(lons))), lats, lons
    
    print(f"[INFO] Computing seismic density grid ({cell_size_deg}° cells)...")
    
    # Create grid
    lats = np.arange(-90 + cell_size_deg/2, 90, cell_size_deg)
    lons = np.arange(-180 + cell_size_deg/2, 180, cell_size_deg)
    
    nlat, nlon = len(lats), len(lons)
    density_grid = np.zeros((nlat, nlon))
    count_grid = np.zeros((nlat, nlon))
    
    # Bin earthquakes into grid cells
    for eq in earthquakes:
        lat = eq["lat"]
        lon = eq["lon"]
        mag = eq.get("magnitude", 1.0) or 1.0
        
        # Find grid cell
        lat_idx = int((lat + 90) / cell_size_deg)
        lon_idx = int((lon + 180) / cell_size_deg)
        
        # Clamp to valid range
        lat_idx = max(0, min(nlat - 1, lat_idx))
        lon_idx = max(0, min(nlon - 1, lon_idx))
        
        # Add to grid
        if weight_by_magnitude:
            # Energy scales as 10^(1.5*M), use log for normalization
            # Simplified: use magnitude directly as weight
            density_grid[lat_idx, lon_idx] += mag
        else:
            density_grid[lat_idx, lon_idx] += 1
        
        count_grid[lat_idx, lon_idx] += 1
    
    # Normalize: log transform to handle wide range
    # Add 1 to avoid log(0)
    density_grid = np.log1p(density_grid)
    
    # Statistics
    nonzero = density_grid > 0
    print(f"[INFO] Seismic grid: {nlat}x{nlon}, {np.sum(nonzero)} cells with activity")
    print(f"[INFO] Density range: {density_grid.min():.2f} - {density_grid.max():.2f}")
    
    # Cache the result
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        np.savez(SEISMIC_GRID_CACHE_FILE, grid=density_grid, lats=lats, lons=lons)
        print(f"[INFO] Cached seismic grid to {SEISMIC_GRID_CACHE_FILE}")
    except Exception as e:
        print(f"[WARN] Could not save grid cache: {e}")
    
    return density_grid, lats, lons


def get_seismic_density(lat: float, lon: float, grid_data: Tuple = None) -> float:
    """
    Get seismic density value at a specific location.
    
    Args:
        lat, lon: Coordinates
        grid_data: Pre-computed (density_grid, lats, lons) tuple
    
    Returns:
        Seismic density value (log-scaled)
    """
    if grid_data is None:
        grid_data = compute_seismic_density_grid()
    
    density_grid, lats, lons = grid_data
    
    # Find nearest grid cell
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    
    return float(density_grid[lat_idx, lon_idx])


# ---------------------------------------------------------
# Seismic Feature Extraction
# ---------------------------------------------------------

def compute_seismic_features(lat: float, lon: float, earthquakes: List[Dict] = None) -> Dict:
    """
    Compute multiple seismic features for a location.
    
    Returns:
        Dict with:
        - density: Log-scaled earthquake count
        - max_magnitude: Largest earthquake within radius
        - mean_depth: Average depth of nearby earthquakes
        - recent_activity: Earthquakes in last 10 years
    """
    if earthquakes is None:
        earthquakes = fetch_earthquakes()
    
    # Radius for "nearby" earthquakes (degrees)
    radius = 2.0
    
    nearby = []
    for eq in earthquakes:
        dlat = eq["lat"] - lat
        dlon = eq["lon"] - lon
        dist = math.sqrt(dlat**2 + dlon**2)
        if dist <= radius:
            nearby.append(eq)
    
    if not nearby:
        return {
            "density": 0.0,
            "max_magnitude": 0.0,
            "mean_depth": 0.0,
            "recent_activity": 0.0,
            "n_earthquakes": 0,
        }
    
    magnitudes = [eq.get("magnitude", 0) or 0 for eq in nearby]
    depths = [eq.get("depth", 0) or 0 for eq in nearby]
    
    # Recent activity (last 10 years)
    ten_years_ago = (datetime.now() - timedelta(days=3650)).timestamp() * 1000
    recent = [eq for eq in nearby if (eq.get("time") or 0) > ten_years_ago]
    
    return {
        "density": math.log1p(len(nearby)),
        "max_magnitude": max(magnitudes),
        "mean_depth": np.mean(depths) if depths else 0.0,
        "recent_activity": math.log1p(len(recent)),
        "n_earthquakes": len(nearby),
    }


# ---------------------------------------------------------
# Cached Grid Access
# ---------------------------------------------------------

_seismic_grid_cache = None


def get_cached_seismic_grid():
    """Get or compute the cached seismic density grid."""
    global _seismic_grid_cache
    
    if _seismic_grid_cache is None:
        _seismic_grid_cache = compute_seismic_density_grid()
    
    return _seismic_grid_cache


def clear_cache():
    """Clear all cached data."""
    global _seismic_grid_cache
    _seismic_grid_cache = None
    
    for f in [EARTHQUAKE_CACHE_FILE, SEISMIC_GRID_CACHE_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Removed cache file: {f}")


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Seismic Data Module Demo")
    print("-" * 40)
    
    # Fetch earthquakes (will use cache if available)
    earthquakes = fetch_earthquakes(
        start_date="2020-01-01",
        min_magnitude=5.0,
        use_cache=True
    )
    
    print(f"\nFetched {len(earthquakes)} earthquakes")
    
    if earthquakes:
        # Show some statistics
        mags = [eq["magnitude"] for eq in earthquakes if eq.get("magnitude")]
        print(f"Magnitude range: {min(mags):.1f} - {max(mags):.1f}")
        
        # Compute density grid
        grid, lats, lons = compute_seismic_density_grid(earthquakes, cell_size_deg=2.0)
        print(f"Grid shape: {grid.shape}")
        
        # Test point lookup
        test_lat, test_lon = 35.0, 140.0  # Japan
        density = get_seismic_density(test_lat, test_lon, (grid, lats, lons))
        print(f"\nSeismic density at Japan ({test_lat}, {test_lon}): {density:.3f}")
        
        test_lat, test_lon = 29.9792, 31.1342  # Giza
        density = get_seismic_density(test_lat, test_lon, (grid, lats, lons))
        print(f"Seismic density at Giza ({test_lat}, {test_lon}): {density:.3f}")

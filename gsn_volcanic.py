#!/usr/bin/env python
"""
GSN Volcanic Data Module

Loads volcano data from the Smithsonian Global Volcanism Program (GVP)
Holocene Volcano Database and computes volcanic proximity metrics.

Many ancient sites are located near volcanic features, suggesting 
volcanic activity may be relevant for GSN node prediction.
"""

import os
import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "GVP_Volcano_List_Holocene_202512122258.xls"
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
VOLCANO_GRID_CACHE = os.path.join(CACHE_DIR, "volcano_distance_grid.npz")

# XML Spreadsheet namespace
XML_NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

# Column indices in the spreadsheet (0-indexed after header row)
COL_NAME = 1
COL_COUNTRY = 2
COL_TYPE = 6
COL_LAST_ERUPTION = 8
COL_LAT = 9
COL_LON = 10
COL_ELEVATION = 11

# Grid parameters
DEFAULT_GRID_RESOLUTION = 1.0  # degrees


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

_volcano_cache = None
_kdtree_cache = None


def load_volcano_data(use_cache: bool = True) -> Optional[List[Dict]]:
    """
    Load volcano data from GVP XML spreadsheet.
    
    Returns:
        List of dicts with keys: name, lat, lon, country, type, elevation, last_eruption
    """
    global _volcano_cache
    
    if _volcano_cache is not None and use_cache:
        return _volcano_cache
    
    if not HAS_LXML:
        print("[ERROR] lxml required for volcano data. Install with: pip install lxml")
        return None
    
    if not os.path.exists(DATA_FILE):
        print(f"[WARN] Volcano data file not found: {DATA_FILE}")
        return None
    
    print(f"[INFO] Loading volcano data from {DATA_FILE}...")
    
    try:
        # Read and clean file content
        with open(DATA_FILE, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Remove problematic characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
        
        # Parse XML
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(content.encode('utf-8'), parser)
        
        # Find worksheet and table
        worksheet = root.find('.//ss:Worksheet', XML_NS)
        if worksheet is None:
            print("[ERROR] No worksheet found in file")
            return None
        
        table = worksheet.find('ss:Table', XML_NS)
        if table is None:
            print("[ERROR] No table found in worksheet")
            return None
        
        rows = table.findall('ss:Row', XML_NS)
        if len(rows) < 3:
            print("[ERROR] Not enough rows in table")
            return None
        
        # Parse data rows (skip header rows 0 and 1)
        volcanoes = []
        
        for row in rows[2:]:
            cells = row.findall('ss:Cell', XML_NS)
            if len(cells) < 12:
                continue
            
            def get_cell_value(idx):
                if idx < len(cells):
                    data = cells[idx].find('ss:Data', XML_NS)
                    return data.text if data is not None else None
                return None
            
            try:
                lat_str = get_cell_value(COL_LAT)
                lon_str = get_cell_value(COL_LON)
                
                if lat_str is None or lon_str is None:
                    continue
                
                lat = float(lat_str)
                lon = float(lon_str)
                
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                elev_str = get_cell_value(COL_ELEVATION)
                
                volcano = {
                    "name": get_cell_value(COL_NAME) or "Unknown",
                    "lat": lat,
                    "lon": lon,
                    "country": get_cell_value(COL_COUNTRY) or "Unknown",
                    "type": get_cell_value(COL_TYPE) or "Unknown",
                    "last_eruption": get_cell_value(COL_LAST_ERUPTION) or "Unknown",
                    "elevation": int(float(elev_str)) if elev_str else 0,
                }
                
                volcanoes.append(volcano)
                
            except (ValueError, TypeError):
                continue
        
        print(f"[INFO] Loaded {len(volcanoes)} volcanoes")
        
        _volcano_cache = volcanoes
        return volcanoes
        
    except Exception as e:
        print(f"[ERROR] Failed to load volcano data: {e}")
        return None


# ---------------------------------------------------------
# Distance Computation
# ---------------------------------------------------------

def build_volcano_tree(volcanoes: List[Dict] = None):
    """Build KD-tree for fast volcano proximity queries."""
    global _kdtree_cache
    
    if _kdtree_cache is not None:
        return _kdtree_cache
    
    if volcanoes is None:
        volcanoes = load_volcano_data()
    
    if volcanoes is None or len(volcanoes) == 0:
        return None
    
    if not HAS_SCIPY:
        print("[WARN] scipy required for distance computation")
        return None
    
    # Convert to 3D Cartesian for spherical distance
    lats = np.array([v["lat"] for v in volcanoes])
    lons = np.array([v["lon"] for v in volcanoes])
    
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    _kdtree_cache = (tree, volcanoes)
    
    return tree, volcanoes


def get_distance_to_nearest_volcano(lat: float, lon: float) -> Optional[float]:
    """
    Get distance to the nearest volcano in kilometers.
    
    Args:
        lat, lon: Query coordinates
    
    Returns:
        Distance in km to nearest volcano, or None if no data
    """
    result = build_volcano_tree()
    
    if result is None:
        return None
    
    tree, volcanoes = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([[qx, qy, qz]])
    
    # Query nearest neighbor
    chord_dist, idx = tree.query(query, k=1)
    
    # Convert chord distance to arc distance in km
    # chord = 2 * sin(angle/2), so angle = 2 * arcsin(chord/2)
    dist_km = 2 * 6371 * np.arcsin(chord_dist[0] / 2)
    
    return float(dist_km)


def get_nearby_volcanoes(lat: float, lon: float, radius_km: float = 500.0) -> List[Dict]:
    """
    Get all volcanoes within a given radius.
    
    Args:
        lat, lon: Query coordinates
        radius_km: Search radius in kilometers
    
    Returns:
        List of nearby volcano dicts with added 'distance_km' key
    """
    result = build_volcano_tree()
    
    if result is None:
        return []
    
    tree, volcanoes = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([qx, qy, qz])
    
    # Convert radius to chord distance
    # chord = 2 * sin(angle/2), angle = dist_km / R
    angle_rad = radius_km / 6371
    chord_radius = 2 * np.sin(angle_rad / 2)
    
    # Query ball
    indices = tree.query_ball_point(query, chord_radius)
    
    nearby = []
    for idx in indices:
        volcano = volcanoes[idx].copy()
        # Calculate actual distance
        volcano["distance_km"] = get_distance_to_nearest_volcano(volcano["lat"], volcano["lon"])
        nearby.append(volcano)
    
    # Sort by distance
    nearby.sort(key=lambda v: v.get("distance_km", float('inf')))
    
    return nearby


def get_volcanic_density(lat: float, lon: float, radius_km: float = 300.0) -> int:
    """
    Get the count of volcanoes within a given radius.
    
    Args:
        lat, lon: Query coordinates
        radius_km: Search radius in kilometers
    
    Returns:
        Number of volcanoes within radius
    """
    return len(get_nearby_volcanoes(lat, lon, radius_km))


# ---------------------------------------------------------
# Grid Computation
# ---------------------------------------------------------

def compute_volcano_distance_grid(
    resolution_deg: float = DEFAULT_GRID_RESOLUTION,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a global grid of distances to nearest volcano.
    
    Args:
        resolution_deg: Grid resolution in degrees
        use_cache: Use cached grid if available
    
    Returns:
        Tuple of (distance_grid, lats, lons) where distances are in km
    """
    # Check cache
    if use_cache and os.path.exists(VOLCANO_GRID_CACHE):
        try:
            data = np.load(VOLCANO_GRID_CACHE)
            cached_res = data.get("resolution", resolution_deg)
            if abs(cached_res - resolution_deg) < 0.01:
                print(f"[INFO] Loaded volcano distance grid from cache")
                return data["grid"], data["lats"], data["lons"]
        except Exception as e:
            print(f"[WARN] Could not load cache: {e}")
    
    print(f"[INFO] Computing volcano distance grid at {resolution_deg}° resolution...")
    
    lats = np.arange(-90 + resolution_deg/2, 90, resolution_deg)
    lons = np.arange(-180 + resolution_deg/2, 180, resolution_deg)
    
    grid = np.zeros((len(lats), len(lons)))
    
    total = len(lats) * len(lons)
    count = 0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            dist = get_distance_to_nearest_volcano(lat, lon)
            grid[i, j] = dist if dist is not None else 10000.0  # Max distance for missing
            count += 1
        
        if (i + 1) % 20 == 0:
            print(f"[INFO] Progress: {count}/{total} ({100*count/total:.1f}%)")
    
    print(f"[INFO] Distance range: {grid.min():.1f} - {grid.max():.1f} km")
    
    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        np.savez(VOLCANO_GRID_CACHE, grid=grid, lats=lats, lons=lons, resolution=resolution_deg)
        print(f"[INFO] Cached volcano distance grid")
    except Exception as e:
        print(f"[WARN] Could not save cache: {e}")
    
    return grid, lats, lons


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def get_volcano_stats() -> Dict:
    """Get statistics about the volcano dataset."""
    volcanoes = load_volcano_data()
    
    if volcanoes is None:
        return {"available": False}
    
    elevations = [v["elevation"] for v in volcanoes if v["elevation"] != 0]
    
    # Count by type
    type_counts = {}
    for v in volcanoes:
        vtype = v["type"]
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    
    # Top 5 types
    top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:5]
    
    return {
        "available": True,
        "n_volcanoes": len(volcanoes),
        "n_countries": len(set(v["country"] for v in volcanoes)),
        "mean_elevation": np.mean(elevations) if elevations else 0,
        "max_elevation": max(elevations) if elevations else 0,
        "top_types": dict(top_types),
    }


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Volcanic Data Module Demo")
    print("-" * 40)
    
    # Load data
    volcanoes = load_volcano_data()
    
    if volcanoes:
        stats = get_volcano_stats()
        print(f"\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Test distance computation at a few locations
        test_points = [
            (29.9792, 31.1342, "Giza"),
            (51.1789, -1.8262, "Stonehenge"),
            (35.6762, 139.6503, "Tokyo"),
            (-13.1631, -72.5450, "Machu Picchu"),
            (19.4326, -99.1332, "Mexico City"),
        ]
        
        print("\nDistance to nearest volcano:")
        for lat, lon, name in test_points:
            dist = get_distance_to_nearest_volcano(lat, lon)
            print(f"  {name}: {dist:.1f} km" if dist else f"  {name}: No data")
        
        # Show nearby volcanoes for one location
        print("\nVolcanoes near Tokyo (within 300km):")
        nearby = get_nearby_volcanoes(35.6762, 139.6503, 300)
        for v in nearby[:5]:
            print(f"  {v['name']} ({v['country']}): {v.get('distance_km', 0):.1f} km")

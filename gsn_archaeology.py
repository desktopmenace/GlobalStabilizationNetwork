#!/usr/bin/env python
"""
GSN Archaeological Sites Database

Extends the known_nodes_extended database with additional ancient sites
from various public sources for enhanced H (geometric coherence) scoring.

Sources:
- Pleiades gazetteer of ancient places (subset)
- UNESCO World Heritage Sites (cultural)
- Megalithic Portal database
- Academic archaeological surveys

Site categories are weighted by relevance to GSN prediction:
- megalithic/pyramid: highest weight (core GSN pattern)
- temple/observatory: high weight (astronomical alignment)
- ancient_city: medium weight (may follow similar placement patterns)
- fortification: lower weight (defensive rather than sacred placement)
"""

import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import existing extended nodes
try:
    from known_nodes_extended import KNOWN_NODES_EXTENDED, CATEGORIES
    HAS_EXTENDED = True
except ImportError:
    KNOWN_NODES_EXTENDED = {}
    CATEGORIES = {}
    HAS_EXTENDED = False

# ---------------------------------------------------------
# Site Category Weights
# ---------------------------------------------------------

SITE_WEIGHTS = {
    "pyramid": 1.0,           # Core GSN pattern
    "megalithic": 1.0,        # Core GSN pattern  
    "observatory": 0.9,       # Astronomical alignment
    "temple": 0.8,            # Sacred geometry
    "geoglyph": 0.8,          # Large-scale earth works
    "ancient_city": 0.6,      # Urban centers
    "sacred_site": 0.7,       # Religious significance
    "burial_mound": 0.5,      # May follow patterns
    "fortification": 0.3,     # Defensive placement
    "unknown": 0.4,           # Default weight
}

# ---------------------------------------------------------
# Additional Archaeological Sites
# ---------------------------------------------------------
# These supplement the known_nodes_extended.py database

ADDITIONAL_SITES = {
    # ==================== EUROPE ====================
    "Avebury": {
        "coords": (51.4281, -1.8542),
        "category": "megalithic",
        "age_bp": 4500,
        "country": "United Kingdom",
        "notes": "Largest stone circle in Britain",
    },
    "Ring_of_Brodgar": {
        "coords": (59.0017, -3.2294),
        "category": "megalithic",
        "age_bp": 4500,
        "country": "United Kingdom",
        "notes": "Neolithic henge in Orkney",
    },
    "Skara_Brae": {
        "coords": (59.0486, -3.3419),
        "category": "ancient_city",
        "age_bp": 5100,
        "country": "United Kingdom",
        "notes": "Neolithic settlement in Orkney",
    },
    "Callanish": {
        "coords": (58.1975, -6.7453),
        "category": "megalithic",
        "age_bp": 4800,
        "country": "United Kingdom",
        "notes": "Standing stones on Isle of Lewis",
    },
    "Almendres_Cromlech": {
        "coords": (38.5569, -8.0611),
        "category": "megalithic",
        "age_bp": 7000,
        "country": "Portugal",
        "notes": "One of largest cromlechs in Iberian Peninsula",
    },
    "Antequera_Dolmens": {
        "coords": (37.0236, -4.5472),
        "category": "megalithic",
        "age_bp": 5700,
        "country": "Spain",
        "notes": "UNESCO megalithic tombs",
    },
    "Bru_na_Boinne": {
        "coords": (53.6947, -6.4750),
        "category": "megalithic",
        "age_bp": 5200,
        "country": "Ireland",
        "notes": "Newgrange, Knowth, Dowth passage tombs",
    },
    "Externsteine": {
        "coords": (51.8686, 8.9161),
        "category": "sacred_site",
        "age_bp": 3000,
        "country": "Germany",
        "notes": "Sandstone rock formation with ancient carvings",
    },
    
    # ==================== MIDDLE EAST ====================
    "Baalbek": {
        "coords": (34.0069, 36.2039),
        "category": "temple",
        "age_bp": 2000,
        "country": "Lebanon",
        "notes": "Roman temple complex with massive foundation stones",
    },
    "Petra": {
        "coords": (30.3285, 35.4444),
        "category": "ancient_city",
        "age_bp": 2300,
        "country": "Jordan",
        "notes": "Nabataean rock-cut city",
    },
    "Jerash": {
        "coords": (32.2747, 35.8914),
        "category": "ancient_city",
        "age_bp": 2600,
        "country": "Jordan",
        "notes": "Roman Decapolis city",
    },
    "Palmyra": {
        "coords": (34.5514, 38.2689),
        "category": "ancient_city",
        "age_bp": 4000,
        "country": "Syria",
        "notes": "Ancient Semitic city",
    },
    
    # ==================== ASIA ====================
    "Sanchi": {
        "coords": (23.4795, 77.7398),
        "category": "temple",
        "age_bp": 2300,
        "country": "India",
        "notes": "Buddhist complex with Great Stupa",
    },
    "Hampi": {
        "coords": (15.3350, 76.4600),
        "category": "ancient_city",
        "age_bp": 600,
        "country": "India",
        "notes": "Vijayanagara Empire capital",
    },
    "Mohenjo_Daro": {
        "coords": (27.3242, 68.1356),
        "category": "ancient_city",
        "age_bp": 4500,
        "country": "Pakistan",
        "notes": "Indus Valley Civilization",
    },
    "Harappa": {
        "coords": (30.6314, 72.8647),
        "category": "ancient_city",
        "age_bp": 4600,
        "country": "Pakistan",
        "notes": "Indus Valley Civilization",
    },
    "Sigiriya": {
        "coords": (7.9572, 80.7600),
        "category": "ancient_city",
        "age_bp": 1500,
        "country": "Sri Lanka",
        "notes": "Ancient rock fortress",
    },
    "Prambanan": {
        "coords": (-7.7520, 110.4914),
        "category": "temple",
        "age_bp": 1150,
        "country": "Indonesia",
        "notes": "Hindu temple compound",
    },
    "Sukhothai": {
        "coords": (17.0168, 99.7031),
        "category": "ancient_city",
        "age_bp": 750,
        "country": "Thailand",
        "notes": "First capital of Siam",
    },
    "Bagan": {
        "coords": (21.1717, 94.8585),
        "category": "temple",
        "age_bp": 1000,
        "country": "Myanmar",
        "notes": "Ancient city with 2000+ temples",
    },
    
    # ==================== AMERICAS ====================
    "Chaco_Canyon": {
        "coords": (36.0604, -107.9584),
        "category": "ancient_city",
        "age_bp": 1050,
        "country": "USA",
        "notes": "Ancestral Puebloan ceremonial center",
    },
    "Cahokia": {
        "coords": (38.6554, -90.0622),
        "category": "burial_mound",
        "age_bp": 1050,
        "country": "USA",
        "notes": "Largest pre-Columbian settlement north of Mexico",
    },
    "Serpent_Mound": {
        "coords": (39.0253, -83.4303),
        "category": "geoglyph",
        "age_bp": 2000,
        "country": "USA",
        "notes": "Effigy mound in Ohio",
    },
    "Poverty_Point": {
        "coords": (32.6365, -91.4069),
        "category": "burial_mound",
        "age_bp": 3400,
        "country": "USA",
        "notes": "Ancient earthworks in Louisiana",
    },
    "Monte_Alban": {
        "coords": (17.0436, -96.7675),
        "category": "ancient_city",
        "age_bp": 2500,
        "country": "Mexico",
        "notes": "Zapotec capital",
    },
    "El_Tajin": {
        "coords": (20.4475, -97.3783),
        "category": "pyramid",
        "age_bp": 1400,
        "country": "Mexico",
        "notes": "Pyramid of the Niches",
    },
    "Palenque": {
        "coords": (17.4839, -92.0461),
        "category": "ancient_city",
        "age_bp": 1800,
        "country": "Mexico",
        "notes": "Maya city-state",
    },
    "Copan": {
        "coords": (14.8400, -89.1400),
        "category": "ancient_city",
        "age_bp": 2000,
        "country": "Honduras",
        "notes": "Maya archaeological site",
    },
    "Chan_Chan": {
        "coords": (-8.1061, -79.0747),
        "category": "ancient_city",
        "age_bp": 900,
        "country": "Peru",
        "notes": "Largest adobe city in the Americas",
    },
    "Sechin_Bajo": {
        "coords": (-9.4711, -78.2614),
        "category": "pyramid",
        "age_bp": 5500,
        "country": "Peru",
        "notes": "One of oldest structures in the Americas",
    },
    
    # ==================== AFRICA ====================
    "Great_Zimbabwe": {
        "coords": (-20.2674, 30.9333),
        "category": "ancient_city",
        "age_bp": 900,
        "country": "Zimbabwe",
        "notes": "Medieval stone city",
    },
    "Meroe": {
        "coords": (16.9386, 33.7489),
        "category": "pyramid",
        "age_bp": 2700,
        "country": "Sudan",
        "notes": "Nubian pyramids",
    },
    "Lalibela": {
        "coords": (12.0319, 39.0439),
        "category": "temple",
        "age_bp": 800,
        "country": "Ethiopia",
        "notes": "Rock-hewn churches",
    },
    "Aksum": {
        "coords": (14.1211, 38.7197),
        "category": "ancient_city",
        "age_bp": 2700,
        "country": "Ethiopia",
        "notes": "Aksumite Empire capital with obelisks",
    },
    
    # ==================== OCEANIA ====================
    "Nan_Madol": {
        "coords": (6.8428, 158.3353),
        "category": "ancient_city",
        "age_bp": 1200,
        "country": "Micronesia",
        "notes": "Ceremonial center built on artificial islets",
    },
}


# ---------------------------------------------------------
# Combined Site Database
# ---------------------------------------------------------

def get_all_sites() -> Dict:
    """Get combined dictionary of all known archaeological sites."""
    all_sites = {}
    
    # Add extended nodes if available
    if HAS_EXTENDED:
        all_sites.update(KNOWN_NODES_EXTENDED)
    
    # Add additional sites (only if not already present)
    for name, data in ADDITIONAL_SITES.items():
        if name not in all_sites:
            all_sites[name] = data
    
    return all_sites


def get_sites_by_category(category: str) -> Dict:
    """Get all sites of a specific category."""
    all_sites = get_all_sites()
    return {
        name: data for name, data in all_sites.items()
        if data.get("category") == category
    }


def get_sites_by_weight_threshold(min_weight: float = 0.5) -> Dict:
    """Get sites with category weight >= threshold."""
    all_sites = get_all_sites()
    return {
        name: data for name, data in all_sites.items()
        if SITE_WEIGHTS.get(data.get("category", "unknown"), 0.4) >= min_weight
    }


# ---------------------------------------------------------
# Spatial Indexing
# ---------------------------------------------------------

_site_tree_cache = None


def build_site_tree(sites: Dict = None):
    """Build KD-tree for fast site proximity queries."""
    global _site_tree_cache
    
    if _site_tree_cache is not None:
        return _site_tree_cache
    
    if sites is None:
        sites = get_all_sites()
    
    if not sites or not HAS_SCIPY:
        return None
    
    # Convert to arrays
    names = list(sites.keys())
    coords = []
    weights = []
    
    for name in names:
        data = sites[name]
        if isinstance(data, dict):
            lat, lon = data["coords"]
            cat = data.get("category", "unknown")
        else:
            lat, lon = data
            cat = "unknown"
        
        coords.append([lat, lon])
        weights.append(SITE_WEIGHTS.get(cat, 0.4))
    
    coords = np.array(coords)
    weights = np.array(weights)
    
    # Convert to 3D Cartesian for spherical distance
    lats_rad = np.radians(coords[:, 0])
    lons_rad = np.radians(coords[:, 1])
    
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    cart_coords = np.column_stack([x, y, z])
    tree = cKDTree(cart_coords)
    
    _site_tree_cache = (tree, names, coords, weights, sites)
    
    return _site_tree_cache


def get_nearby_sites(lat: float, lon: float, radius_km: float = 500.0) -> List[Dict]:
    """
    Get all archaeological sites within a given radius.
    
    Args:
        lat, lon: Query coordinates
        radius_km: Search radius in kilometers
    
    Returns:
        List of site dicts with added 'distance_km' and 'weight' keys
    """
    result = build_site_tree()
    
    if result is None:
        return []
    
    tree, names, coords, weights, sites = result
    
    # Convert query point to 3D
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    qx = np.cos(lat_rad) * np.cos(lon_rad)
    qy = np.cos(lat_rad) * np.sin(lon_rad)
    qz = np.sin(lat_rad)
    query = np.array([qx, qy, qz])
    
    # Convert radius to chord distance
    angle_rad = radius_km / 6371
    chord_radius = 2 * np.sin(angle_rad / 2)
    
    # Query ball
    indices = tree.query_ball_point(query, chord_radius)
    
    nearby = []
    for idx in indices:
        name = names[idx]
        site = sites[name].copy() if isinstance(sites[name], dict) else {"coords": sites[name]}
        site["name"] = name
        site["weight"] = weights[idx]
        
        # Calculate distance
        slat, slon = coords[idx]
        site["distance_km"] = haversine_km(lat, lon, slat, slon)
        
        nearby.append(site)
    
    # Sort by distance
    nearby.sort(key=lambda s: s["distance_km"])
    
    return nearby


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------
# Enhanced H Scoring with Archaeological Sites
# ---------------------------------------------------------

def compute_H_archaeological(
    lat: float, 
    lon: float, 
    distance_scale: float = 30.0,
    use_weights: bool = True,
    min_category_weight: float = 0.5,
) -> float:
    """
    Compute geometric coherence H using archaeological sites.
    
    This provides additional H score based on proximity and alignment
    with known archaeological sites, weighted by category relevance.
    
    Args:
        lat, lon: Query coordinates
        distance_scale: Decay scale in degrees
        use_weights: Apply category weights
        min_category_weight: Minimum site weight to include
    
    Returns:
        Archaeological coherence score (0-1)
    """
    sites = get_sites_by_weight_threshold(min_category_weight)
    
    if not sites:
        return 0.0
    
    # Import angle calculation from predictor
    try:
        from gsn_node_predictor import great_circle_angle, penrose_kernel_extended
    except ImportError:
        # Fallback: use simple distance-based scoring
        return _simple_site_score(lat, lon, sites, distance_scale)
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for name, data in sites.items():
        if isinstance(data, dict):
            slat, slon = data["coords"]
            cat = data.get("category", "unknown")
        else:
            slat, slon = data
            cat = "unknown"
        
        theta = great_circle_angle(lat, lon, slat, slon)
        
        # Distance weight
        dist_weight = math.exp(-theta / distance_scale)
        
        # Category weight
        cat_weight = SITE_WEIGHTS.get(cat, 0.4) if use_weights else 1.0
        
        # Penrose coherence
        coherence = penrose_kernel_extended(theta)
        
        combined_weight = dist_weight * cat_weight
        weighted_sum += combined_weight * coherence
        weight_sum += combined_weight
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def _simple_site_score(lat: float, lon: float, sites: Dict, scale: float) -> float:
    """Simple distance-based site proximity score."""
    scores = []
    
    for name, data in sites.items():
        if isinstance(data, dict):
            slat, slon = data["coords"]
            cat = data.get("category", "unknown")
        else:
            slat, slon = data
            cat = "unknown"
        
        dist = haversine_km(lat, lon, slat, slon)
        weight = SITE_WEIGHTS.get(cat, 0.4)
        score = weight * math.exp(-dist / (scale * 111))  # 111 km per degree
        scores.append(score)
    
    return max(scores) if scores else 0.0


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def get_site_stats() -> Dict:
    """Get statistics about the archaeological site database."""
    all_sites = get_all_sites()
    
    if not all_sites:
        return {"available": False}
    
    # Count by category
    cat_counts = {}
    for name, data in all_sites.items():
        cat = data.get("category", "unknown") if isinstance(data, dict) else "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    # Count by country
    countries = set()
    for name, data in all_sites.items():
        if isinstance(data, dict):
            countries.add(data.get("country", "Unknown"))
    
    return {
        "available": True,
        "n_sites": len(all_sites),
        "n_categories": len(cat_counts),
        "n_countries": len(countries),
        "category_counts": cat_counts,
        "has_extended_nodes": HAS_EXTENDED,
        "n_additional_sites": len(ADDITIONAL_SITES),
    }


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Archaeological Sites Database Demo")
    print("-" * 40)
    
    stats = get_site_stats()
    print(f"\nDatabase Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test H scoring at a few locations
    test_points = [
        (29.9792, 31.1342, "Giza"),
        (51.1789, -1.8262, "Stonehenge"),
        (35.6762, 139.6503, "Tokyo"),
        (40.7128, -74.0060, "New York"),
    ]
    
    print("\nArchaeological H scores:")
    for lat, lon, name in test_points:
        h = compute_H_archaeological(lat, lon)
        nearby = get_nearby_sites(lat, lon, 200)
        print(f"  {name}: H={h:.3f}, {len(nearby)} sites within 200km")
    
    # Show site categories
    print("\nSite category weights:")
    for cat, weight in sorted(SITE_WEIGHTS.items(), key=lambda x: -x[1]):
        count = stats["category_counts"].get(cat, 0)
        print(f"  {cat}: {weight:.1f} ({count} sites)")

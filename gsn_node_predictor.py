#!/usr/bin/env python
import math
import os
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

# Optional dependency imports for real data mode
try:
    import xarray as xr
    USE_XARRAY = True
except ImportError:
    USE_XARRAY = False

try:
    from scipy import ndimage
    USE_SCIPY = True
except ImportError:
    USE_SCIPY = False

# ---------------------------------------------------------
# CONFIG: Paths and variable names for NetCDF datasets
# ---------------------------------------------------------
GRAVITY_FILE = "gravity_model.nc"   # replace with your gravity file
CRUST_FILE   = "crust_model.nc"     # replace with your crust file
GA_VAR_NAME  = "ga"                 # gravity anomaly variable name in GRAVITY_FILE
CT_VAR_NAME  = "thickness"          # crustal thickness variable name in CRUST_FILE
BOUNDARY_FILE = "gsn_data/plate-boundaries.kmz"  # KML/KMZ of plate boundaries

# ---------------------------------------------------------
# Known GSN Nodes (simplified set, degrees)
# ---------------------------------------------------------
KNOWN_NODES = {
    "Giza": (29.9792, 31.1342),               # Egypt
    "Teotihuacan": (19.6925, -98.8433),       # Mexico
    "Shaanxi_Pyramids": (34.0, 109.0),        # China (approx)
    "Gunung_Padang": (-6.59, 107.05),         # Indonesia
    "Moche": (-8.0, -78.0),                   # Peru (approx)
    "Tiwanaku": (-16.55, -68.67),             # Bolivia
    "Mt_Hayes": (63.63, -146.83),             # Alaska
    "Geikie_Peninsula": (70.3, -27.0),        # East Greenland (approx)
    "Azores_Plateau": (38.7, -28.0)           # Azores region (approx)
}

# Penrose-like preferred angles (degrees)
PENROSE_ANGLES = [36.0, 72.0, 108.0, 144.0]

# Extended sacred geometry angles (degrees)
EXTENDED_ANGLES = [
    19.47,   # Tetrahedral angle (arcsin(1/3))
    23.5,    # Earth's axial tilt
    30.0,    # 1/12 of circle
    36.0,    # Penrose / golden ratio
    45.0,    # 1/8 of circle
    60.0,    # 1/6 of circle (hexagonal/equilateral)
    72.0,    # Penrose / pentagon
    90.0,    # 1/4 of circle
    108.0,   # Penrose / pentagon interior
    120.0,   # 1/3 of circle (hexagon interior)
    137.5,   # Fibonacci golden angle (360/phi^2)
    144.0,   # Penrose
    150.0,   # 5/12 of circle
    180.0,   # Antipodal
]

# Sacred angle descriptions for display
ANGLE_DESCRIPTIONS = {
    19.47: "Tetrahedral",
    23.5: "Axial tilt",
    30.0: "Dodecagon",
    36.0: "Penrose/Golden",
    45.0: "Octagon",
    60.0: "Hexagon/Triangle",
    72.0: "Pentagon",
    90.0: "Quadrant",
    108.0: "Pentagon interior",
    120.0: "Hexagon interior",
    137.5: "Fibonacci golden",
    144.0: "Penrose",
    150.0: "Dodecagon",
    180.0: "Antipodal",
}


# ---------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------
def deg2rad(deg):
    return deg * math.pi / 180.0


def great_circle_angle(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance angle (in degrees) between two points on a sphere.
    """
    phi1, phi2 = deg2rad(lat1), deg2rad(lat2)
    dlon = deg2rad(lon2 - lon1)
    cos_angle = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(dlon)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    return angle_rad * 180.0 / math.pi


def frange(start, stop, step):
    """
    Floating-point range generator.
    """
    x = start
    while x <= stop + 1e-6:
        yield x
        x += step


# ---------------------------------------------------------
# Penrose / GSN geometry layer (H)
# ---------------------------------------------------------
def penrose_kernel(theta_deg, sigma=6.0):
    """
    Penrose-like angular coherence: high when theta is near
    36°, 72°, 108°, or 144°.
    """
    max_val = 0.0
    for target in PENROSE_ANGLES:
        diff = theta_deg - target
        val = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
        if val > max_val:
            max_val = val
    return max_val


def compute_H(lat, lon):
    """
    Geometric coherence H based on angular relation to known GSN nodes.
    Higher H -> candidate is harmonically consistent with GSN pattern.
    """
    scores = []
    for (nlat, nlon) in KNOWN_NODES.values():
        theta = great_circle_angle(lat, lon, nlat, nlon)
        scores.append(penrose_kernel(theta))
    return sum(scores)/len(scores) if scores else 0.0


def penrose_kernel_extended(theta_deg, sigma=6.0):
    """
    Extended Penrose-like angular coherence using additional sacred geometry angles.
    High when theta is near any of the extended angle set.
    """
    max_val = 0.0
    for target in EXTENDED_ANGLES:
        diff = theta_deg - target
        val = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
        if val > max_val:
            max_val = val
    return max_val


def compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True, known_nodes=None):
    """
    Distance-weighted geometric coherence H.
    
    Closer nodes contribute more than distant ones, providing better
    spatial sensitivity for node prediction.
    
    Args:
        lat, lon: Coordinates to evaluate
        distance_scale: Exponential decay scale in degrees (smaller = more local)
        use_extended: Use extended sacred geometry angles
        known_nodes: Override default KNOWN_NODES dict
    
    Returns:
        Weighted geometric coherence score
    """
    if known_nodes is None:
        known_nodes = KNOWN_NODES
    
    kernel_func = penrose_kernel_extended if use_extended else penrose_kernel
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for node_data in known_nodes.values():
        # Handle both tuple and dict formats
        if isinstance(node_data, dict):
            nlat, nlon = node_data.get("coords", (0, 0))
        else:
            nlat, nlon = node_data
        
        theta = great_circle_angle(lat, lon, nlat, nlon)
        
        # Distance weight: exponential decay
        # Closer nodes have higher weight
        weight = math.exp(-theta / distance_scale)
        
        # Angular coherence
        coherence = kernel_func(theta)
        
        weighted_sum += weight * coherence
        weight_sum += weight
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def compute_H_multi(lat, lon, known_nodes=None):
    """
    Compute multiple H variants for comparison/ensemble.
    
    Returns:
        Dict with different H scoring methods
    """
    if known_nodes is None:
        known_nodes = KNOWN_NODES
    
    return {
        "H_basic": compute_H(lat, lon),
        "H_weighted": compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=False, known_nodes=known_nodes),
        "H_weighted_ext": compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True, known_nodes=known_nodes),
        "H_local": compute_H_weighted(lat, lon, distance_scale=15.0, use_extended=True, known_nodes=known_nodes),
        "H_global": compute_H_weighted(lat, lon, distance_scale=60.0, use_extended=True, known_nodes=known_nodes),
    }


def compute_H_combined(lat, lon, arch_weight=0.3, known_nodes=None):
    """
    Compute combined H score using both core nodes and archaeological sites.
    
    This combines:
    - Traditional GSN node-based H scoring (core sites)
    - Archaeological site proximity and alignment scoring (extended sites)
    
    Args:
        lat, lon: Coordinates to evaluate
        arch_weight: Weight for archaeological component (0-1)
        known_nodes: Override default KNOWN_NODES
    
    Returns:
        Combined H score
    """
    # Core H score from traditional nodes
    H_core = compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True, known_nodes=known_nodes)
    
    # Archaeological H score (if available)
    H_arch = 0.0
    try:
        from gsn_archaeology import compute_H_archaeological
        H_arch = compute_H_archaeological(lat, lon, distance_scale=30.0, use_weights=True)
    except ImportError:
        # Fall back to core-only if archaeology module not available
        arch_weight = 0.0
    
    # Combine scores
    core_weight = 1.0 - arch_weight
    return core_weight * H_core + arch_weight * H_arch


def compute_H_full(lat, lon, known_nodes=None, weights=None):
    """
    Compute comprehensive H score combining all available methods.
    
    Combines:
    - Penrose angle-based coherence (traditional)
    - Archaeological site coherence
    - Advanced geometric analysis (alignments, Voronoi, golden ratio)
    
    Args:
        lat, lon: Coordinates to evaluate
        known_nodes: Override default nodes
        weights: Optional dict of component weights
    
    Returns:
        Combined H score (0-1)
    """
    if weights is None:
        weights = {
            "penrose": 0.35,      # Traditional angle-based
            "archaeological": 0.15,  # Archaeological sites
            "geometric": 0.50,    # Advanced geometric (alignment, voronoi, golden, triangle)
        }
    
    # Penrose-weighted H
    H_penrose = compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True, known_nodes=known_nodes)
    
    # Archaeological H
    H_arch = 0.0
    try:
        from gsn_archaeology import compute_H_archaeological
        H_arch = compute_H_archaeological(lat, lon, distance_scale=30.0, use_weights=True)
    except ImportError:
        weights["archaeological"] = 0.0
    
    # Geometric H (alignments, Voronoi, golden ratio, triangulation)
    H_geo = 0.0
    try:
        from gsn_geometry import compute_H_geometric
        H_geo = compute_H_geometric(lat, lon, known_nodes)
    except ImportError:
        weights["geometric"] = 0.0
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return H_penrose
    
    # Weighted combination
    H = (weights["penrose"] * H_penrose +
         weights["archaeological"] * H_arch +
         weights["geometric"] * H_geo) / total_weight
    
    return H


def compute_H_breakdown(lat, lon, known_nodes=None):
    """
    Compute all H score components for detailed analysis.
    
    Returns:
        Dict with all individual H scores
    """
    result = {
        "H_basic": compute_H(lat, lon),
        "H_weighted": compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=False, known_nodes=known_nodes),
        "H_weighted_extended": compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True, known_nodes=known_nodes),
    }
    
    # Archaeological
    try:
        from gsn_archaeology import compute_H_archaeological
        result["H_archaeological"] = compute_H_archaeological(lat, lon)
    except ImportError:
        result["H_archaeological"] = None
    
    # Geometric components
    try:
        from gsn_geometry import compute_H_all_geometric
        geo_scores = compute_H_all_geometric(lat, lon, known_nodes)
        result.update(geo_scores)
    except ImportError:
        pass
    
    # Combined scores
    result["H_combined"] = compute_H_combined(lat, lon, known_nodes=known_nodes)
    result["H_full"] = compute_H_full(lat, lon, known_nodes=known_nodes)
    
    return result


def get_H_pattern_matches(lat, lon, known_nodes=None, top_n=5, sigma=6.0):
    """
    Get detailed breakdown of which angular patterns matched with which nodes.
    
    Returns list of matches sorted by score, each containing:
    - node_name: Name of the matching node
    - angle: The great circle angle to that node
    - matched_angle: The sacred angle it matched
    - match_score: How well it matched (0-1)
    - angle_type: Description of the matched angle
    
    Args:
        lat, lon: Coordinates to evaluate
        known_nodes: Dict of known nodes
        top_n: Number of top matches to return
        sigma: Gaussian width for angle matching
    
    Returns:
        List of match dicts sorted by score descending
    """
    if known_nodes is None:
        known_nodes = KNOWN_NODES
    
    matches = []
    
    for node_name, node_data in known_nodes.items():
        # Handle both tuple and dict formats
        if isinstance(node_data, dict):
            nlat, nlon = node_data.get("coords", (0, 0))
        else:
            nlat, nlon = node_data
        
        # Compute angle to this node
        theta = great_circle_angle(lat, lon, nlat, nlon)
        
        # Find best matching sacred angle
        best_score = 0.0
        best_angle = None
        
        for target in EXTENDED_ANGLES:
            diff = theta - target
            score = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
            if score > best_score:
                best_score = score
                best_angle = target
        
        if best_score > 0.1:  # Only include meaningful matches
            matches.append({
                "node_name": node_name.replace("_", " ").title(),
                "angle": round(theta, 1),
                "matched_angle": best_angle,
                "match_score": round(best_score, 2),
                "angle_type": ANGLE_DESCRIPTIONS.get(best_angle, "Sacred"),
            })
    
    # Sort by match score descending
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches[:top_n]


def compute_great_circle_alignments(lat, lon, known_nodes=None, tolerance=2.0):
    """
    Check if a location lies on great circles connecting pairs of known nodes.
    
    A high alignment score means the candidate lies on or near multiple 
    lines connecting ancient sites.
    
    Args:
        lat, lon: Candidate coordinates
        known_nodes: Dict of known nodes
        tolerance: Max angular distance from great circle (degrees)
    
    Returns:
        Dict with alignment_score and list of alignments
    """
    if known_nodes is None:
        known_nodes = KNOWN_NODES
    
    from itertools import combinations
    
    nodes_list = []
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            nlat, nlon = data.get("coords", (0, 0))
        else:
            nlat, nlon = data
        nodes_list.append({"name": name, "lat": nlat, "lon": nlon})
    
    alignments = []
    
    # Check each pair of nodes
    for node1, node2 in combinations(nodes_list, 2):
        # Distance from candidate to great circle through node1 and node2
        # Using cross-track distance formula
        dist = _cross_track_distance(
            lat, lon,
            node1["lat"], node1["lon"],
            node2["lat"], node2["lon"]
        )
        
        if dist is not None and dist < tolerance:
            alignments.append({
                "node1": node1["name"].replace("_", " ").title(),
                "node2": node2["name"].replace("_", " ").title(),
                "distance": round(dist, 2),
            })
    
    # Score based on number of alignments
    alignment_score = min(len(alignments) / 5.0, 1.0)  # Normalize to 0-1
    
    return {
        "alignment_score": round(alignment_score, 2),
        "num_alignments": len(alignments),
        "alignments": alignments[:5],  # Top 5
    }


def _cross_track_distance(lat, lon, lat1, lon1, lat2, lon2):
    """
    Angular distance from point (lat, lon) to great circle through (lat1, lon1) and (lat2, lon2).
    Returns distance in degrees.
    """
    try:
        # Convert to radians
        phi = math.radians(lat)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        lambda_c = math.radians(lon)
        lambda1 = math.radians(lon1)
        lambda2 = math.radians(lon2)
        
        # Angular distance from point 1 to candidate
        d13 = 2 * math.asin(math.sqrt(
            math.sin((phi - phi1)/2)**2 +
            math.cos(phi1) * math.cos(phi) * math.sin((lambda_c - lambda1)/2)**2
        ))
        
        # Initial bearing from point 1 to point 2
        theta12 = math.atan2(
            math.sin(lambda2 - lambda1) * math.cos(phi2),
            math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lambda2 - lambda1)
        )
        
        # Initial bearing from point 1 to candidate
        theta13 = math.atan2(
            math.sin(lambda_c - lambda1) * math.cos(phi),
            math.cos(phi1) * math.sin(phi) - math.sin(phi1) * math.cos(phi) * math.cos(lambda_c - lambda1)
        )
        
        # Cross-track angular distance
        dxt = math.asin(math.sin(d13) * math.sin(theta13 - theta12))
        
        return abs(math.degrees(dxt))
    except Exception:
        return None


def compute_symmetry_patterns(lat, lon, known_nodes=None, angle_tolerance=5.0):
    """
    Detect if candidate forms sacred geometric patterns with existing nodes.
    
    Checks for:
    - Equilateral triangles (60° angles)
    - Golden triangles (36°-72°-72°)
    - Pentagon patterns (108° interior)
    
    Args:
        lat, lon: Candidate coordinates
        known_nodes: Dict of known nodes
        angle_tolerance: Tolerance for angle matching (degrees)
    
    Returns:
        Dict with patterns found and symmetry score
    """
    if known_nodes is None:
        known_nodes = KNOWN_NODES
    
    from itertools import combinations
    
    nodes_list = []
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            nlat, nlon = data.get("coords", (0, 0))
        else:
            nlat, nlon = data
        nodes_list.append({"name": name, "lat": nlat, "lon": nlon})
    
    patterns = []
    
    # Check triangles formed with pairs of nodes
    for node1, node2 in combinations(nodes_list, 2):
        # Get the three angles of the triangle
        angle_at_candidate = great_circle_angle(node1["lat"], node1["lon"], node2["lat"], node2["lon"])
        angle_to_node1 = great_circle_angle(lat, lon, node1["lat"], node1["lon"])
        angle_to_node2 = great_circle_angle(lat, lon, node2["lat"], node2["lon"])
        
        # Check for equilateral (all ~60°)
        if (abs(angle_at_candidate - 60) < angle_tolerance and
            abs(angle_to_node1 - 60) < angle_tolerance and
            abs(angle_to_node2 - 60) < angle_tolerance):
            patterns.append({
                "type": "Equilateral Triangle",
                "nodes": [node1["name"].replace("_", " ").title(), 
                         node2["name"].replace("_", " ").title()],
                "quality": "strong",
            })
        
        # Check for golden triangle (36-72-72)
        angles = sorted([angle_at_candidate, angle_to_node1, angle_to_node2])
        if (abs(angles[0] - 36) < angle_tolerance and
            abs(angles[1] - 72) < angle_tolerance and
            abs(angles[2] - 72) < angle_tolerance):
            patterns.append({
                "type": "Golden Triangle",
                "nodes": [node1["name"].replace("_", " ").title(), 
                         node2["name"].replace("_", " ").title()],
                "quality": "strong",
            })
    
    # Symmetry score based on patterns found
    symmetry_score = min(len(patterns) / 3.0, 1.0)
    
    return {
        "symmetry_score": round(symmetry_score, 2),
        "num_patterns": len(patterns),
        "patterns": patterns[:3],  # Top 3
    }


# ---------------------------------------------------------
# Geophysical suitability layer (G)
# ---------------------------------------------------------
G_PARAMS = {
    "ga_scale": 30.0,
    "ct_mean": 35.0,
    "ct_std": 10.0,
    "L": 800.0,
    "w1": 1.0,
    "w2": 1.0,
    "w3": 1.0,
}


def describe_scoring_params():
    print(
        f"[INFO] G normalization: ga_scale={G_PARAMS['ga_scale']}, "
        f"ct_mean={G_PARAMS['ct_mean']}, ct_std={G_PARAMS['ct_std']}, L={G_PARAMS['L']} km"
    )
    print(
        f"[INFO] G weights: w1={G_PARAMS['w1']} (GA), w2={G_PARAMS['w2']} (CT), w3={G_PARAMS['w3']} (boundary)"
    )


def normalize_geophysics(ga, ct, dist_boundary_km, params=None):
    """
    Return normalized geophysical components.
    """
    p = params or G_PARAMS
    ga_norm = ga / p["ga_scale"]
    ct_norm = (ct - p["ct_mean"]) / p["ct_std"]
    tb_score = math.exp(- (dist_boundary_km ** 2) / (2.0 * p["L"] * p["L"]))
    return ga_norm, ct_norm, tb_score


def compute_G(ga, ct, dist_boundary_km, params=None):
    """
    Geophysical suitability G using configured parameters.
    """
    p = params or G_PARAMS
    ga_norm, ct_norm, tb_score = normalize_geophysics(ga, ct, dist_boundary_km, p)
    G = p["w1"] * ga_norm + p["w2"] * ct_norm + p["w3"] * tb_score
    return G, {"ga_norm": ga_norm, "ct_norm": ct_norm, "tb": tb_score}


def classify_F(F):
    """
    Simple classification of node strength based on F.
    """
    if F >= 2.0:
        return "STRONG candidate node"
    elif F >= 1.0:
        return "MODERATE candidate node"
    elif F >= 0.3:
        return "WEAK candidate node"
    else:
        return "UNLIKELY node (background)"


# ---------------------------------------------------------
# Region labeling (for nicer output)
# ---------------------------------------------------------
def approximate_region(lat, lon):
    """
    Very coarse, heuristic region labeling for human readability.
    Adjust as needed.
    """
    if -10 <= lat <= 10:
        if -50 <= lon <= 0:
            return "Equatorial South/East Atlantic"
        if 0 <= lon <= 60:
            return "Equatorial Africa / West Indian Ocean"
        if 60 <= lon <= 180:
            return "Equatorial West/Central Pacific"
        if -180 <= lon < -50:
            return "Equatorial Central/South Pacific"

    if lat > 20:
        if -80 <= lon <= 0:
            return "North Atlantic / W Europe–N Africa"
        if 0 <= lon <= 80:
            return "Southern Europe / Middle East"
        if 80 <= lon <= 180:
            return "North Pacific / Near Asia"
        if -180 <= lon < -80:
            return "North Pacific / Near America"

    if lat < -20:
        if -80 <= lon <= 0:
            return "South Atlantic off S America/Africa"
        if 0 <= lon <= 120:
            return "South Indian Ocean"
        if 120 <= lon <= 180:
            return "South Pacific near Zealandia"
        if -180 <= lon < -80:
            return "South Pacific off S America"

    return "Open Ocean / Broad Region"


# ---------------------------------------------------------
# Optional: real data integration via NetCDF
# ---------------------------------------------------------
_ga_ds = None
_ct_ds = None
_plate_boundaries = []  # list of polylines, each polyline is list[(lat, lon)]


def load_plate_boundaries():
    """
    Load plate boundary polylines from a KMZ/KML file.
    Approximates distance by nearest vertex (good enough for coarse scans).
    """
    if not os.path.exists(BOUNDARY_FILE):
        print(f"[WARN] Plate boundary file not found: {BOUNDARY_FILE}")
        return []

    try:
        with zipfile.ZipFile(BOUNDARY_FILE, "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                print(f"[WARN] No KML found inside {BOUNDARY_FILE}")
                return []
            with zf.open(kml_names[0]) as f:
                tree = ET.parse(f)

        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        polylines = []
        for ls in tree.iterfind(".//kml:LineString", ns):
            coord_elem = ls.find("kml:coordinates", ns)
            if coord_elem is None or not coord_elem.text:
                continue
            pts = []
            for token in coord_elem.text.strip().split():
                parts = token.split(",")
                if len(parts) >= 2:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    pts.append((lat, lon))
            if len(pts) >= 2:
                polylines.append(pts)

        if not polylines:
            print(f"[WARN] No polylines parsed from {BOUNDARY_FILE}")
        else:
            print(f"[INFO] Loaded {len(polylines)} plate boundary polylines from KMZ.")
        return polylines
    except Exception as e:
        print(f"[WARN] Failed to load plate boundaries: {e}")
        return []

def try_load_datasets():
    global _ga_ds, _ct_ds
    if not USE_XARRAY:
        return False

    missing = []
    if not os.path.exists(GRAVITY_FILE):
        missing.append(GRAVITY_FILE)
    if not os.path.exists(CRUST_FILE):
        missing.append(CRUST_FILE)
    if missing:
        print(f"[ERROR] Missing required NetCDF files: {', '.join(missing)}")
        return False

    try:
        import xarray as xr
        _ga_ds = xr.open_dataset(GRAVITY_FILE)
        _ct_ds = xr.open_dataset(CRUST_FILE)
        return True
    except Exception as e:
        print(f"[WARN] Could not load NetCDF datasets: {e}")
        _ga_ds = None
        _ct_ds = None
        return False


def get_value_from_grid(lat, lon, data_var):
    """
    Get value from a gridded DataArray at given lat, lon using nearest-neighbor.
    Assumes coordinates are 'lat' and 'lon'.
    """
    point = data_var.sel(lat=lat, lon=lon, method="nearest")
    return float(point.values)


def approximate_distance_to_boundary(lat, lon):
    """
    Distance (km) to nearest plate boundary segment (great-circle).
    Uses loaded KMZ boundaries; falls back to 500 km if none loaded.
    """
    if not _plate_boundaries:
        return 500.0

    def to_vec(phi_deg, lam_deg):
        phi = deg2rad(phi_deg)
        lam = deg2rad(lam_deg)
        return (
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        )

    p = to_vec(lat, lon)

    def angle(u, v):
        dot = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def segment_min_angle(a, b):
        """
        Minimum angular distance between point p and great-circle segment ab.
        """
        # Great-circle normal
        n = (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )
        n_norm = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        if n_norm < 1e-12:
            return angle(p, a)  # degenerate; fallback to endpoint
        n_hat = (n[0]/n_norm, n[1]/n_norm, n[2]/n_norm)

        # Angular distance from p to great circle
        cross_pn = (
            p[1]*n_hat[2] - p[2]*n_hat[1],
            p[2]*n_hat[0] - p[0]*n_hat[2],
            p[0]*n_hat[1] - p[1]*n_hat[0],
        )
        cross_norm = math.sqrt(cross_pn[0]**2 + cross_pn[1]**2 + cross_pn[2]**2)
        if cross_norm < 1e-12:
            dist_gc = 0.0
        else:
            dist_gc = math.asin(min(1.0, max(-1.0, cross_norm)))

        # Check if projection lies between endpoints along the great-circle arc.
        dist_ab = angle(a, b)
        dist_ap = angle(a, p)
        dist_pb = angle(p, b)
        if dist_ap + dist_pb - dist_ab <= 1e-6:
            return dist_gc  # projection falls on segment
        else:
            return min(dist_ap, dist_pb)  # nearest endpoint

    min_ang = None
    for poly in _plate_boundaries:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            a = to_vec(poly[i][0], poly[i][1])
            b = to_vec(poly[i+1][0], poly[i+1][1])
            ang = segment_min_angle(a, b)
            if (min_ang is None) or (ang < min_ang):
                min_ang = ang

    if min_ang is None:
        return 500.0
    return min_ang * 6371.0  # radians * Earth radius (km)


def get_geophysical_inputs(lat, lon):
    """
    Return (ga, ct, dist_boundary_km) for a given lat, lon using NetCDF if available.
    """
    if _ga_ds is None or _ct_ds is None:
        raise RuntimeError("NetCDF datasets are not loaded. Provide real data files before running.")

    ga_var = _ga_ds[GA_VAR_NAME]
    ct_var = _ct_ds[CT_VAR_NAME]

    ga = get_value_from_grid(lat, lon, ga_var)
    ct = get_value_from_grid(lat, lon, ct_var)
    dist_boundary_km = approximate_distance_to_boundary(lat, lon)

    return ga, ct, dist_boundary_km


# ---------------------------------------------------------
# Vectorized grid computations for full-resolution scans
# ---------------------------------------------------------
_boundary_distance_grid = None  # cached distance grid (km)
_boundary_grid_lats = None
_boundary_grid_lons = None


def compute_boundary_distance_grid(lats, lons):
    """
    Compute distance to nearest plate boundary for an entire lat/lon grid.
    Uses scipy.ndimage.distance_transform_edt for fast computation.
    
    Args:
        lats: 1D array of latitudes
        lons: 1D array of longitudes
    
    Returns:
        2D array (nlat, nlon) of distances in km to nearest boundary
    """
    global _boundary_distance_grid, _boundary_grid_lats, _boundary_grid_lons
    
    if not USE_SCIPY:
        raise RuntimeError("scipy is required for vectorized boundary distance. Install with: pip install scipy")
    
    if not _plate_boundaries:
        # No boundaries loaded - return default distance
        print("[WARN] No plate boundaries loaded, using default 500 km distance")
        return np.full((len(lats), len(lons)), 500.0)
    
    nlat, nlon = len(lats), len(lons)
    print(f"[INFO] Computing boundary distance grid ({nlat} x {nlon} = {nlat * nlon:,} points)...")
    
    # Create a binary mask of boundary pixels
    # Map lat/lon to pixel indices
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_res = (lat_max - lat_min) / (nlat - 1) if nlat > 1 else 1.0
    lon_res = (lon_max - lon_min) / (nlon - 1) if nlon > 1 else 1.0
    
    # Initialize boundary mask (True = boundary pixel)
    boundary_mask = np.zeros((nlat, nlon), dtype=bool)
    
    # Rasterize each boundary polyline using Bresenham-like line drawing
    def latlon_to_pixel(lat, lon):
        i = int(round((lat - lat_min) / lat_res)) if lat_res > 0 else 0
        j = int(round((lon - lon_min) / lon_res)) if lon_res > 0 else 0
        i = max(0, min(nlat - 1, i))
        j = max(0, min(nlon - 1, j))
        return i, j
    
    for poly in _plate_boundaries:
        if len(poly) < 2:
            continue
        for k in range(len(poly) - 1):
            lat1, lon1 = poly[k]
            lat2, lon2 = poly[k + 1]
            i1, j1 = latlon_to_pixel(lat1, lon1)
            i2, j2 = latlon_to_pixel(lat2, lon2)
            
            # Bresenham's line algorithm
            di = abs(i2 - i1)
            dj = abs(j2 - j1)
            si = 1 if i1 < i2 else -1
            sj = 1 if j1 < j2 else -1
            err = di - dj
            
            i, j = i1, j1
            while True:
                boundary_mask[i, j] = True
                if i == i2 and j == j2:
                    break
                e2 = 2 * err
                if e2 > -dj:
                    err -= dj
                    i += si
                if e2 < di:
                    err += di
                    j += sj
    
    n_boundary_pixels = boundary_mask.sum()
    print(f"[INFO] Rasterized {n_boundary_pixels:,} boundary pixels")
    
    # Compute Euclidean distance transform (in pixel units)
    # distance_transform_edt computes distance from False pixels to nearest True pixel
    # We want distance from all pixels to nearest boundary, so invert the mask
    pixel_distance = ndimage.distance_transform_edt(~boundary_mask)
    
    # Convert pixel distance to km
    # Account for latitude-dependent longitude scaling
    lat_grid = lats[:, np.newaxis]  # (nlat, 1)
    
    # Average pixel size in km at each latitude
    # Latitude degree ~ 111 km everywhere
    # Longitude degree ~ 111 * cos(lat) km
    lat_km_per_pixel = lat_res * 111.0
    lon_km_per_pixel = lon_res * 111.0 * np.cos(np.radians(lat_grid))
    
    # Approximate average pixel size (geometric mean)
    avg_km_per_pixel = np.sqrt(lat_km_per_pixel * lon_km_per_pixel)
    
    distance_km = pixel_distance * avg_km_per_pixel
    
    # Cache the result
    _boundary_distance_grid = distance_km
    _boundary_grid_lats = lats
    _boundary_grid_lons = lons
    
    print(f"[INFO] Boundary distance grid complete: min={distance_km.min():.1f} km, max={distance_km.max():.1f} km")
    
    return distance_km


def compute_H_grid(lats, lons, sigma=6.0):
    """
    Compute geometric coherence H for an entire lat/lon grid using vectorized operations.
    
    H measures Penrose-like angular coherence with known GSN nodes.
    
    Args:
        lats: 1D array of latitudes (degrees)
        lons: 1D array of longitudes (degrees)
        sigma: Gaussian kernel width for Penrose angle matching (degrees)
    
    Returns:
        2D array (nlat, nlon) of H values
    """
    nlat, nlon = len(lats), len(lons)
    print(f"[INFO] Computing H grid ({nlat} x {nlon} = {nlat * nlon:,} points)...")
    
    # Create 2D coordinate grids
    lat_grid = np.radians(lats[:, np.newaxis])  # (nlat, 1)
    lon_grid = np.radians(lons[np.newaxis, :])  # (1, nlon)
    
    # Pre-compute sin/cos for all grid points
    sin_lat = np.sin(lat_grid)  # (nlat, 1)
    cos_lat = np.cos(lat_grid)  # (nlat, 1)
    
    # Known node coordinates
    node_coords = np.array(list(KNOWN_NODES.values()))  # (n_nodes, 2)
    node_lats = np.radians(node_coords[:, 0])  # (n_nodes,)
    node_lons = np.radians(node_coords[:, 1])  # (n_nodes,)
    
    sin_node_lat = np.sin(node_lats)  # (n_nodes,)
    cos_node_lat = np.cos(node_lats)  # (n_nodes,)
    
    # Penrose target angles in radians for comparison (but we work in degrees for kernel)
    penrose_angles = np.array(PENROSE_ANGLES)  # (n_angles,)
    
    # Accumulate H scores across all nodes
    H_sum = np.zeros((nlat, nlon), dtype=np.float64)
    
    for i_node in range(len(node_coords)):
        # Great-circle angle from all grid points to this node
        # cos(angle) = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon)
        dlon = lon_grid - node_lons[i_node]  # (1, nlon)
        
        cos_angle = (
            sin_lat * sin_node_lat[i_node] +
            cos_lat * cos_node_lat[i_node] * np.cos(dlon)
        )  # (nlat, nlon)
        
        # Clamp to [-1, 1] for numerical stability
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Angle in degrees
        theta_deg = np.degrees(np.arccos(cos_angle))  # (nlat, nlon)
        
        # Penrose kernel: max over all target angles
        # For each target angle, compute exp(-(theta - target)^2 / (2*sigma^2))
        # Then take the max across targets
        kernel_vals = np.zeros((nlat, nlon), dtype=np.float64)
        for target in penrose_angles:
            diff = theta_deg - target
            val = np.exp(-(diff * diff) / (2.0 * sigma * sigma))
            kernel_vals = np.maximum(kernel_vals, val)
        
        H_sum += kernel_vals
    
    # Average across nodes
    H_grid = H_sum / len(node_coords) if len(node_coords) > 0 else H_sum
    
    print(f"[INFO] H grid complete: min={H_grid.min():.4f}, max={H_grid.max():.4f}")
    
    return H_grid


def compute_G_grid(lats, lons, boundary_distance_grid=None, params=None):
    """
    Compute geophysical suitability G for an entire lat/lon grid.
    
    G combines gravity anomaly, crustal thickness, and distance to plate boundaries.
    
    Args:
        lats: 1D array of latitudes (degrees)
        lons: 1D array of longitudes (degrees)
        boundary_distance_grid: Pre-computed boundary distance grid (nlat, nlon) in km.
                                If None, will be computed.
        params: Dict of G parameters (ga_scale, ct_mean, ct_std, L, w1, w2, w3)
    
    Returns:
        Tuple of (G_grid, components_dict) where components_dict has ga_norm, ct_norm, tb grids
    """
    if _ga_ds is None or _ct_ds is None:
        raise RuntimeError("NetCDF datasets are required for G grid computation.")
    
    p = params or G_PARAMS
    nlat, nlon = len(lats), len(lons)
    print(f"[INFO] Computing G grid ({nlat} x {nlon} = {nlat * nlon:,} points)...")
    
    # Get gravity anomaly grid
    ga_var = _ga_ds[GA_VAR_NAME]
    print(f"[INFO] Reading gravity anomaly grid from {GRAVITY_FILE}...")
    
    # Select the region we need and load into memory
    # Use xarray's vectorized selection
    ga_grid = ga_var.sel(lat=lats, lon=lons, method="nearest").values
    if ga_grid.ndim == 1:
        # Handle case where we get 1D output
        ga_grid = ga_grid.reshape(nlat, nlon)
    print(f"[INFO] GA grid loaded: shape={ga_grid.shape}, min={np.nanmin(ga_grid):.2f}, max={np.nanmax(ga_grid):.2f}")
    
    # Get crustal thickness grid  
    ct_var = _ct_ds[CT_VAR_NAME]
    print(f"[INFO] Reading crustal thickness grid from {CRUST_FILE}...")
    ct_grid = ct_var.sel(lat=lats, lon=lons, method="nearest").values
    if ct_grid.ndim == 1:
        ct_grid = ct_grid.reshape(nlat, nlon)
    print(f"[INFO] CT grid loaded: shape={ct_grid.shape}, min={np.nanmin(ct_grid):.2f}, max={np.nanmax(ct_grid):.2f}")
    
    # Get or compute boundary distance grid
    if boundary_distance_grid is None:
        boundary_distance_grid = compute_boundary_distance_grid(lats, lons)
    
    # Handle NaN values
    ga_grid = np.nan_to_num(ga_grid, nan=0.0)
    ct_grid = np.nan_to_num(ct_grid, nan=p["ct_mean"])
    
    # Compute normalized components
    ga_norm = ga_grid / p["ga_scale"]
    ct_norm = (ct_grid - p["ct_mean"]) / p["ct_std"]
    tb_score = np.exp(-(boundary_distance_grid ** 2) / (2.0 * p["L"] * p["L"]))
    
    # Compute G
    G_grid = p["w1"] * ga_norm + p["w2"] * ct_norm + p["w3"] * tb_score
    
    print(f"[INFO] G grid complete: min={G_grid.min():.4f}, max={G_grid.max():.4f}")
    
    components = {
        "ga_norm": ga_norm,
        "ct_norm": ct_norm,
        "tb": tb_score,
        "ga_raw": ga_grid,
        "ct_raw": ct_grid,
    }
    
    return G_grid, components


def compute_F_grid(lats, lons, alpha=1.0, beta=1.0, gamma=0.0, g_params=None, include_astronomy=False):
    """
    Compute the full F grid (Node Index) at native resolution.
    
    F = alpha * G + beta * H + gamma * A
    
    Args:
        lats: 1D array of latitudes (degrees)
        lons: 1D array of longitudes (degrees)
        alpha: Weight for geophysical component G
        beta: Weight for geometric component H
        gamma: Weight for astronomical component A (default 0 = disabled)
        g_params: Optional dict of G parameters
        include_astronomy: Whether to compute A scores (requires gamma > 0)
    
    Returns:
        Dict containing:
            - F: 2D array (nlat, nlon) of F values
            - G: 2D array of G values
            - H: 2D array of H values
            - A: 2D array of A values (if gamma > 0)
            - lats: latitude array
            - lons: longitude array
            - components: dict of normalized component grids
    """
    nlat, nlon = len(lats), len(lons)
    total_points = nlat * nlon
    print(f"\n[INFO] Computing full-resolution F grid...")
    print(f"[INFO] Grid dimensions: {nlat} lat x {nlon} lon = {total_points:,} points")
    print(f"[INFO] Weights: alpha={alpha} (G), beta={beta} (H), gamma={gamma} (A)")
    
    import time
    start_time = time.time()
    
    # Compute boundary distance grid first (used by G)
    boundary_dist = compute_boundary_distance_grid(lats, lons)
    
    # Compute H grid
    H_grid = compute_H_grid(lats, lons)
    
    # Compute G grid
    G_grid, components = compute_G_grid(lats, lons, boundary_distance_grid=boundary_dist, params=g_params)
    
    # Compute A grid (astronomical) if gamma > 0
    A_grid = None
    if gamma > 0 and include_astronomy:
        try:
            from gsn_astronomy import compute_A_score, get_constellation_visibility
            print("[INFO] Computing astronomical (A) scores...")
            
            # For grid computation, use simplified visibility-based A score
            # (Pattern matching requires specific node sets, not suitable for grids)
            A_grid = np.zeros((nlat, nlon))
            
            # Sample visibility at grid points (subsample for performance)
            sample_step = max(1, nlat // 20)
            for i in range(0, nlat, sample_step):
                visibility = get_constellation_visibility(lats[i], lons[nlon // 2])
                sacred_score = np.mean([visibility.get(c, 0) for c in 
                                       ["orion", "pleiades", "ursa_major"]])
                # Apply to nearby latitudes
                for di in range(sample_step):
                    if i + di < nlat:
                        A_grid[i + di, :] = sacred_score
            
            print(f"[INFO] A stats: min={A_grid.min():.4f}, max={A_grid.max():.4f}")
        except ImportError:
            print("[WARN] gsn_astronomy not available, A scores set to 0")
            A_grid = np.zeros((nlat, nlon))
    
    # Compute F
    F_grid = alpha * G_grid + beta * H_grid
    if A_grid is not None and gamma > 0:
        F_grid = F_grid + gamma * A_grid
    
    elapsed = time.time() - start_time
    print(f"\n[INFO] F grid computation complete in {elapsed:.1f} seconds")
    print(f"[INFO] F stats: min={F_grid.min():.4f}, max={F_grid.max():.4f}, "
          f"median={np.median(F_grid):.4f}")
    
    result = {
        "F": F_grid,
        "G": G_grid,
        "H": H_grid,
        "lats": lats,
        "lons": lons,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "components": components,
        "boundary_distance": boundary_dist,
    }
    
    if A_grid is not None:
        result["A"] = A_grid
    
    return result


# ---------------------------------------------------------
# ML-Based F Grid Computation
# ---------------------------------------------------------

# Cached ML scorer instance
_ml_scorer = None
_ml_model_path = "gsn_ml_grid_scorer.pth"


def get_ml_scorer(model_path=None, force_reload=False):
    """
    Get or create cached ML scorer instance.
    
    Args:
        model_path: Path to trained model file
        force_reload: Force reloading from disk
    
    Returns:
        MLGridScorer instance or None if not available
    """
    global _ml_scorer, _ml_model_path
    
    if model_path is not None:
        _ml_model_path = model_path
    
    if _ml_scorer is not None and not force_reload:
        return _ml_scorer
    
    try:
        from gsn_ml_grid_scorer import MLGridScorer
        import os
        
        if os.path.exists(_ml_model_path):
            _ml_scorer = MLGridScorer(_ml_model_path)
            print(f"[INFO] Loaded ML scorer from {_ml_model_path}")
            return _ml_scorer
        else:
            print(f"[WARN] ML model not found at {_ml_model_path}")
            return None
    except ImportError:
        print("[WARN] gsn_ml_grid_scorer module not available")
        return None


def train_ml_scorer(known_nodes=None, config=None, save_path=None, verbose=True):
    """
    Train a new ML scorer from known nodes.
    
    Args:
        known_nodes: Dict of known nodes (default: KNOWN_NODES)
        config: TrainingConfig (uses defaults if None)
        save_path: Path to save trained model
        verbose: Print progress
    
    Returns:
        Trained MLGridScorer instance
    """
    global _ml_scorer
    
    try:
        from gsn_ml_grid_scorer import (
            MLGridScorer, TrainingConfig, extract_grid_features
        )
    except ImportError:
        print("[ERROR] gsn_ml_grid_scorer module not available")
        return None
    
    if known_nodes is None:
        try:
            from known_nodes_extended import KNOWN_NODES_EXTENDED
            known_nodes = KNOWN_NODES_EXTENDED
        except ImportError:
            known_nodes = KNOWN_NODES
    
    if config is None:
        config = TrainingConfig()
    
    if save_path is None:
        save_path = _ml_model_path
    
    # Extract coordinates
    coords = []
    for name, data in known_nodes.items():
        if isinstance(data, dict):
            lat, lon = data.get("coords", (0, 0))
        else:
            lat, lon = data
        coords.append((lat, lon))
    
    print(f"[INFO] Training ML scorer from {len(coords)} known nodes...")
    
    # Extract features for each node using full geophysical data if available
    features = []
    for lat, lon in coords:
        # Try to get real geophysical values
        try:
            ga, ct, dist_km = get_geophysical_inputs(lat, lon)
            ga_norm = ga / G_PARAMS["ga_scale"]
            ct_norm = (ct - G_PARAMS["ct_mean"]) / G_PARAMS["ct_std"]
            tb_score = math.exp(-(dist_km ** 2) / (2.0 * G_PARAMS["L"] ** 2))
            H_val = compute_H(lat, lon)
        except Exception:
            # Use defaults if real data not available
            ga_norm = 0.5
            ct_norm = 0.0
            tb_score = 0.5
            H_val = compute_H(lat, lon)
        
        feat = extract_grid_features(
            lat, lon,
            G_val=ga_norm + ct_norm + tb_score,
            H_val=H_val,
            ga_norm=ga_norm,
            ct_norm=ct_norm,
            boundary_score=tb_score,
            known_node_coords=coords
        )
        features.append(feat)
    
    import numpy as np
    features = np.array(features, dtype=np.float32)
    
    # Train
    scorer = MLGridScorer()
    result = scorer.train(coords, features, config, verbose)
    
    # Save
    scorer.save(save_path)
    
    # Update cached scorer
    _ml_scorer = scorer
    
    return scorer


def compute_F_grid_ml(lats, lons, model_path=None, g_params=None, fallback_linear=True):
    """
    Compute F grid using ML model instead of linear combination.
    
    This is a drop-in replacement for compute_F_grid() that uses
    a trained neural network to score grid points.
    
    Args:
        lats: 1D array of latitudes (degrees)
        lons: 1D array of longitudes (degrees)
        model_path: Path to trained model (uses default if None)
        g_params: Optional dict of G parameters
        fallback_linear: If True, fall back to linear formula if ML unavailable
    
    Returns:
        Dict containing:
            - F: 2D array (nlat, nlon) of ML-predicted F values
            - G: 2D array of G values
            - H: 2D array of H values
            - lats: latitude array
            - lons: longitude array
            - components: dict of normalized component grids
            - method: "ml" or "linear" indicating which method was used
    """
    import time
    
    nlat, nlon = len(lats), len(lons)
    total_points = nlat * nlon
    print(f"\n[INFO] Computing ML-based F grid...")
    print(f"[INFO] Grid dimensions: {nlat} lat x {nlon} lon = {total_points:,} points")
    
    start_time = time.time()
    
    # Get ML scorer
    scorer = get_ml_scorer(model_path)
    
    if scorer is None:
        if fallback_linear:
            print("[WARN] ML scorer not available, falling back to linear formula")
            result = compute_F_grid(lats, lons, alpha=1.0, beta=1.0, g_params=g_params)
            result["method"] = "linear"
            return result
        else:
            raise RuntimeError("ML scorer not available and fallback disabled")
    
    # Compute base grids (G, H, components)
    boundary_dist = compute_boundary_distance_grid(lats, lons)
    H_grid = compute_H_grid(lats, lons)
    G_grid, components = compute_G_grid(lats, lons, boundary_distance_grid=boundary_dist, params=g_params)
    
    # Use ML scorer to predict F
    print("[INFO] Predicting F values with ML model...")
    F_grid = scorer.predict_grid(lats, lons, G_grid, H_grid, components)
    
    # Scale to similar range as linear F for compatibility
    # ML outputs 0-1 probability; scale to typical F range
    F_grid = F_grid * 4.0  # Scale to approximately 0-4 range
    
    elapsed = time.time() - start_time
    print(f"\n[INFO] ML F grid computation complete in {elapsed:.1f} seconds")
    print(f"[INFO] F stats: min={F_grid.min():.4f}, max={F_grid.max():.4f}, "
          f"median={np.median(F_grid):.4f}")
    
    result = {
        "F": F_grid,
        "G": G_grid,
        "H": H_grid,
        "lats": lats,
        "lons": lons,
        "components": components,
        "boundary_distance": boundary_dist,
        "method": "ml",
    }
    
    return result


def extract_top_candidates_from_grid(result, top_n=50, min_sep_deg=2.0):
    """
    Extract top-N F candidates from a full grid result with spatial separation.
    
    Args:
        result: Dict from compute_F_grid()
        top_n: Number of top candidates to extract
        min_sep_deg: Minimum separation between candidates in degrees
    
    Returns:
        List of dicts with candidate information
    """
    F_grid = result["F"]
    lats = result["lats"]
    lons = result["lons"]
    G_grid = result["G"]
    H_grid = result["H"]
    
    # Flatten and sort by F
    nlat, nlon = F_grid.shape
    flat_indices = np.argsort(F_grid.ravel())[::-1]  # descending
    
    candidates = []
    for flat_idx in flat_indices:
        if len(candidates) >= top_n:
            break
        
        i = flat_idx // nlon
        j = flat_idx % nlon
        lat = float(lats[i])
        lon = float(lons[j])
        F_val = float(F_grid[i, j])
        
        # Check separation from already selected candidates
        too_close = False
        for cand in candidates:
            d = great_circle_angle(lat, lon, cand["lat"], cand["lon"])
            if d < min_sep_deg:
                too_close = True
                break
        
        if not too_close:
            candidates.append({
                "F": F_val,
                "G": float(G_grid[i, j]),
                "H": float(H_grid[i, j]),
                "lat": lat,
                "lon": lon,
                "region": approximate_region(lat, lon),
                "grid_i": i,
                "grid_j": j,
            })
    
    return candidates


# ---------------------------------------------------------
# Mode 1: Single-location evaluation
# ---------------------------------------------------------
def mode_single_location():
    print("\n=== Mode 1: Evaluate Single Location (G, H, F) ===\n")
    lat = float(input("Enter candidate latitude   (degrees, -90 to 90)   : "))
    lon = float(input("Enter candidate longitude  (degrees, -180 to 180) : "))

    if _ga_ds is None or _ct_ds is None:
        print("\n[ERROR] NetCDF datasets are required for Mode 1 in real-data-only mode.")
        print("        Please provide the files and restart.")
        return

    ga, ct, dist_boundary_km = get_geophysical_inputs(lat, lon)
    print(f"\n[INFO] Using NetCDF-based GA={ga:.2f} mGal, CT={ct:.2f} km, dist_boundary~{dist_boundary_km:.1f} km\n")

    print("\nOptional: weight tuning for G components (gravity, crust, boundary).")
    print("Press Enter for all defaults (1.0, 1.0, 1.0).")
    try:
        w1_str = input("Weight for gravity anomaly term w1 [default 1.0]: ")
        w1 = float(w1_str) if w1_str.strip() else 1.0
    except ValueError:
        w1 = 1.0

    try:
        w2_str = input("Weight for crustal thickness term w2 [default 1.0]: ")
        w2 = float(w2_str) if w2_str.strip() else 1.0
    except ValueError:
        w2 = 1.0

    try:
        w3_str = input("Weight for tectonic boundary term w3 [default 1.0]: ")
        w3 = float(w3_str) if w3_str.strip() else 1.0
    except ValueError:
        w3 = 1.0

    g_params = {**G_PARAMS, "w1": w1, "w2": w2, "w3": w3}
    G, g_comp = compute_G(ga, ct, dist_boundary_km, params=g_params)
    H = compute_H(lat, lon)

    print("\nCombine geophysical (G) and geometric (H) components into Node Index F.")
    try:
        alpha_str = input("Alpha (weight on G) [default 1.0]: ")
        alpha = float(alpha_str) if alpha_str.strip() else 1.0
    except ValueError:
        alpha = 1.0

    try:
        beta_str = input("Beta (weight on H) [default 1.0]: ")
        beta = float(beta_str) if beta_str.strip() else 1.0
    except ValueError:
        beta = 1.0

    F = alpha * G + beta * H

    region = approximate_region(lat, lon)

    print("\n=== Node Prediction Results ===")
    print(f"Location:    lat = {lat:.2f}°, lon = {lon:.2f}°  ({region})")
    print(f"Geophysical suitability G  = {G:.3f}")
    print(f"Geometric coherence H      = {H:.3f}")
    print(f"Composite Node Index F     = {F:.3f}")
    print(f"Classification: {classify_F(F)}\n")


# ---------------------------------------------------------
# Mode 2: Global scan for top-N H peaks (geometry-only)
# ---------------------------------------------------------
def mode_global_scan_H():
    print("\n=== Mode 2: Global Scan for Geometric (H) Peaks ===\n")
    print("This scans the globe on a coarse grid using only geometric coherence H.")
    print("High-H locations are Penrose-consistent candidates for undiscovered nodes.\n")

    try:
        step_str = input("Grid step in degrees (e.g. 5, 10) [default 10]: ")
        step_deg = float(step_str) if step_str.strip() else 10.0
    except ValueError:
        step_deg = 10.0

    try:
        N_str = input("How many top candidate locations to list? [default 10]: ")
        top_N = int(N_str) if N_str.strip() else 10
    except ValueError:
        top_N = 10

    try:
        min_sep_str = input("Minimum separation between candidates (deg) [default 10]: ")
        min_sep = float(min_sep_str) if min_sep_str.strip() else 10.0
    except ValueError:
        min_sep = 10.0

    print(f"\nScanning globe with step {step_deg}°; this may take a moment...\n")

    candidates = []
    lats = [lat for lat in frange(-90.0 + step_deg/2.0, 90.0 - step_deg/2.0, step_deg)]
    lons = [lon for lon in frange(-180.0 + step_deg/2.0, 180.0 - step_deg/2.0, step_deg)]

    for lat in lats:
        for lon in lons:
            H = compute_H(lat, lon)
            candidates.append((H, lat, lon))

    candidates.sort(reverse=True, key=lambda x: x[0])

    selected = []
    for H_val, lat, lon in candidates:
        if len(selected) >= top_N:
            break
        too_close = False
        for (_, s_lat, s_lon) in selected:
            d = great_circle_angle(lat, lon, s_lat, s_lon)
            if d < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append((H_val, lat, lon))

    print("=== Top Geometric H Peaks (Penrose-Consistent Candidates) ===\n")
    print("Rank |   H-score |   Latitude |  Longitude | Region")
    print("-----+-----------+-----------+-----------+-------------------------------")
    for rank, (H_val, lat, lon) in enumerate(selected, start=1):
        region = approximate_region(lat, lon)
        print(f"{rank:4d} | {H_val:9.3f} | {lat:+9.2f}° | {lon:+10.2f}° | {region}")

    print("\nNote: These are geometry-only predictions based on Penrose-like angles to known nodes.")
    print("      Next step is to cross-check with gravity, crust, and tectonic data.\n")


# ---------------------------------------------------------
# Shared helper: compute top F candidates (G+H) for scans/maps
# ---------------------------------------------------------
def compute_top_F_candidates(
    step_deg=10.0,
    top_N=10,
    min_sep=10.0,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
    lat_min=None,
    lat_max=None,
    lon_min=None,
    lon_max=None,
    return_stats=False,
    include_astronomy=False,
):
    if _ga_ds is None or _ct_ds is None:
        raise RuntimeError("NetCDF datasets are required for F-mode computations.")

    # Try to import astronomy module if gamma > 0
    compute_A = None
    if gamma > 0 and include_astronomy:
        try:
            from gsn_astronomy import compute_A_score
            compute_A = compute_A_score
        except ImportError:
            print("[WARN] gsn_astronomy not available, A scores disabled")

    lat_lo = -90.0 + step_deg / 2.0 if lat_min is None else max(-90.0, lat_min)
    lat_hi = 90.0 - step_deg / 2.0 if lat_max is None else min(90.0, lat_max)
    lon_lo = -180.0 + step_deg / 2.0 if lon_min is None else max(-180.0, lon_min)
    lon_hi = 180.0 - step_deg / 2.0 if lon_max is None else min(180.0, lon_max)

    lats = [lat for lat in frange(lat_lo, lat_hi, step_deg)]
    lons = [lon for lon in frange(lon_lo, lon_hi, step_deg)]

    raw_candidates = []
    for lat in lats:
        for lon in lons:
            H = compute_H(lat, lon)
            ga, ct, dist_boundary_km = get_geophysical_inputs(lat, lon)
            G, comps = compute_G(ga, ct, dist_boundary_km)
            
            # Compute A score if enabled
            A = 0.0
            if compute_A is not None:
                A = compute_A(lat, lon)
            
            F = alpha * G + beta * H + gamma * A
            raw_candidates.append((F, G, H, A, lat, lon, ga, ct, dist_boundary_km, comps))

    raw_candidates.sort(reverse=True, key=lambda x: x[0])

    selected = []
    for entry in raw_candidates:
        F_val, G_val, H_val, A_val, lat, lon, ga, ct, dist_km, comps = entry
        if len(selected) >= top_N:
            break
        too_close = False
        for cand in selected:
            d = great_circle_angle(lat, lon, cand["lat"], cand["lon"])
            if d < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append(
                {
                    "F": F_val,
                    "G": G_val,
                    "H": H_val,
                    "A": A_val,
                    "lat": lat,
                    "lon": lon,
                    "region": approximate_region(lat, lon),
                    "ga": ga,
                    "ct": ct,
                    "dist_boundary_km": dist_km,
                    "ga_norm": comps["ga_norm"],
                    "ct_norm": comps["ct_norm"],
                    "tb": comps["tb"],
                }
            )

    if not return_stats:
        return selected

    F_all = np.array([x[0] for x in raw_candidates])
    stats = {
        "count": int(len(raw_candidates)),
        "min": float(F_all.min()) if len(F_all) else float("nan"),
        "max": float(F_all.max()) if len(F_all) else float("nan"),
        "median": float(np.median(F_all)) if len(F_all) else float("nan"),
        "p90": float(np.percentile(F_all, 90)) if len(F_all) else float("nan"),
        "p95": float(np.percentile(F_all, 95)) if len(F_all) else float("nan"),
    }
    return selected, stats


# ---------------------------------------------------------
# Mode 3: Global scan + G and F using real data (if available)
# ---------------------------------------------------------
def mode_global_scan_F():
    if not USE_XARRAY:
        print("\n[ERROR] xarray is not available. Install it with `pip install xarray netCDF4`.")
        return

    if _ga_ds is None or _ct_ds is None:
        print("\n[ERROR] NetCDF datasets are required for Mode 3. Provide real data files and restart.\n")
        return

    print("\n=== Mode 3: Global Scan with F (G+H) using real data ===\n")
    describe_scoring_params()

    try:
        step_str = input("Grid step in degrees (e.g. 10, 15) [default 10]: ")
        step_deg = float(step_str) if step_str.strip() else 10.0
    except ValueError:
        step_deg = 10.0

    try:
        N_str = input("How many top candidate locations to list? [default 10]: ")
        top_N = int(N_str) if N_str.strip() else 10
    except ValueError:
        top_N = 10

    default_min_sep = 1.5 * step_deg
    try:
        min_sep_str = input(f"Minimum separation between candidates (deg) [default {default_min_sep:.1f}]: ")
        min_sep = float(min_sep_str) if min_sep_str.strip() else default_min_sep
    except ValueError:
        min_sep = default_min_sep

    try:
        lat_min_str = input("Optional lat min (degrees, blank for global): ")
        lat_min = float(lat_min_str) if lat_min_str.strip() else None
    except ValueError:
        lat_min = None
    try:
        lat_max_str = input("Optional lat max (degrees, blank for global): ")
        lat_max = float(lat_max_str) if lat_max_str.strip() else None
    except ValueError:
        lat_max = None
    try:
        lon_min_str = input("Optional lon min (degrees, blank for global): ")
        lon_min = float(lon_min_str) if lon_min_str.strip() else None
    except ValueError:
        lon_min = None
    try:
        lon_max_str = input("Optional lon max (degrees, blank for global): ")
        lon_max = float(lon_max_str) if lon_max_str.strip() else None
    except ValueError:
        lon_max = None

    print("\nWeights for combining G and H into F (F = alpha*G + beta*H).")
    try:
        alpha_str = input("Alpha (weight on G) [default 1.0]: ")
        alpha = float(alpha_str) if alpha_str.strip() else 1.0
    except ValueError:
        alpha = 1.0

    try:
        beta_str = input("Beta (weight on H) [default 1.0]: ")
        beta = float(beta_str) if beta_str.strip() else 1.0
    except ValueError:
        beta = 1.0

    print(f"\nScanning globe with step {step_deg}°; this may take a bit...\n")

    selected, stats = compute_top_F_candidates(
        step_deg=step_deg,
        top_N=top_N,
        min_sep=min_sep,
        alpha=alpha,
        beta=beta,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        return_stats=True,
    )

    print("=== Top F (G+H) Node Candidates ===\n")
    print(f"[INFO] F distribution over grid: min={stats['min']:.3f}, median={stats['median']:.3f}, p90={stats['p90']:.3f}, max={stats['max']:.3f}")
    print("Rank |     F     |    G    |    H    |   Lat   |   Lon   | Region")
    print("-----+-----------+---------+---------+---------+---------+---------------------------")
    for rank, cand in enumerate(selected, start=1):
        print(
            f"{rank:4d} | {cand['F']:9.3f} | {cand['G']:7.3f} | {cand['H']:7.3f} | "
            f"{cand['lat']:+7.2f}° | {cand['lon']:+7.2f}° | {cand['region']}"
        )

    print("\nNote: In this mode, G uses NetCDF GA/CT datasets and boundary distance derived from the loaded KMZ.")
    print("      H is Penrose-style geometric coherence with known GSN nodes.\n")


# ---------------------------------------------------------
# Mode 4: Generate Mollweide map of F candidates
# ---------------------------------------------------------
def mode_plot_mollweide_map():
    if not USE_XARRAY:
        print("\n[ERROR] xarray is not available. Install it with `pip install xarray netCDF4 cartopy matplotlib`.")
        return

    if _ga_ds is None or _ct_ds is None:
        print("\n[ERROR] NetCDF datasets are required for map generation. Provide real data files and restart.\n")
        return

    describe_scoring_params()

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.ticker as mticker  # type: ignore
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
    except Exception as e:
        print(f"\n[ERROR] Missing plotting dependency: {e}")
        print("Install with: pip install cartopy matplotlib")
        return

    try:
        step_str = input("Grid step in degrees (e.g. 10, 15) [default 10]: ")
        step_deg = float(step_str) if step_str.strip() else 10.0
    except ValueError:
        step_deg = 10.0

    try:
        N_str = input("How many top candidate locations to list? [default 30]: ")
        top_N = int(N_str) if N_str.strip() else 30
    except ValueError:
        top_N = 30

    default_min_sep = 1.5 * step_deg
    try:
        min_sep_str = input(f"Minimum separation between candidates (deg) [default {default_min_sep:.1f}]: ")
        min_sep = float(min_sep_str) if min_sep_str.strip() else default_min_sep
    except ValueError:
        min_sep = default_min_sep

    try:
        clon_str = input("Central longitude for map (e.g. 0 or 180) [default 0]: ")
        central_lon = float(clon_str) if clon_str.strip() else 0.0
    except ValueError:
        central_lon = 0.0

    try:
        lat_min_str = input("Optional lat min (degrees, blank for global): ")
        lat_min = float(lat_min_str) if lat_min_str.strip() else None
    except ValueError:
        lat_min = None
    try:
        lat_max_str = input("Optional lat max (degrees, blank for global): ")
        lat_max = float(lat_max_str) if lat_max_str.strip() else None
    except ValueError:
        lat_max = None
    try:
        lon_min_str = input("Optional lon min (degrees, blank for global): ")
        lon_min = float(lon_min_str) if lon_min_str.strip() else None
    except ValueError:
        lon_min = None
    try:
        lon_max_str = input("Optional lon max (degrees, blank for global): ")
        lon_max = float(lon_max_str) if lon_max_str.strip() else None
    except ValueError:
        lon_max = None

    print("\nWeights for combining G and H into F (F = alpha*G + beta*H).")
    try:
        alpha_str = input("Alpha (weight on G) [default 1.0]: ")
        alpha = float(alpha_str) if alpha_str.strip() else 1.0
    except ValueError:
        alpha = 1.0

    try:
        beta_str = input("Beta (weight on H) [default 1.0]: ")
        beta = float(beta_str) if beta_str.strip() else 1.0
    except ValueError:
        beta = 1.0

    try:
        annotate_str = input("How many top candidates to label? [default 5]: ")
        annotate_top = int(annotate_str) if annotate_str.strip() else 5
    except ValueError:
        annotate_top = 5

    show_boundaries = input("Show plate boundaries? [Y/n]: ").strip().lower() != "n"

    output_path = input("Save map to file (e.g., map.png or leave blank to skip): ").strip()

    print(f"\nComputing top {top_N} candidates at {step_deg}° grid; this may take a bit...\n")
    candidates, stats = compute_top_F_candidates(
        step_deg=step_deg,
        top_N=top_N,
        min_sep=min_sep,
        alpha=alpha,
        beta=beta,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        return_stats=True,
    )

    # Categorize candidates
    strong = [c for c in candidates if c["F"] >= 3.0]
    moderate = [c for c in candidates if 2.0 <= c["F"] < 3.0]
    weak = [c for c in candidates if 1.0 <= c["F"] < 2.0]

    proj = ccrs.Mollweide(central_longitude=central_lon)
    fig = plt.figure(figsize=(11, 7))
    ax = plt.axes(projection=proj)
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbe9ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", linestyle=":")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(range(-60, 61, 30))

    # Known nodes
    for name, (lat, lon) in KNOWN_NODES.items():
        ax.scatter(lon, lat, transform=ccrs.PlateCarree(), s=60, color="#e41a1c", marker="o", zorder=4)
        ax.text(lon + 2, lat + 2, name, transform=ccrs.PlateCarree(), fontsize=7, color="#e41a1c")

    # Plot plate boundaries if available
    if show_boundaries and _plate_boundaries:
        for poly in _plate_boundaries:
            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            ax.plot(
                lons,
                lats,
                transform=ccrs.PlateCarree(),
                linewidth=0.5,
                color="#666666",
                alpha=0.6,
                zorder=2,
            )

    # Candidates with continuous colorbar by F
    Fs = np.array([c["F"] for c in candidates])
    lats_all = [c["lat"] for c in candidates]
    lons_all = [c["lon"] for c in candidates]
    norm = plt.Normalize(vmin=float(Fs.min()), vmax=float(Fs.max()))
    cmap = plt.cm.plasma
    sc_all = ax.scatter(
        lons_all,
        lats_all,
        c=Fs,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        s=45,
        marker="^",
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    cbar = plt.colorbar(sc_all, orientation="vertical", pad=0.02, shrink=0.8, ax=ax)
    cbar.set_label("F (alpha*G + beta*H)", fontsize=8)

    print(f"[INFO] F distribution over grid: min={stats['min']:.3f}, median={stats['median']:.3f}, p90={stats['p90']:.3f}, max={stats['max']:.3f}")

    # Annotate top candidates (with overlap guard and alternating offsets)
    annotated = []
    offsets = [(2, -3), (-2, 3), (2, 3), (-2, -3), (3, 0), (-3, 0)]
    for idx, cand in enumerate(candidates[:max(0, annotate_top)]):
        skip = False
        for prev in annotated:
            if great_circle_angle(cand["lat"], cand["lon"], prev["lat"], prev["lon"]) < 4.0:
                skip = True
                break
        if skip:
            continue
        label = f"{cand['region']} (F={cand['F']:.2f})"
        dx, dy = offsets[idx % len(offsets)]
        ax.text(
            cand["lon"] + dx,
            cand["lat"] + dy,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=7,
            color="black",
            zorder=6,
        )
        annotated.append(cand)

    ax.set_title(
        "Global Stabilization Network and Predicted Node Candidates\nMollweide Projection",
        fontsize=12,
        pad=12,
    )

    # Optional legend for known nodes and candidates
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markersize=7, label="Known GSN"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=cmap(0.8), markeredgecolor="black", markersize=7, label="Candidates"),
            plt.Line2D([0], [0], color="#666666", linewidth=0.8, label="Plate boundaries" if show_boundaries else "Plate boundaries (hidden)"),
        ],
        loc="lower left",
        fontsize=8,
        frameon=True,
    )

    # Annotate run parameters on figure
    param_text = (
        f"step={step_deg}°, min_sep={min_sep:.1f}°, top_N={top_N}, annotate={annotate_top}\n"
        f"alpha={alpha}, beta={beta}, bounds: lat[{lat_min if lat_min is not None else 'all'}, {lat_max if lat_max is not None else 'all'}], "
        f"lon[{lon_min if lon_min is not None else 'all'}, {lon_max if lon_max is not None else 'all'}]\n"
        f"central_lon={central_lon}, boundaries={'on' if show_boundaries else 'off'}"
    )
    ax.text(
        0.99,
        -0.08,
        param_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="gray",
    )

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved map to {output_path}")
        except Exception as e:
            print(f"[WARN] Could not save map: {e}")

    # Emit Google Maps links for quick navigation
    print("\nGoogle Maps links for plotted candidates:")
    for idx, cand in enumerate(candidates, start=1):
        gmaps_url = f"https://www.google.com/maps?q={cand['lat']},{cand['lon']}"
        print(f"{idx:3d}. F={cand['F']:.3f} lat={cand['lat']:+.2f} lon={cand['lon']:+.2f} -> {gmaps_url}")

    plt.show()


def mode_validate_known_nodes():
    if _ga_ds is None or _ct_ds is None:
        print("\n[ERROR] NetCDF datasets are required for validation.")
        return

    try:
        alpha_str = input("Alpha (weight on G) [default 1.0]: ")
        alpha = float(alpha_str) if alpha_str.strip() else 1.0
    except ValueError:
        alpha = 1.0

    try:
        beta_str = input("Beta (weight on H) [default 1.0]: ")
        beta = float(beta_str) if beta_str.strip() else 1.0
    except ValueError:
        beta = 1.0

    results = []
    for name, (lat, lon) in KNOWN_NODES.items():
        ga, ct, dist_km = get_geophysical_inputs(lat, lon)
        G, comps = compute_G(ga, ct, dist_km)
        H = compute_H(lat, lon)
        F = alpha * G + beta * H
        results.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "ga": ga,
                "ct": ct,
                "dist": dist_km,
                "G": G,
                "H": H,
                "F": F,
                **comps,
            }
        )

    results.sort(key=lambda x: x["F"], reverse=True)
    F_vals = np.array([r["F"] for r in results])
    print("\n=== Validation: Known Nodes ===\n")
    print(f"[INFO] F stats: min={F_vals.min():.3f}, mean={F_vals.mean():.3f}, median={np.median(F_vals):.3f}, max={F_vals.max():.3f}")
    print("Rank |     F     |    G    |    H    |    GA(mGal) | CT(km) | Dist(km) | Name")
    print("-----+-----------+---------+---------+-------------+--------+----------+--------------------")
    for idx, r in enumerate(results, start=1):
        print(
            f"{idx:4d} | {r['F']:9.3f} | {r['G']:7.3f} | {r['H']:7.3f} | "
            f"{r['ga']:11.2f} | {r['ct']:6.2f} | {r['dist']:8.1f} | {r['name']}"
        )
    print("")


# ---------------------------------------------------------
# Mode 6: Full-resolution global scan (native NetCDF resolution)
# ---------------------------------------------------------
def mode_full_resolution_scan():
    """
    Scan the entire globe at native NetCDF resolution (~2 arc-minutes).
    Outputs F grid to NetCDF and optionally generates a heatmap.
    """
    if not USE_XARRAY:
        print("\n[ERROR] xarray is required. Install with: pip install xarray netCDF4")
        return
    
    if not USE_SCIPY:
        print("\n[ERROR] scipy is required for full-resolution scan. Install with: pip install scipy")
        return
    
    if _ga_ds is None or _ct_ds is None:
        print("\n[ERROR] NetCDF datasets are required. Provide real data files and restart.\n")
        return
    
    print("\n=== Mode 6: Full-Resolution Global Scan ===\n")
    print("This computes F values at the native NetCDF grid resolution.")
    print("WARNING: This can take several minutes and use significant memory.\n")
    
    describe_scoring_params()
    
    # Get coordinate arrays from the gravity dataset
    ga_var = _ga_ds[GA_VAR_NAME]
    native_lats = ga_var.coords["lat"].values
    native_lons = ga_var.coords["lon"].values
    
    print(f"\n[INFO] Native grid: {len(native_lats)} lat x {len(native_lons)} lon = {len(native_lats) * len(native_lons):,} points")
    print(f"[INFO] Lat range: {native_lats.min():.4f} to {native_lats.max():.4f}")
    print(f"[INFO] Lon range: {native_lons.min():.4f} to {native_lons.max():.4f}")
    
    # Ask if user wants to subsample
    print("\nOptions:")
    print("  1) Full native resolution (may be slow)")
    print("  2) Subsample to reduce computation time")
    res_choice = input("Choose [1/2, default 1]: ").strip()
    
    if res_choice == "2":
        try:
            factor_str = input("Subsample factor (e.g., 2 = every 2nd point, 10 = every 10th) [default 10]: ")
            factor = int(factor_str) if factor_str.strip() else 10
        except ValueError:
            factor = 10
        lats = native_lats[::factor]
        lons = native_lons[::factor]
        print(f"[INFO] Subsampled to {len(lats)} lat x {len(lons)} lon = {len(lats) * len(lons):,} points")
    else:
        lats = native_lats
        lons = native_lons
    
    # Get weights
    print("\nWeights for combining G and H into F (F = alpha*G + beta*H).")
    try:
        alpha_str = input("Alpha (weight on G) [default 1.0]: ")
        alpha = float(alpha_str) if alpha_str.strip() else 1.0
    except ValueError:
        alpha = 1.0
    
    try:
        beta_str = input("Beta (weight on H) [default 1.0]: ")
        beta = float(beta_str) if beta_str.strip() else 1.0
    except ValueError:
        beta = 1.0
    
    # Compute the full F grid
    result = compute_F_grid(lats, lons, alpha=alpha, beta=beta)
    
    # Output options
    print("\n=== Output Options ===")
    
    # Save to NetCDF
    output_nc = input("Save F grid to NetCDF file (e.g., F_grid.nc, or blank to skip): ").strip()
    if output_nc:
        try:
            import xarray as xr
            
            ds = xr.Dataset(
                {
                    "F": (["lat", "lon"], result["F"]),
                    "G": (["lat", "lon"], result["G"]),
                    "H": (["lat", "lon"], result["H"]),
                    "boundary_distance_km": (["lat", "lon"], result["boundary_distance"]),
                },
                coords={
                    "lat": result["lats"],
                    "lon": result["lons"],
                },
                attrs={
                    "description": "GSN Node Index (F = alpha*G + beta*H)",
                    "alpha": alpha,
                    "beta": beta,
                    "ga_scale": G_PARAMS["ga_scale"],
                    "ct_mean": G_PARAMS["ct_mean"],
                    "ct_std": G_PARAMS["ct_std"],
                    "L_km": G_PARAMS["L"],
                },
            )
            ds.to_netcdf(output_nc)
            print(f"[INFO] Saved F grid to {output_nc}")
        except Exception as e:
            print(f"[ERROR] Failed to save NetCDF: {e}")
    
    # Extract and print top candidates
    try:
        top_n_str = input("How many top candidates to extract? [default 30]: ")
        top_n = int(top_n_str) if top_n_str.strip() else 30
    except ValueError:
        top_n = 30
    
    try:
        min_sep_str = input("Minimum separation between candidates (degrees) [default 2.0]: ")
        min_sep = float(min_sep_str) if min_sep_str.strip() else 2.0
    except ValueError:
        min_sep = 2.0
    
    candidates = extract_top_candidates_from_grid(result, top_n=top_n, min_sep_deg=min_sep)
    
    print(f"\n=== Top {len(candidates)} F Candidates (Native Resolution) ===\n")
    print("Rank |     F     |    G    |    H    |   Lat   |   Lon   | Region")
    print("-----+-----------+---------+---------+---------+---------+---------------------------")
    for rank, cand in enumerate(candidates, start=1):
        print(
            f"{rank:4d} | {cand['F']:9.3f} | {cand['G']:7.3f} | {cand['H']:7.3f} | "
            f"{cand['lat']:+7.2f}° | {cand['lon']:+7.2f}° | {cand['region']}"
        )
    
    # Optional heatmap
    plot_choice = input("\nGenerate heatmap plot? [y/N]: ").strip().lower()
    if plot_choice == "y":
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            print("[INFO] Generating heatmap...")
            
            proj = ccrs.Robinson()
            fig = plt.figure(figsize=(14, 8))
            ax = plt.axes(projection=proj)
            ax.set_global()
            
            # Plot F as a filled contour / pcolormesh
            lon_grid, lat_grid = np.meshgrid(result["lons"], result["lats"])
            
            im = ax.pcolormesh(
                lon_grid, lat_grid, result["F"],
                transform=ccrs.PlateCarree(),
                cmap="plasma",
                shading="auto",
            )
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="white", linestyle=":")
            
            # Plate boundaries
            if _plate_boundaries:
                for poly in _plate_boundaries:
                    plats = [p[0] for p in poly]
                    plons = [p[1] for p in poly]
                    ax.plot(plons, plats, transform=ccrs.PlateCarree(),
                            linewidth=0.4, color="cyan", alpha=0.6)
            
            # Known nodes
            for name, (nlat, nlon) in KNOWN_NODES.items():
                ax.scatter(nlon, nlat, transform=ccrs.PlateCarree(),
                           s=40, color="red", marker="o", edgecolor="white", linewidth=0.5, zorder=10)
            
            # Top candidates
            for cand in candidates[:10]:
                ax.scatter(cand["lon"], cand["lat"], transform=ccrs.PlateCarree(),
                           s=30, color="lime", marker="^", edgecolor="black", linewidth=0.3, zorder=11)
            
            cbar = plt.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.8, ax=ax)
            cbar.set_label(f"F (Node Index, α={alpha}, β={beta})", fontsize=10)
            
            ax.set_title("GSN Node Index F - Full Resolution Heatmap", fontsize=12, pad=10)
            
            plt.tight_layout()
            
            save_path = input("Save heatmap to file (e.g., heatmap.png, or blank to show only): ").strip()
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"[INFO] Saved heatmap to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"[ERROR] Failed to generate heatmap: {e}")
            print("Install dependencies: pip install matplotlib cartopy")
    
    # Google Maps links
    print("\nGoogle Maps links for top candidates:")
    for idx, cand in enumerate(candidates[:20], start=1):
        gmaps_url = f"https://www.google.com/maps?q={cand['lat']},{cand['lon']}"
        print(f"{idx:3d}. F={cand['F']:.3f} lat={cand['lat']:+.2f} lon={cand['lon']:+.2f} -> {gmaps_url}")
    
    print("\nDone!")


# ---------------------------------------------------------
# Mode 7: ML-Based Grid Scan
# ---------------------------------------------------------
def mode_ml_grid_scan():
    """
    ML-based grid scan using trained neural network.
    
    Options:
    - Train new model from known nodes
    - Use existing trained model
    - Generate ML-based heatmap
    """
    print("\n=== Mode 7: ML-Based Grid Scan ===\n")
    print("This mode uses a trained neural network to score grid points,")
    print("replacing the hand-crafted F = alpha*G + beta*H formula.\n")
    
    # Check dependencies
    try:
        from gsn_ml_grid_scorer import MLGridScorer, TrainingConfig
        ml_available = True
    except ImportError:
        print("[ERROR] gsn_ml_grid_scorer module not available.")
        print("Make sure gsn_ml_grid_scorer.py is in the same directory.")
        return
    
    try:
        import torch
        torch_available = True
        print(f"[INFO] PyTorch version: {torch.__version__}")
    except ImportError:
        print("[ERROR] PyTorch not available. Install with: pip install torch")
        return
    
    # Check for existing model
    import os
    model_path = "gsn_ml_grid_scorer.pth"
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        print(f"[INFO] Found existing model: {model_path}")
    else:
        print("[INFO] No trained model found.")
    
    print("\nOptions:")
    print("  1) Train new ML model from known nodes")
    print("  2) Load existing model and run ML grid scan")
    print("  3) Compare ML vs Linear scoring")
    
    sub_choice = input("\nChoose option [1/2/3]: ").strip()
    
    if sub_choice == "1":
        _mode_7_train_model(model_path)
    elif sub_choice == "2":
        if not model_exists:
            print("[ERROR] No trained model found. Train a model first (option 1).")
            return
        _mode_7_run_scan(model_path)
    elif sub_choice == "3":
        if not model_exists:
            print("[ERROR] No trained model found. Train a model first (option 1).")
            return
        _mode_7_compare(model_path)
    else:
        print("Invalid option.")


def _mode_7_train_model(model_path):
    """Train a new ML model from known nodes."""
    from gsn_ml_grid_scorer import TrainingConfig
    
    print("\n=== Training ML Grid Scorer ===\n")
    
    # Training parameters
    try:
        epochs_str = input("Number of training epochs [default 150]: ").strip()
        epochs = int(epochs_str) if epochs_str else 150
    except ValueError:
        epochs = 150
    
    try:
        aug_str = input("Augmentations per positive sample [default 10]: ").strip()
        n_aug = int(aug_str) if aug_str else 10
    except ValueError:
        n_aug = 10
    
    try:
        neg_str = input("Number of random negatives [default 2000]: ").strip()
        n_neg = int(neg_str) if neg_str else 2000
    except ValueError:
        n_neg = 2000
    
    config = TrainingConfig(
        epochs=epochs,
        n_augmentations=n_aug,
        n_random_negatives=n_neg,
    )
    
    # Load known nodes
    try:
        from known_nodes_extended import KNOWN_NODES_EXTENDED
        known_nodes = KNOWN_NODES_EXTENDED
        print(f"[INFO] Loaded {len(known_nodes)} nodes from known_nodes_extended.py")
    except ImportError:
        known_nodes = KNOWN_NODES
        print(f"[INFO] Using {len(known_nodes)} core nodes from KNOWN_NODES")
    
    # Train
    print("\nStarting training...")
    scorer = train_ml_scorer(known_nodes, config, model_path, verbose=True)
    
    if scorer is not None:
        print(f"\n[SUCCESS] Model trained and saved to {model_path}")
        
        # Offer to run a quick scan
        run_scan = input("\nRun a quick ML grid scan now? [Y/n]: ").strip().lower()
        if run_scan != "n":
            _mode_7_run_scan(model_path)


def _mode_7_run_scan(model_path):
    """Run ML-based grid scan."""
    print("\n=== ML Grid Scan ===\n")
    
    if _ga_ds is None or _ct_ds is None:
        print("[ERROR] NetCDF datasets required for grid scan.")
        return
    
    # Get grid parameters
    ga_var = _ga_ds[GA_VAR_NAME]
    native_lats = ga_var.coords["lat"].values
    native_lons = ga_var.coords["lon"].values
    
    print(f"[INFO] Native grid: {len(native_lats)} x {len(native_lons)}")
    
    try:
        factor_str = input("Subsample factor [default 20]: ").strip()
        factor = int(factor_str) if factor_str else 20
    except ValueError:
        factor = 20
    
    lats = native_lats[::factor]
    lons = native_lons[::factor]
    print(f"[INFO] Using {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
    
    # Run ML-based computation
    result = compute_F_grid_ml(lats, lons, model_path=model_path)
    
    method_used = result.get("method", "ml")
    print(f"\n[INFO] Scoring method: {method_used.upper()}")
    
    # Extract candidates
    try:
        top_n_str = input("Number of top candidates [default 30]: ").strip()
        top_n = int(top_n_str) if top_n_str else 30
    except ValueError:
        top_n = 30
    
    candidates = extract_top_candidates_from_grid(result, top_n=top_n, min_sep_deg=2.0)
    
    print(f"\n=== Top {len(candidates)} ML Candidates ===\n")
    print("Rank |     F     |    G    |    H    |   Lat   |   Lon   | Region")
    print("-----+-----------+---------+---------+---------+---------+---------------------------")
    for rank, cand in enumerate(candidates, start=1):
        print(
            f"{rank:4d} | {cand['F']:9.3f} | {cand['G']:7.3f} | {cand['H']:7.3f} | "
            f"{cand['lat']:+7.2f}° | {cand['lon']:+7.2f}° | {cand['region']}"
        )
    
    # Optional heatmap
    plot_choice = input("\nGenerate heatmap? [y/N]: ").strip().lower()
    if plot_choice == "y":
        _generate_ml_heatmap(result, candidates, "ML-Based")
    
    # Google Maps links
    print("\nGoogle Maps links for top candidates:")
    for idx, cand in enumerate(candidates[:15], start=1):
        gmaps_url = f"https://www.google.com/maps?q={cand['lat']},{cand['lon']}"
        print(f"{idx:3d}. F={cand['F']:.3f} -> {gmaps_url}")


def _mode_7_compare(model_path):
    """Compare ML vs Linear scoring methods."""
    print("\n=== Comparing ML vs Linear Scoring ===\n")
    
    if _ga_ds is None or _ct_ds is None:
        print("[ERROR] NetCDF datasets required for comparison.")
        return
    
    # Get grid
    ga_var = _ga_ds[GA_VAR_NAME]
    native_lats = ga_var.coords["lat"].values
    native_lons = ga_var.coords["lon"].values
    
    try:
        factor_str = input("Subsample factor [default 30]: ").strip()
        factor = int(factor_str) if factor_str else 30
    except ValueError:
        factor = 30
    
    lats = native_lats[::factor]
    lons = native_lons[::factor]
    print(f"[INFO] Comparing on {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
    
    import time
    
    # Linear scoring
    print("\n[1/2] Computing Linear F = alpha*G + beta*H...")
    start = time.time()
    result_linear = compute_F_grid(lats, lons, alpha=1.0, beta=1.0)
    linear_time = time.time() - start
    
    # ML scoring
    print("\n[2/2] Computing ML-based F...")
    start = time.time()
    result_ml = compute_F_grid_ml(lats, lons, model_path=model_path, fallback_linear=False)
    ml_time = time.time() - start
    
    # Compare
    print("\n=== Comparison Results ===\n")
    print(f"{'Method':<10} | {'Time (s)':<10} | {'F min':<10} | {'F max':<10} | {'F median':<10}")
    print("-" * 60)
    print(f"{'Linear':<10} | {linear_time:<10.2f} | {result_linear['F'].min():<10.3f} | "
          f"{result_linear['F'].max():<10.3f} | {np.median(result_linear['F']):<10.3f}")
    print(f"{'ML':<10} | {ml_time:<10.2f} | {result_ml['F'].min():<10.3f} | "
          f"{result_ml['F'].max():<10.3f} | {np.median(result_ml['F']):<10.3f}")
    
    # Correlation between methods
    F_linear_flat = result_linear['F'].flatten()
    F_ml_flat = result_ml['F'].flatten()
    correlation = np.corrcoef(F_linear_flat, F_ml_flat)[0, 1]
    print(f"\nCorrelation between methods: {correlation:.4f}")
    
    # Extract candidates from both
    cands_linear = extract_top_candidates_from_grid(result_linear, top_n=20, min_sep_deg=3.0)
    cands_ml = extract_top_candidates_from_grid(result_ml, top_n=20, min_sep_deg=3.0)
    
    # Check overlap
    def coords_match(c1, c2, threshold_deg=3.0):
        return great_circle_angle(c1['lat'], c1['lon'], c2['lat'], c2['lon']) < threshold_deg
    
    overlap_count = 0
    for c_ml in cands_ml:
        for c_lin in cands_linear:
            if coords_match(c_ml, c_lin):
                overlap_count += 1
                break
    
    print(f"Candidate overlap (top 20): {overlap_count}/{len(cands_ml)} ({100*overlap_count/len(cands_ml):.1f}%)")
    
    # Show unique ML candidates
    print("\n=== Unique ML Candidates (not in Linear top 20) ===")
    unique_ml = []
    for c_ml in cands_ml:
        is_unique = True
        for c_lin in cands_linear:
            if coords_match(c_ml, c_lin):
                is_unique = False
                break
        if is_unique:
            unique_ml.append(c_ml)
    
    for i, cand in enumerate(unique_ml[:10], 1):
        print(f"  {i}. F={cand['F']:.3f} at ({cand['lat']:+.2f}, {cand['lon']:+.2f}) - {cand['region']}")


def _generate_ml_heatmap(result, candidates, title_prefix=""):
    """Generate heatmap for ML results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        print("[INFO] Generating heatmap...")
        
        proj = ccrs.Robinson()
        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=proj)
        ax.set_global()
        
        lon_grid, lat_grid = np.meshgrid(result["lons"], result["lats"])
        
        im = ax.pcolormesh(
            lon_grid, lat_grid, result["F"],
            transform=ccrs.PlateCarree(),
            cmap="plasma",
            shading="auto",
        )
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="white", linestyle=":")
        
        # Plate boundaries
        if _plate_boundaries:
            for poly in _plate_boundaries:
                plats = [p[0] for p in poly]
                plons = [p[1] for p in poly]
                ax.plot(plons, plats, transform=ccrs.PlateCarree(),
                        linewidth=0.4, color="cyan", alpha=0.6)
        
        # Known nodes
        for name, (nlat, nlon) in KNOWN_NODES.items():
            ax.scatter(nlon, nlat, transform=ccrs.PlateCarree(),
                       s=40, color="red", marker="o", edgecolor="white", linewidth=0.5, zorder=10)
        
        # Top candidates
        for cand in candidates[:10]:
            ax.scatter(cand["lon"], cand["lat"], transform=ccrs.PlateCarree(),
                       s=30, color="lime", marker="^", edgecolor="black", linewidth=0.3, zorder=11)
        
        cbar = plt.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.8, ax=ax)
        method = result.get("method", "ml").upper()
        cbar.set_label(f"F ({method} Score)", fontsize=10)
        
        ax.set_title(f"GSN Node Index - {title_prefix} Scoring", fontsize=12, pad=10)
        
        plt.tight_layout()
        
        save_path = input("Save heatmap to file (blank to show only): ").strip()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Saved to {save_path}")
        
        plt.show()
        
    except ImportError as e:
        print(f"[ERROR] Missing plotting library: {e}")
        print("Install with: pip install matplotlib cartopy")


# ---------------------------------------------------------
# Main entry
# ---------------------------------------------------------
def main():
    print("\n=== Global Stabilization Network Node Tool ===")
    print("  1) Evaluate a single location (G, H, F)")
    print("  2) Scan globe for top-N geometric (H) peaks")
    print("  3) Scan globe for top-N F (G+H) candidates using real data")
    print("  4) Generate Mollweide map of top-N F candidates (cartopy required)")
    print("  5) Validate known GSN nodes")
    print("  6) Full-resolution global scan (native grid, scipy required)")
    print("  7) ML-based grid scan (neural network scoring)\n")

    # Try load datasets early for modes 1 and 3
    loaded = try_load_datasets()
    if loaded:
        print("[INFO] NetCDF gravity/crust datasets loaded successfully.")
    else:
        print("[ERROR] NetCDF datasets not loaded. Provide real data files before running Modes 1, 3, 4, 5, 6, or 7.")

    # Load plate boundaries (used by all modes when available)
    global _plate_boundaries
    _plate_boundaries = load_plate_boundaries()

    choice = input("\nChoose mode [1/2/3/4/5/6/7]: ").strip()

    if choice == "1":
        mode_single_location()
    elif choice == "2":
        mode_global_scan_H()
    elif choice == "3":
        mode_global_scan_F()
    elif choice == "4":
        mode_plot_mollweide_map()
    elif choice == "5":
        mode_validate_known_nodes()
    elif choice == "6":
        mode_full_resolution_scan()
    elif choice == "7":
        mode_ml_grid_scan()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()

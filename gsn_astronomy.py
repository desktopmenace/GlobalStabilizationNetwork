"""
GSN Astronomy Module

Integrates astronomical/celestial data with GSN node predictions through:
- Constellation pattern matching
- Visibility analysis
- Celestial alignment scoring

Uses skyfield for accurate astronomical calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from datetime import datetime, timezone

# Try to import skyfield
try:
    from skyfield.api import load, Topos, Star
    from skyfield import almanac
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False
    print("[WARN] skyfield not installed. Install with: pip install skyfield")

# =========================================================
# Star Catalog and Constellation Data (J2000 coordinates)
# =========================================================

# Major constellations with key stars (RA in degrees, Dec in degrees)
CONSTELLATIONS = {
    "orion": {
        "name": "Orion",
        "stars": [
            {"name": "Betelgeuse", "ra": 88.79, "dec": 7.41, "mag": 0.42},
            {"name": "Rigel", "ra": 78.63, "dec": -8.20, "mag": 0.13},
            {"name": "Bellatrix", "ra": 81.28, "dec": 6.35, "mag": 1.64},
            {"name": "Alnilam", "ra": 84.05, "dec": -1.20, "mag": 1.69},  # Belt center
            {"name": "Alnitak", "ra": 85.19, "dec": -1.94, "mag": 1.77},  # Belt left
            {"name": "Mintaka", "ra": 83.00, "dec": -0.30, "mag": 2.23},  # Belt right
            {"name": "Saiph", "ra": 86.94, "dec": -9.67, "mag": 2.06},
        ],
        "pattern": [(0, 2), (2, 5), (5, 3), (3, 4), (4, 1), (1, 6), (6, 4), (0, 3)],
        "sacred": True,
    },
    "orion_belt": {
        "name": "Orion's Belt",
        "stars": [
            {"name": "Alnitak", "ra": 85.19, "dec": -1.94, "mag": 1.77},
            {"name": "Alnilam", "ra": 84.05, "dec": -1.20, "mag": 1.69},
            {"name": "Mintaka", "ra": 83.00, "dec": -0.30, "mag": 2.23},
        ],
        "pattern": [(0, 1), (1, 2)],
        "sacred": True,
    },
    "pleiades": {
        "name": "Pleiades (Seven Sisters)",
        "stars": [
            {"name": "Alcyone", "ra": 56.87, "dec": 24.11, "mag": 2.87},
            {"name": "Atlas", "ra": 57.29, "dec": 24.05, "mag": 3.63},
            {"name": "Electra", "ra": 56.22, "dec": 24.11, "mag": 3.70},
            {"name": "Maia", "ra": 56.46, "dec": 24.37, "mag": 3.88},
            {"name": "Merope", "ra": 56.58, "dec": 23.95, "mag": 4.18},
            {"name": "Taygeta", "ra": 56.30, "dec": 24.47, "mag": 4.30},
            {"name": "Celaeno", "ra": 56.20, "dec": 24.29, "mag": 5.45},
        ],
        "pattern": [(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (2, 6)],
        "sacred": True,
    },
    "ursa_major": {
        "name": "Ursa Major (Big Dipper)",
        "stars": [
            {"name": "Dubhe", "ra": 165.93, "dec": 61.75, "mag": 1.79},
            {"name": "Merak", "ra": 165.46, "dec": 56.38, "mag": 2.37},
            {"name": "Phecda", "ra": 178.46, "dec": 53.69, "mag": 2.44},
            {"name": "Megrez", "ra": 183.86, "dec": 57.03, "mag": 3.31},
            {"name": "Alioth", "ra": 193.51, "dec": 55.96, "mag": 1.77},
            {"name": "Mizar", "ra": 200.98, "dec": 54.93, "mag": 2.27},
            {"name": "Alkaid", "ra": 206.89, "dec": 49.31, "mag": 1.86},
        ],
        "pattern": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 0)],
        "sacred": True,
    },
    "draco": {
        "name": "Draco",
        "stars": [
            {"name": "Thuban", "ra": 211.10, "dec": 64.38, "mag": 3.65},  # Ancient pole star
            {"name": "Eltanin", "ra": 269.15, "dec": 51.49, "mag": 2.23},
            {"name": "Rastaban", "ra": 262.61, "dec": 52.30, "mag": 2.79},
            {"name": "Kochab", "ra": 222.68, "dec": 74.16, "mag": 2.08},
        ],
        "pattern": [(0, 1), (1, 2), (2, 3)],
        "sacred": True,
    },
    "cygnus": {
        "name": "Cygnus (Northern Cross)",
        "stars": [
            {"name": "Deneb", "ra": 310.36, "dec": 45.28, "mag": 1.25},
            {"name": "Sadr", "ra": 305.56, "dec": 40.26, "mag": 2.20},
            {"name": "Gienah", "ra": 305.25, "dec": 33.97, "mag": 2.46},
            {"name": "Albireo", "ra": 292.68, "dec": 27.96, "mag": 3.18},
            {"name": "Fawaris", "ra": 296.24, "dec": 45.13, "mag": 2.87},
        ],
        "pattern": [(0, 1), (1, 2), (2, 3), (1, 4)],
        "sacred": True,
    },
    # Zodiac constellations
    "aries": {
        "name": "Aries",
        "stars": [
            {"name": "Hamal", "ra": 31.79, "dec": 23.46, "mag": 2.00},
            {"name": "Sheratan", "ra": 28.66, "dec": 20.81, "mag": 2.64},
            {"name": "Mesarthim", "ra": 28.38, "dec": 19.29, "mag": 3.88},
        ],
        "pattern": [(0, 1), (1, 2)],
        "sacred": False,
    },
    "taurus": {
        "name": "Taurus",
        "stars": [
            {"name": "Aldebaran", "ra": 68.98, "dec": 16.51, "mag": 0.85},
            {"name": "Elnath", "ra": 81.57, "dec": 28.61, "mag": 1.65},
            {"name": "Ain", "ra": 67.15, "dec": 19.18, "mag": 3.53},
        ],
        "pattern": [(0, 1), (0, 2)],
        "sacred": False,
    },
    "leo": {
        "name": "Leo",
        "stars": [
            {"name": "Regulus", "ra": 152.09, "dec": 11.97, "mag": 1.35},
            {"name": "Denebola", "ra": 177.26, "dec": 14.57, "mag": 2.14},
            {"name": "Algieba", "ra": 146.46, "dec": 19.84, "mag": 2.28},
        ],
        "pattern": [(0, 2), (0, 1)],
        "sacred": False,
    },
    "scorpius": {
        "name": "Scorpius",
        "stars": [
            {"name": "Antares", "ra": 247.35, "dec": -26.43, "mag": 0.96},
            {"name": "Shaula", "ra": 263.40, "dec": -37.10, "mag": 1.63},
            {"name": "Sargas", "ra": 264.33, "dec": -42.99, "mag": 1.87},
        ],
        "pattern": [(0, 1), (1, 2)],
        "sacred": False,
    },
}

# Sacred astronomical alignments
ALIGNMENTS = {
    "summer_solstice": {"month": 6, "day": 21, "type": "sunrise", "description": "Summer Solstice"},
    "winter_solstice": {"month": 12, "day": 21, "type": "sunrise", "description": "Winter Solstice"},
    "vernal_equinox": {"month": 3, "day": 20, "type": "sunrise", "description": "Vernal Equinox"},
    "autumnal_equinox": {"month": 9, "day": 22, "type": "sunrise", "description": "Autumnal Equinox"},
}

# Cached data
_timescale = None
_ephemeris = None


def _get_timescale():
    """Get cached timescale."""
    global _timescale
    if _timescale is None and HAS_SKYFIELD:
        _timescale = load.timescale()
    return _timescale


def _get_ephemeris():
    """Get cached ephemeris (downloads on first use)."""
    global _ephemeris
    if _ephemeris is None and HAS_SKYFIELD:
        try:
            _ephemeris = load('de421.bsp')
        except Exception as e:
            print(f"[WARN] Could not load ephemeris: {e}")
    return _ephemeris


# =========================================================
# Visibility Calculations
# =========================================================

def get_constellation_visibility(lat: float, lon: float, date: datetime = None) -> Dict[str, float]:
    """
    Calculate visibility scores for major constellations from a location.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        date: Date/time for calculation (default: now)
    
    Returns:
        Dict mapping constellation name to visibility score (0-1)
    """
    if not HAS_SKYFIELD:
        # Fallback: simple geometric visibility based on declination
        return _simple_visibility(lat)
    
    ts = _get_timescale()
    eph = _get_ephemeris()
    
    if ts is None or eph is None:
        return _simple_visibility(lat)
    
    if date is None:
        t = ts.now()
    else:
        t = ts.utc(date.year, date.month, date.day, 12, 0, 0)
    
    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    earth = eph['earth']
    
    visibility = {}
    
    for const_name, const_data in CONSTELLATIONS.items():
        stars = const_data["stars"]
        
        # Calculate center of constellation
        center_ra = np.mean([s["ra"] for s in stars])
        center_dec = np.mean([s["dec"] for s in stars])
        
        try:
            # Create star object at constellation center
            star = Star(ra_hours=center_ra / 15.0, dec_degrees=center_dec)
            
            # Calculate apparent position
            astrometric = (earth + observer).at(t).observe(star)
            alt, az, _ = astrometric.apparent().altaz()
            
            # Visibility score based on altitude
            if alt.degrees > 0:
                # Higher altitude = better visibility (max at 45+ degrees)
                visibility[const_name] = min(1.0, alt.degrees / 45.0)
            else:
                visibility[const_name] = 0.0
                
        except Exception:
            # Fallback for this constellation
            visibility[const_name] = _simple_star_visibility(lat, center_dec)
    
    return visibility


def _simple_visibility(lat: float) -> Dict[str, float]:
    """Simple visibility calculation without skyfield."""
    visibility = {}
    for const_name, const_data in CONSTELLATIONS.items():
        center_dec = np.mean([s["dec"] for s in const_data["stars"]])
        visibility[const_name] = _simple_star_visibility(lat, center_dec)
    return visibility


def _simple_star_visibility(lat: float, dec: float) -> float:
    """
    Simple visibility based on latitude and declination.
    A star is visible if its declination is within (90 - |lat|) of the celestial pole.
    """
    # Stars are visible if dec > lat - 90 (northern) or dec < lat + 90 (southern)
    max_dec = lat + 90
    min_dec = lat - 90
    
    if min_dec <= dec <= max_dec:
        # Closer to zenith = higher visibility
        zenith_dist = abs(dec - lat)
        return max(0, 1 - zenith_dist / 90)
    return 0.0


def get_visible_constellations(lat: float, lon: float, date: datetime = None) -> List[str]:
    """Get list of currently visible constellations from a location."""
    visibility = get_constellation_visibility(lat, lon, date)
    return [name for name, score in visibility.items() if score > 0.1]


# =========================================================
# Solstice/Equinox Alignment
# =========================================================

def compute_solstice_azimuth(lat: float, lon: float, event: str = "summer_solstice") -> float:
    """
    Compute the sunrise/sunset azimuth for a solstice or equinox.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees  
        event: One of "summer_solstice", "winter_solstice", "vernal_equinox", "autumnal_equinox"
    
    Returns:
        Azimuth in degrees (0 = North, 90 = East, etc.)
    """
    if event not in ALIGNMENTS:
        return 90.0  # Default to due east
    
    alignment = ALIGNMENTS[event]
    
    if not HAS_SKYFIELD:
        # Simple approximation
        return _approximate_sunrise_azimuth(lat, alignment["month"], alignment["day"])
    
    ts = _get_timescale()
    eph = _get_ephemeris()
    
    if ts is None or eph is None:
        return _approximate_sunrise_azimuth(lat, alignment["month"], alignment["day"])
    
    try:
        # Use current year
        year = datetime.now().year
        t = ts.utc(year, alignment["month"], alignment["day"], 6, 0, 0)
        
        observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
        earth = eph['earth']
        sun = eph['sun']
        
        # Get sun position at approximate sunrise
        astrometric = (earth + observer).at(t).observe(sun)
        alt, az, _ = astrometric.apparent().altaz()
        
        return az.degrees
        
    except Exception:
        return _approximate_sunrise_azimuth(lat, alignment["month"], alignment["day"])


def _approximate_sunrise_azimuth(lat: float, month: int, day: int) -> float:
    """Approximate sunrise azimuth based on latitude and date."""
    # Solar declination approximation
    day_of_year = (datetime(2024, month, day) - datetime(2024, 1, 1)).days
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # Sunrise azimuth formula (simplified)
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    
    cos_az = np.sin(dec_rad) / np.cos(lat_rad)
    cos_az = np.clip(cos_az, -1, 1)
    
    azimuth = 90 - np.degrees(np.arcsin(cos_az))
    return azimuth


def compute_solstice_alignment_score(lat: float, lon: float, nodes: List[Dict] = None) -> float:
    """
    Compute how well a location aligns with other nodes along solstice directions.
    
    Args:
        lat, lon: Location coordinates
        nodes: List of node dicts with 'lat', 'lon' keys
    
    Returns:
        Alignment score 0-1
    """
    if not nodes or len(nodes) < 2:
        return 0.5  # Neutral score
    
    # Get solstice azimuths for this location
    summer_az = compute_solstice_azimuth(lat, lon, "summer_solstice")
    winter_az = compute_solstice_azimuth(lat, lon, "winter_solstice")
    
    # Check if any nodes lie along these directions
    alignment_scores = []
    
    for node in nodes:
        if node.get("lat") == lat and node.get("lon") == lon:
            continue
        
        # Compute bearing to node
        bearing = _compute_bearing(lat, lon, node["lat"], node["lon"])
        
        # Check alignment with solstice directions (within 5 degrees)
        summer_diff = min(abs(bearing - summer_az), abs(bearing - (summer_az + 180) % 360))
        winter_diff = min(abs(bearing - winter_az), abs(bearing - (winter_az + 180) % 360))
        
        min_diff = min(summer_diff, winter_diff)
        
        if min_diff < 15:  # Within 15 degrees
            alignment_scores.append(1 - min_diff / 15)
    
    if alignment_scores:
        return np.mean(alignment_scores)
    return 0.0


def _compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute bearing from point 1 to point 2 in degrees."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2_r - lon1_r
    
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


# =========================================================
# Constellation Pattern Matching
# =========================================================

def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to unit scale centered at origin."""
    coords = np.array(coords, dtype=float)
    if len(coords) < 2:
        return coords
    
    centered = coords - coords.mean(axis=0)
    scale = np.sqrt((centered ** 2).sum() / len(coords))
    
    if scale > 0:
        return centered / scale
    return centered


def match_pattern(node_coords: List[Tuple[float, float]], 
                  constellation_coords: List[Tuple[float, float]]) -> float:
    """
    Match node positions to constellation star positions using Procrustes analysis.
    
    Args:
        node_coords: List of (lat, lon) tuples for nodes
        constellation_coords: List of (dec, ra_scaled) tuples for constellation stars
    
    Returns:
        Similarity score 0-1 (1 = perfect match)
    """
    if len(node_coords) != len(constellation_coords):
        return 0.0
    
    if len(node_coords) < 3:
        return 0.0
    
    try:
        from scipy.spatial import procrustes
        
        nodes_norm = normalize_coordinates(np.array(node_coords))
        stars_norm = normalize_coordinates(np.array(constellation_coords))
        
        # Procrustes analysis finds optimal rotation/scaling
        _, _, disparity = procrustes(nodes_norm, stars_norm)
        
        # Convert disparity to similarity
        similarity = max(0, 1 - disparity)
        return similarity
        
    except ImportError:
        # Fallback without scipy.spatial.procrustes
        return _simple_pattern_match(node_coords, constellation_coords)
    except Exception:
        return 0.0


def _simple_pattern_match(coords1: List, coords2: List) -> float:
    """Simple pattern matching without Procrustes."""
    if len(coords1) != len(coords2):
        return 0.0
    
    # Normalize both
    c1 = normalize_coordinates(np.array(coords1))
    c2 = normalize_coordinates(np.array(coords2))
    
    # Compute pairwise distances in each pattern
    def pairwise_dists(coords):
        n = len(coords)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(((coords[i] - coords[j]) ** 2).sum())
                dists.append(d)
        return sorted(dists)
    
    d1 = pairwise_dists(c1)
    d2 = pairwise_dists(c2)
    
    if not d1:
        return 0.0
    
    # Compare distance distributions
    diff = sum(abs(a - b) for a, b in zip(d1, d2)) / len(d1)
    return max(0, 1 - diff)


def find_constellation_matches(nodes: List[Dict], min_stars: int = 3, 
                               threshold: float = 0.6) -> List[Dict]:
    """
    Find groups of nodes that match constellation patterns.
    
    Args:
        nodes: List of node dicts with 'lat', 'lon' keys
        min_stars: Minimum number of stars/nodes to match
        threshold: Minimum similarity score to report
    
    Returns:
        List of match dicts sorted by score (descending)
    """
    matches = []
    
    for const_name, const_data in CONSTELLATIONS.items():
        stars = const_data["stars"]
        
        if len(stars) < min_stars:
            continue
        
        # Get constellation coordinates (dec, ra/15 to approximate equal scaling)
        const_coords = [(s["dec"], s["ra"] / 15) for s in stars]
        
        # If we have fewer nodes than stars, skip
        if len(nodes) < len(stars):
            continue
        
        # Try matching subsets of nodes to this constellation
        # Limit combinations for performance
        node_list = list(nodes)
        max_combinations = 1000
        
        for node_subset in combinations(node_list, len(stars)):
            max_combinations -= 1
            if max_combinations <= 0:
                break
            
            node_coords = [(n["lat"], n["lon"]) for n in node_subset]
            score = match_pattern(node_coords, const_coords)
            
            if score >= threshold:
                matches.append({
                    "constellation": const_name,
                    "constellation_name": const_data["name"],
                    "score": score,
                    "nodes": node_subset,
                    "sacred": const_data.get("sacred", False),
                })
    
    return sorted(matches, key=lambda x: -x["score"])


# =========================================================
# A Score (Astronomical Alignment Score)
# =========================================================

def compute_A_score(lat: float, lon: float, nodes: List[Dict] = None,
                   weights: Dict[str, float] = None) -> float:
    """
    Compute astronomical alignment score for a location.
    
    Components:
    - pattern: How well nearby nodes match constellation patterns
    - visibility: Visibility of sacred constellations (Orion, Pleiades, etc.)
    - alignment: Alignment with solstice/equinox sunrise/sunset directions
    
    Args:
        lat, lon: Location coordinates
        nodes: List of known nodes (for pattern matching)
        weights: Component weights (default: pattern=0.4, visibility=0.3, alignment=0.3)
    
    Returns:
        A score between 0 and 1
    """
    if weights is None:
        weights = {"pattern": 0.3, "visibility": 0.4, "alignment": 0.3}
    
    # 1. Pattern matching score
    pattern_score = 0.0
    if nodes and len(nodes) >= 3:
        matches = find_constellation_matches(nodes, min_stars=3, threshold=0.5)
        if matches:
            # Weight by whether it's a sacred constellation
            for m in matches[:3]:  # Top 3 matches
                if m["sacred"]:
                    pattern_score = max(pattern_score, m["score"])
                else:
                    pattern_score = max(pattern_score, m["score"] * 0.7)
    
    # 2. Visibility of sacred constellations
    visibility = get_constellation_visibility(lat, lon)
    sacred_consts = ["orion", "orion_belt", "pleiades", "ursa_major", "draco", "cygnus"]
    visible_scores = [visibility.get(c, 0) for c in sacred_consts]
    visibility_score = np.mean(visible_scores) if visible_scores else 0.5
    
    # 3. Solstice alignment score
    alignment_score = compute_solstice_alignment_score(lat, lon, nodes)
    
    # Combine scores
    A = (weights["pattern"] * pattern_score +
         weights["visibility"] * visibility_score +
         weights["alignment"] * alignment_score)
    
    return float(np.clip(A, 0, 1))


def compute_A_breakdown(lat: float, lon: float, nodes: List[Dict] = None) -> Dict:
    """
    Compute A score with detailed breakdown of components.
    
    Returns:
        Dict with 'A', 'pattern', 'visibility', 'alignment', and 'details'
    """
    # Pattern matching
    pattern_score = 0.0
    pattern_matches = []
    if nodes and len(nodes) >= 3:
        matches = find_constellation_matches(nodes, min_stars=3, threshold=0.5)
        pattern_matches = matches[:5]
        if matches:
            for m in matches[:3]:
                if m["sacred"]:
                    pattern_score = max(pattern_score, m["score"])
                else:
                    pattern_score = max(pattern_score, m["score"] * 0.7)
    
    # Visibility
    visibility = get_constellation_visibility(lat, lon)
    sacred_consts = ["orion", "orion_belt", "pleiades", "ursa_major", "draco", "cygnus"]
    visible_scores = {c: visibility.get(c, 0) for c in sacred_consts}
    visibility_score = np.mean(list(visible_scores.values()))
    
    # Alignment
    alignment_score = compute_solstice_alignment_score(lat, lon, nodes)
    solstice_azimuths = {
        "summer_solstice": compute_solstice_azimuth(lat, lon, "summer_solstice"),
        "winter_solstice": compute_solstice_azimuth(lat, lon, "winter_solstice"),
    }
    
    # Combined A score
    A = 0.3 * pattern_score + 0.4 * visibility_score + 0.3 * alignment_score
    
    return {
        "A": float(np.clip(A, 0, 1)),
        "pattern_score": pattern_score,
        "visibility_score": visibility_score,
        "alignment_score": alignment_score,
        "details": {
            "pattern_matches": pattern_matches,
            "constellation_visibility": visible_scores,
            "solstice_azimuths": solstice_azimuths,
        }
    }


# =========================================================
# Utility Functions
# =========================================================

def get_constellation_stars(name: str) -> List[Dict]:
    """Get star data for a constellation."""
    if name in CONSTELLATIONS:
        return CONSTELLATIONS[name]["stars"]
    return []


def get_sacred_constellations() -> List[str]:
    """Get list of constellations marked as sacred."""
    return [name for name, data in CONSTELLATIONS.items() if data.get("sacred", False)]


def get_all_constellation_names() -> List[str]:
    """Get all constellation names."""
    return list(CONSTELLATIONS.keys())

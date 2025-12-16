#!/usr/bin/env python
"""
GSN Advanced Geometric Analysis Module

Provides geometric analysis tools for detecting patterns in the spatial
distribution of ancient sites:

1. Great Circle Alignment Detection
2. Spherical Delaunay Triangulation
3. Spherical Voronoi Cell Analysis
4. Golden Ratio Distance Relationships

These geometric patterns can reveal hidden relationships between sites
that traditional angle-based scoring may miss.
"""

import math
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np

try:
    from scipy.spatial import SphericalVoronoi, ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio: 1.618033988749895
EARTH_RADIUS_KM = 6371.0

# Extended phi-related ratios
PHI_RATIOS = [
    PHI,        # 1.618
    PHI ** 2,   # 2.618
    PHI ** 3,   # 4.236
    1 / PHI,    # 0.618
    1 / PHI ** 2,  # 0.382
    2 * PHI,    # 3.236
    PHI + 1,    # 2.618 (same as phi^2)
]

# Fibonacci sequence for distance patterns
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


# ---------------------------------------------------------
# Coordinate Conversion Utilities
# ---------------------------------------------------------

def latlon_to_xyz(lat: float, lon: float) -> np.ndarray:
    """Convert latitude/longitude to 3D unit vector."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.array([x, y, z])


def xyz_to_latlon(xyz: np.ndarray) -> Tuple[float, float]:
    """Convert 3D unit vector to latitude/longitude."""
    x, y, z = xyz / np.linalg.norm(xyz)
    
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    
    return lat, lon


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def angular_distance_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angular distance between two unit vectors in degrees."""
    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ---------------------------------------------------------
# 1. Great Circle Alignment Detection
# ---------------------------------------------------------

def compute_great_circle_pole(coord1: Tuple[float, float], 
                               coord2: Tuple[float, float]) -> np.ndarray:
    """
    Compute the pole (normal vector) of the great circle passing through two points.
    
    The pole is perpendicular to the plane containing both points and Earth's center.
    """
    p1 = latlon_to_xyz(coord1[0], coord1[1])
    p2 = latlon_to_xyz(coord2[0], coord2[1])
    
    # Cross product gives normal to the plane
    pole = np.cross(p1, p2)
    norm = np.linalg.norm(pole)
    
    if norm < 1e-10:
        # Points are antipodal or identical - undefined great circle
        return None
    
    return pole / norm


def great_circle_distance_to_pole(coord: Tuple[float, float], 
                                   pole: np.ndarray) -> float:
    """
    Compute angular distance from a point to the plane defined by a pole.
    
    Returns angle in degrees. Points ON the great circle have distance = 90°.
    """
    if pole is None:
        return 90.0
    
    p = latlon_to_xyz(coord[0], coord[1])
    dot = np.dot(p, pole)
    
    # Distance from plane is arcsin(dot), we want distance from great circle
    # Great circle is at 90° from pole
    angle_from_pole = np.degrees(np.arcsin(np.clip(dot, -1.0, 1.0)))
    
    return 90.0 - abs(angle_from_pole)


def find_great_circle_alignments(
    nodes: Dict,
    min_sites: int = 3,
    tolerance_deg: float = 1.5,
) -> List[Dict]:
    """
    Find great circles that pass through multiple known sites.
    
    Args:
        nodes: Dict of known sites with coordinates
        min_sites: Minimum number of sites for a valid alignment
        tolerance_deg: Angular tolerance for considering a site "on" the circle
    
    Returns:
        List of alignment dicts with pole, sites list, and strength
    """
    # Extract coordinates
    coords = []
    for name, data in nodes.items():
        if isinstance(data, dict):
            lat, lon = data.get("coords", (0, 0))
        else:
            lat, lon = data
        coords.append((name, (lat, lon)))
    
    if len(coords) < min_sites:
        return []
    
    alignments = []
    seen_poles = []
    
    # Check all pairs of sites
    for (n1, c1), (n2, c2) in combinations(coords, 2):
        pole = compute_great_circle_pole(c1, c2)
        
        if pole is None:
            continue
        
        # Check if we've already found this alignment (within tolerance)
        is_duplicate = False
        for existing_pole, _ in seen_poles:
            if angular_distance_deg(pole, existing_pole) < tolerance_deg:
                is_duplicate = True
                break
            # Also check antipodal pole
            if angular_distance_deg(pole, -existing_pole) < tolerance_deg:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
        
        # Count sites near this great circle
        aligned_sites = []
        for n3, c3 in coords:
            dist = great_circle_distance_to_pole(c3, pole)
            if dist < tolerance_deg:
                aligned_sites.append(n3)
        
        if len(aligned_sites) >= min_sites:
            alignments.append({
                "pole": pole,
                "sites": aligned_sites,
                "strength": len(aligned_sites),
            })
            seen_poles.append((pole, aligned_sites))
    
    # Sort by strength (most sites first)
    alignments.sort(key=lambda a: -a["strength"])
    
    return alignments


def alignment_score(lat: float, lon: float, 
                    alignments: List[Dict], 
                    sigma: float = 2.0) -> float:
    """
    Score a location by proximity to detected great circle alignments.
    
    Args:
        lat, lon: Query coordinates
        alignments: List of alignment dicts from find_great_circle_alignments
        sigma: Gaussian decay parameter in degrees
    
    Returns:
        Alignment score (0-1)
    """
    if not alignments:
        return 0.0
    
    coord = (lat, lon)
    scores = []
    
    for align in alignments:
        pole = align["pole"]
        dist = great_circle_distance_to_pole(coord, pole)
        
        # Gaussian scoring: closer to great circle = higher score
        score = np.exp(-(dist**2) / (2 * sigma**2))
        
        # Weight by alignment strength (normalized)
        strength_weight = min(align["strength"] / 10.0, 1.0)
        score *= strength_weight
        
        scores.append(score)
    
    # Return maximum score (best alignment)
    return max(scores) if scores else 0.0


# ---------------------------------------------------------
# Cached Alignment Data
# ---------------------------------------------------------

_alignment_cache = None


def get_cached_alignments(nodes: Dict = None, min_sites: int = 3) -> List[Dict]:
    """Get or compute cached alignments."""
    global _alignment_cache
    
    if _alignment_cache is None and nodes is not None:
        _alignment_cache = find_great_circle_alignments(nodes, min_sites=min_sites)
        print(f"[INFO] Found {len(_alignment_cache)} great circle alignments")
    
    return _alignment_cache or []


def clear_alignment_cache():
    """Clear the alignment cache."""
    global _alignment_cache
    _alignment_cache = None


# ---------------------------------------------------------
# 2. Spherical Delaunay Triangulation
# ---------------------------------------------------------

_voronoi_cache = None


def compute_spherical_voronoi(nodes: Dict) -> Optional['SphericalVoronoi']:
    """
    Compute Spherical Voronoi tessellation of known sites.
    
    The Delaunay triangulation is the dual of the Voronoi diagram.
    """
    global _voronoi_cache
    
    if _voronoi_cache is not None:
        return _voronoi_cache
    
    if not HAS_SCIPY:
        print("[WARN] scipy required for Voronoi analysis")
        return None
    
    # Convert nodes to 3D points
    points = []
    for data in nodes.values():
        if isinstance(data, dict):
            lat, lon = data.get("coords", (0, 0))
        else:
            lat, lon = data
        points.append(latlon_to_xyz(lat, lon))
    
    if len(points) < 4:
        print("[WARN] Need at least 4 points for spherical Voronoi")
        return None
    
    points = np.array(points)
    
    try:
        sv = SphericalVoronoi(points, radius=1, center=np.zeros(3))
        sv.sort_vertices_of_regions()
        _voronoi_cache = sv
        print(f"[INFO] Computed spherical Voronoi with {len(sv.vertices)} vertices")
        return sv
    except Exception as e:
        print(f"[ERROR] Spherical Voronoi failed: {e}")
        return None


def get_delaunay_triangles(sv: 'SphericalVoronoi') -> List[Tuple[int, int, int]]:
    """
    Extract Delaunay triangles from Voronoi tessellation.
    
    Each Voronoi vertex corresponds to a Delaunay triangle.
    """
    if sv is None:
        return []
    
    triangles = []
    
    # Each Voronoi vertex is equidistant from 3+ input points
    # Those points form a Delaunay triangle
    n_points = len(sv.points)
    
    for vertex_idx, vertex in enumerate(sv.vertices):
        # Find the 3 closest input points to this vertex
        distances = [angular_distance_deg(vertex, p) for p in sv.points]
        sorted_indices = np.argsort(distances)
        
        # The 3 closest should be equidistant (they form the triangle)
        tri = tuple(sorted(sorted_indices[:3]))
        if tri not in triangles:
            triangles.append(tri)
    
    return triangles


def triangle_quality_score(lat: float, lon: float, 
                           nodes: Dict,
                           sv: 'SphericalVoronoi' = None) -> float:
    """
    Score how well a point fits into the Delaunay triangulation.
    
    Higher scores for:
    - Points at triangle centroids
    - Points that would create equilateral triangles
    - Points that would create golden-ratio triangles
    """
    if sv is None:
        sv = compute_spherical_voronoi(nodes)
    
    if sv is None:
        return 0.0
    
    query = latlon_to_xyz(lat, lon)
    
    # Find distance to nearest Voronoi vertex (triangle circumcenter)
    min_vertex_dist = float('inf')
    for vertex in sv.vertices:
        dist = angular_distance_deg(query, vertex)
        min_vertex_dist = min(min_vertex_dist, dist)
    
    # Score based on proximity to circumcenters
    # Being at a circumcenter means equidistant from 3 sites
    circumcenter_score = np.exp(-min_vertex_dist / 10.0)
    
    # Find containing triangle and evaluate quality
    triangles = get_delaunay_triangles(sv)
    
    best_triangle_score = 0.0
    node_list = list(nodes.values())
    
    for tri in triangles:
        # Get triangle vertices
        tri_points = [sv.points[i] for i in tri]
        
        # Check if query point is "inside" this triangle (on sphere)
        # Simplified: check if it's close to the centroid
        centroid = np.mean(tri_points, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Project to sphere
        
        dist_to_centroid = angular_distance_deg(query, centroid)
        
        if dist_to_centroid < 30:  # Within reasonable range
            # Compute triangle edge lengths
            edges = []
            for i in range(3):
                p1, p2 = tri_points[i], tri_points[(i+1) % 3]
                edge_len = angular_distance_deg(p1, p2)
                edges.append(edge_len)
            
            # Equilateral score: all edges similar length
            edge_std = np.std(edges) / max(np.mean(edges), 1)
            equilateral_score = np.exp(-edge_std * 5)
            
            # Golden ratio score: edge ratios near phi
            edges_sorted = sorted(edges)
            if edges_sorted[0] > 1:
                ratios = [edges_sorted[i+1] / edges_sorted[i] for i in range(2)]
                phi_matches = sum(
                    np.exp(-abs(r - PHI) * 5) for r in ratios
                )
                golden_score = phi_matches / 2
            else:
                golden_score = 0
            
            triangle_score = 0.5 * equilateral_score + 0.5 * golden_score
            triangle_score *= np.exp(-dist_to_centroid / 20)
            
            best_triangle_score = max(best_triangle_score, triangle_score)
    
    return 0.5 * circumcenter_score + 0.5 * best_triangle_score


# ---------------------------------------------------------
# 3. Spherical Voronoi Cell Analysis
# ---------------------------------------------------------

def voronoi_boundary_score(lat: float, lon: float,
                           nodes: Dict,
                           sv: 'SphericalVoronoi' = None) -> float:
    """
    Score proximity to Voronoi cell boundaries.
    
    High scores for points equidistant from 3+ known sites
    (Voronoi vertices - natural candidates for new nodes).
    """
    if sv is None:
        sv = compute_spherical_voronoi(nodes)
    
    if sv is None:
        return 0.0
    
    query = latlon_to_xyz(lat, lon)
    
    # Find distance to nearest Voronoi vertex
    min_vertex_dist = float('inf')
    for vertex in sv.vertices:
        dist = np.linalg.norm(query - vertex)
        min_vertex_dist = min(min_vertex_dist, dist)
    
    # Convert chord distance to angular distance
    vertex_angle = 2 * np.degrees(np.arcsin(min_vertex_dist / 2))
    
    # Score: closer to vertex = higher score
    vertex_score = np.exp(-vertex_angle / 5.0)
    
    # Also score proximity to edges (2-site boundaries)
    min_edge_dist = float('inf')
    
    for region in sv.regions:
        if len(region) < 2:
            continue
        
        for i in range(len(region)):
            v1 = sv.vertices[region[i]]
            v2 = sv.vertices[region[(i+1) % len(region)]]
            
            # Distance from query to line segment v1-v2
            # Simplified: distance to midpoint
            midpoint = (v1 + v2) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)
            dist = np.linalg.norm(query - midpoint)
            min_edge_dist = min(min_edge_dist, dist)
    
    edge_angle = 2 * np.degrees(np.arcsin(min(min_edge_dist / 2, 1.0)))
    edge_score = np.exp(-edge_angle / 10.0)
    
    # Vertices (3+ site boundaries) are more important than edges
    return 0.7 * vertex_score + 0.3 * edge_score


def voronoi_cell_regularity(nodes: Dict) -> float:
    """
    Measure regularity of Voronoi tessellation.
    
    A highly regular tessellation (low variance in cell areas) suggests
    the sites follow a systematic pattern.
    """
    sv = compute_spherical_voronoi(nodes)
    
    if sv is None:
        return 0.5  # Neutral score
    
    # Compute approximate cell areas using solid angles
    areas = []
    
    for region in sv.regions:
        if len(region) < 3:
            continue
        
        # Compute spherical polygon area
        vertices = [sv.vertices[i] for i in region]
        area = compute_spherical_polygon_area(vertices)
        areas.append(area)
    
    if not areas:
        return 0.5
    
    # Coefficient of variation: std/mean
    cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 1.0
    
    # Low CV = regular, high CV = irregular
    # Return score: higher = more regular
    return np.exp(-cv)


def compute_spherical_polygon_area(vertices: List[np.ndarray]) -> float:
    """
    Compute the area of a spherical polygon on a unit sphere.
    Uses the spherical excess formula.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    # Sum of interior angles
    angle_sum = 0.0
    
    for i in range(n):
        p1 = vertices[(i - 1) % n]
        p2 = vertices[i]
        p3 = vertices[(i + 1) % n]
        
        # Vectors from p2 to neighbors
        v1 = p1 - p2
        v3 = p3 - p2
        
        # Project onto tangent plane
        v1 = v1 - np.dot(v1, p2) * p2
        v3 = v3 - np.dot(v3, p2) * p2
        
        # Angle between
        norm1, norm3 = np.linalg.norm(v1), np.linalg.norm(v3)
        if norm1 > 1e-10 and norm3 > 1e-10:
            cos_angle = np.clip(np.dot(v1, v3) / (norm1 * norm3), -1, 1)
            angle_sum += np.arccos(cos_angle)
    
    # Spherical excess = sum of angles - (n-2)*pi
    excess = angle_sum - (n - 2) * np.pi
    
    return max(excess, 0.0)


# ---------------------------------------------------------
# 4. Golden Ratio Distance Relationships
# ---------------------------------------------------------

def golden_ratio_score(lat: float, lon: float, 
                       nodes: Dict,
                       tolerance: float = 0.08) -> float:
    """
    Score how well distances to known sites follow golden ratio relationships.
    
    Checks for distance ratios that match phi, phi^2, 1/phi, etc.
    """
    # Compute distances to all nodes
    distances = []
    for data in nodes.values():
        if isinstance(data, dict):
            nlat, nlon = data.get("coords", (0, 0))
        else:
            nlat, nlon = data
        
        d = haversine_km(lat, lon, nlat, nlon)
        if d > 10:  # Ignore very close sites
            distances.append(d)
    
    if len(distances) < 2:
        return 0.0
    
    distances.sort()
    
    # Count phi-ratio matches
    matches = 0
    total_pairs = 0
    
    for i in range(len(distances) - 1):
        for j in range(i + 1, min(i + 10, len(distances))):  # Limit comparisons
            d1, d2 = distances[i], distances[j]
            
            if d1 > 0:
                ratio = d2 / d1
                
                # Check against all phi-related ratios
                for target in PHI_RATIOS:
                    if abs(ratio - target) < tolerance * target:
                        matches += 1
                        break
                
                total_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    return min(matches / max(total_pairs, 1) * 2, 1.0)


def fibonacci_distance_score(lat: float, lon: float,
                             nodes: Dict,
                             base_km: float = 100.0,
                             tolerance: float = 0.1) -> float:
    """
    Score how well distances follow Fibonacci sequence pattern.
    
    Checks if distances are close to base_km * fibonacci_number.
    """
    distances = []
    for data in nodes.values():
        if isinstance(data, dict):
            nlat, nlon = data.get("coords", (0, 0))
        else:
            nlat, nlon = data
        
        d = haversine_km(lat, lon, nlat, nlon)
        if d > 10:
            distances.append(d)
    
    if not distances:
        return 0.0
    
    matches = 0
    for d in distances:
        for fib in FIBONACCI:
            target = base_km * fib
            if abs(d - target) < tolerance * target:
                matches += 1
                break
    
    return min(matches / len(distances) * 2, 1.0)


# ---------------------------------------------------------
# 5. Combined Geometric H Score
# ---------------------------------------------------------

def compute_H_geometric(lat: float, lon: float, 
                        nodes: Dict = None,
                        weights: Dict = None) -> float:
    """
    Compute enhanced H score combining all geometric analyses.
    
    Args:
        lat, lon: Query coordinates
        nodes: Dict of known sites
        weights: Optional custom weights for components
    
    Returns:
        Combined geometric H score (0-1)
    """
    if nodes is None:
        # Try to import from known_nodes_extended
        try:
            from known_nodes_extended import KNOWN_NODES_EXTENDED
            nodes = KNOWN_NODES_EXTENDED
        except ImportError:
            try:
                from gsn_node_predictor import KNOWN_NODES
                nodes = KNOWN_NODES
            except ImportError:
                return 0.0
    
    if not nodes:
        return 0.0
    
    # Default weights
    if weights is None:
        weights = {
            "alignment": 0.25,
            "voronoi": 0.25,
            "golden": 0.25,
            "triangle": 0.25,
        }
    
    # Compute components
    alignments = get_cached_alignments(nodes)
    sv = compute_spherical_voronoi(nodes)
    
    H_align = alignment_score(lat, lon, alignments)
    H_voronoi = voronoi_boundary_score(lat, lon, nodes, sv)
    H_golden = golden_ratio_score(lat, lon, nodes)
    H_tri = triangle_quality_score(lat, lon, nodes, sv)
    
    # Weighted combination
    H = (weights.get("alignment", 0.25) * H_align +
         weights.get("voronoi", 0.25) * H_voronoi +
         weights.get("golden", 0.25) * H_golden +
         weights.get("triangle", 0.25) * H_tri)
    
    return H


def compute_H_all_geometric(lat: float, lon: float, nodes: Dict = None) -> Dict:
    """
    Compute all geometric H components separately for analysis.
    
    Returns:
        Dict with individual scores for each geometric component
    """
    if nodes is None:
        try:
            from known_nodes_extended import KNOWN_NODES_EXTENDED
            nodes = KNOWN_NODES_EXTENDED
        except ImportError:
            nodes = {}
    
    alignments = get_cached_alignments(nodes)
    sv = compute_spherical_voronoi(nodes)
    
    return {
        "H_alignment": alignment_score(lat, lon, alignments),
        "H_voronoi": voronoi_boundary_score(lat, lon, nodes, sv),
        "H_golden": golden_ratio_score(lat, lon, nodes),
        "H_triangle": triangle_quality_score(lat, lon, nodes, sv),
        "H_fibonacci": fibonacci_distance_score(lat, lon, nodes),
        "H_geometric": compute_H_geometric(lat, lon, nodes),
    }


# ---------------------------------------------------------
# Statistics and Analysis
# ---------------------------------------------------------

def get_geometry_stats(nodes: Dict = None) -> Dict:
    """Get statistics about geometric patterns in the node distribution."""
    if nodes is None:
        try:
            from known_nodes_extended import KNOWN_NODES_EXTENDED
            nodes = KNOWN_NODES_EXTENDED
        except ImportError:
            return {"available": False}
    
    alignments = find_great_circle_alignments(nodes, min_sites=3)
    sv = compute_spherical_voronoi(nodes)
    
    return {
        "available": True,
        "n_nodes": len(nodes),
        "n_alignments_3plus": len([a for a in alignments if a["strength"] >= 3]),
        "n_alignments_4plus": len([a for a in alignments if a["strength"] >= 4]),
        "n_alignments_5plus": len([a for a in alignments if a["strength"] >= 5]),
        "max_alignment_strength": max([a["strength"] for a in alignments]) if alignments else 0,
        "voronoi_regularity": voronoi_cell_regularity(nodes),
        "n_voronoi_vertices": len(sv.vertices) if sv else 0,
    }


# ---------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------

if __name__ == "__main__":
    print("GSN Geometric Analysis Module Demo")
    print("-" * 40)
    
    # Try to load nodes
    try:
        from known_nodes_extended import KNOWN_NODES_EXTENDED as nodes
        print(f"Loaded {len(nodes)} extended nodes")
    except ImportError:
        try:
            from gsn_node_predictor import KNOWN_NODES as nodes
            print(f"Loaded {len(nodes)} basic nodes")
        except ImportError:
            print("No nodes available for demo")
            nodes = {}
    
    if nodes:
        # Compute statistics
        stats = get_geometry_stats(nodes)
        print("\nGeometric Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Show top alignments
        alignments = find_great_circle_alignments(nodes, min_sites=4)
        print(f"\nTop Great Circle Alignments (4+ sites):")
        for i, align in enumerate(alignments[:5]):
            print(f"  {i+1}. {align['strength']} sites: {', '.join(align['sites'][:5])}...")
        
        # Test scoring at known locations
        test_points = [
            (29.9792, 31.1342, "Giza"),
            (51.1789, -1.8262, "Stonehenge"),
            (37.2231, 38.9225, "Gobekli Tepe"),
            (40.7128, -74.0060, "New York"),
        ]
        
        print("\nGeometric H scores:")
        for lat, lon, name in test_points:
            scores = compute_H_all_geometric(lat, lon, nodes)
            print(f"\n  {name}:")
            for k, v in scores.items():
                print(f"    {k}: {v:.3f}")

#!/usr/bin/env python
"""
GSN Node Predictor - Streamlit Web Application (Enhanced)

A web-based interface for the Global Stabilization Network node prediction tool.
Features interactive maps with OpenStreetMap/Leaflet, heatmap overlays,
on-demand scanning, extended data sources, and uncertainty quantification.

Run with: streamlit run gsn_web_app.py
"""

import streamlit as st
import numpy as np
import folium
from folium.plugins import (
    HeatMap,
    Fullscreen,
    MiniMap,
    Draw,
    MarkerCluster,
    LocateControl,
    Geocoder
)
from streamlit_folium import st_folium

# Import core functions from the existing predictor module
import gsn_node_predictor as gsn

# Import enhanced modules
try:
    from known_nodes_extended import KNOWN_NODES_EXTENDED, get_coords_dict, CATEGORIES
    HAS_EXTENDED_NODES = True
except ImportError:
    HAS_EXTENDED_NODES = False

try:
    from gsn_data_sources import (
        check_data_availability, compute_G_extended, G_COMPONENTS,
        get_all_components
    )
    HAS_DATA_SOURCES = True
except ImportError:
    HAS_DATA_SOURCES = False

try:
    from gsn_validation import (
        compute_recall_at_k, generate_validation_report,
        compute_average_distance_to_nearest, geographic_stratified_cv,
        spatial_buffer_cv, GEOGRAPHIC_REGIONS
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

try:
    from gsn_seismic import get_seismic_density, compute_seismic_density_grid
    HAS_SEISMIC = True
except ImportError:
    HAS_SEISMIC = False

try:
    from gsn_uncertainty import (
        bootstrap_F_score, classify_confidence, compute_data_quality_score,
        bootstrap_H_confidence, propagate_G_uncertainty, monte_carlo_F,
        ConfidenceInterval, UncertaintyResult
    )
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False

try:
    from gsn_statistical_validation import (
        test_pattern_significance, ripley_k_spherical, alignment_significance,
        run_full_validation, CSRTestResult, ValidationSummary
    )
    HAS_STATISTICAL_VALIDATION = True
except ImportError:
    HAS_STATISTICAL_VALIDATION = False

try:
    from gsn_bayesian_scorer import (
        BayesianScorer, CombinedBayesianScorer, PosteriorResult
    )
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

try:
    from gsn_heatflow import get_heatflow, get_heatflow_stats
    HAS_HEATFLOW = True
except ImportError:
    HAS_HEATFLOW = False

try:
    from gsn_volcanic import get_distance_to_nearest_volcano, get_volcano_stats
    HAS_VOLCANIC = True
except ImportError:
    HAS_VOLCANIC = False

try:
    from gsn_archaeology import get_site_stats, get_nearby_sites, SITE_WEIGHTS
    HAS_ARCHAEOLOGY = True
except ImportError:
    HAS_ARCHAEOLOGY = False

try:
    from gsn_stress_map import is_available as wsm_available, get_wsm_stats
    HAS_STRESS_MAP = wsm_available()
except ImportError:
    HAS_STRESS_MAP = False

try:
    from gsn_geometry import (
        compute_H_geometric, compute_H_all_geometric, get_geometry_stats,
        find_great_circle_alignments
    )
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False

# Try to import astronomy module
try:
    from gsn_astronomy import (
        compute_A_score, compute_A_breakdown, get_constellation_visibility,
        get_visible_constellations, find_constellation_matches,
        CONSTELLATIONS, get_sacred_constellations
    )
    HAS_ASTRONOMY = True
except ImportError:
    HAS_ASTRONOMY = False

# Try to import network module
try:
    from gsn_network import (
        GSNNetwork, get_network, compute_N_score, get_network_stats,
        HAS_NETWORKX
    )
    HAS_NETWORK = HAS_NETWORKX
except ImportError:
    HAS_NETWORK = False

# Try to import temporal module
try:
    from gsn_temporal import (
        TemporalAnalyzer, get_analyzer, compute_T_score,
        get_epochs, get_epoch_alignment, EPOCHS
    )
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

# Try to import ML grid scorer module
try:
    from gsn_ml_grid_scorer import (
        MLGridScorer, TrainingConfig, GRID_FEATURE_NAMES,
        train_grid_scorer_from_nodes, HAS_TORCH
    )
    HAS_ML_SCORER = HAS_TORCH
except ImportError:
    HAS_ML_SCORER = False

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="GSN Node Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Configuration Presets
# ---------------------------------------------------------
PRESETS = {
    "Fast Scan": {
        "description": "Quick overview at coarse resolution",
        "step_deg": 10.0,
        "top_n": 20,
        "min_sep": 15.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.0,
        "h_method": "weighted",
        "use_geometry": False,
        "use_extended_data": False,
        "use_astronomy": False,
    },
    "Balanced": {
        "description": "Good balance of speed and accuracy",
        "step_deg": 5.0,
        "top_n": 50,
        "min_sep": 10.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.0,
        "h_method": "combined",
        "use_geometry": True,
        "use_extended_data": True,
        "use_astronomy": False,
    },
    "Accurate": {
        "description": "High accuracy with all features enabled",
        "step_deg": 2.0,
        "top_n": 50,
        "min_sep": 5.0,
        "alpha": 1.0,
        "beta": 1.2,
        "gamma": 0.5,
        "h_method": "full",
        "use_geometry": True,
        "use_extended_data": True,
        "use_astronomy": True,
    },
    "Geometric Focus": {
        "description": "Emphasize geometric patterns (alignments, Voronoi)",
        "step_deg": 5.0,
        "top_n": 40,
        "min_sep": 8.0,
        "alpha": 0.8,
        "beta": 1.5,
        "gamma": 0.0,
        "h_method": "geometric",
        "use_geometry": True,
        "use_extended_data": False,
        "use_astronomy": False,
    },
    "Astronomical Focus": {
        "description": "Include celestial alignments and constellation patterns",
        "step_deg": 5.0,
        "top_n": 40,
        "min_sep": 8.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "h_method": "combined",
        "use_geometry": True,
        "use_extended_data": False,
        "use_astronomy": True,
    },
    "Geophysical Focus": {
        "description": "Emphasize geophysical data sources",
        "step_deg": 5.0,
        "top_n": 40,
        "min_sep": 8.0,
        "alpha": 1.5,
        "beta": 0.8,
        "gamma": 0.0,
        "h_method": "weighted",
        "use_geometry": False,
        "use_extended_data": True,
        "use_astronomy": False,
    },
    "ML Scoring": {
        "description": "Use trained neural network for scoring",
        "step_deg": 5.0,
        "top_n": 40,
        "min_sep": 8.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.0,
        "h_method": "full",
        "use_geometry": True,
        "use_extended_data": True,
        "use_astronomy": False,
        "use_ml": True,
    },
    "Custom": {
        "description": "Configure all settings manually",
        "step_deg": 5.0,
        "top_n": 50,
        "min_sep": 8.0,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.0,
        "h_method": "full",
        "use_geometry": True,
        "use_extended_data": True,
        "use_astronomy": False,
    },
}

H_METHODS = {
    "basic": "Basic (original Penrose angles)",
    "weighted": "Weighted (distance-weighted, extended angles)",
    "combined": "Combined (weighted + archaeological)",
    "full": "Full (all methods combined)",
    "geometric": "Geometric (alignments, Voronoi, golden ratio)",
}

# Layer groups for simplified UI
LAYER_GROUPS = {
    "Base": {
        "icon": "🗺️",
        "description": "Known nodes and plate boundaries",
        "includes": ["known_nodes", "plate_boundaries"],
        "default": True,
    },
    "Geophysical": {
        "icon": "🌋",
        "description": "Volcanic, seismic, heat flow data",
        "includes": ["volcanoes", "heatflow", "seismic"],
        "default": False,
        "requires": [HAS_VOLCANIC, HAS_HEATFLOW, HAS_SEISMIC],
    },
    "Geometric": {
        "icon": "📐",
        "description": "Alignments, patterns, clusters",
        "includes": ["alignments", "clusters", "voronoi"],
        "default": False,
        "requires": [HAS_GEOMETRY],
    },
    "Celestial": {
        "icon": "✨",
        "description": "Astronomical correlations",
        "includes": ["visibility", "patterns", "solstice"],
        "default": False,
        "requires": [HAS_ASTRONOMY],
    },
    "Network": {
        "icon": "🕸️",
        "description": "Network connectivity and flow",
        "includes": ["edges", "hubs", "bridges"],
        "default": False,
        "requires": [HAS_NETWORK],
    },
    "Temporal": {
        "icon": "⏳",
        "description": "Historical epoch alignments",
        "includes": ["precession", "orion_correlation"],
        "default": False,
        "requires": [HAS_TEMPORAL],
    },
}

# Modern color palette for map elements
LAYER_COLORS = {
    # Known nodes - warm red
    "nodes": "#E63946",
    "nodes_fill": "#E63946",
    
    # Candidates - gold gradient (by F score)
    "candidates_high": "#FFD700",
    "candidates_mid": "#FFA500",
    "candidates_low": "#8B4513",
    
    # Volcanoes - orange
    "volcanoes": "#F77F00",
    "volcanoes_fill": "#FCBF49",
    
    # Plate boundaries - subtle gray
    "boundaries": "#6C757D",
    
    # Alignments - purple
    "alignments": "#7209B7",
    
    # Clusters - blue tones
    "clusters": ["#3A86FF", "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
    
    # Heatmap gradient
    "heatmap": {0.2: "#3A86FF", 0.4: "#00B4D8", 0.6: "#90E0EF", 0.8: "#FFD700", 1.0: "#E63946"},
    
    # Network edges
    "network_delaunay": "#3A86FF",
    "network_golden": "#FFD700",
    "network_distance": "#00B4D8",
    "network_hub": "#FF006E",
}

# ---------------------------------------------------------
# Basemap Options
# ---------------------------------------------------------
BASEMAP_OPTIONS = {
    "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": None},
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "ESRI"
    },
    "Terrain": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "OpenTopoMap"
    },
    "Dark Mode": {"tiles": "CartoDB dark_matter", "attr": None},
    "Light Mode": {"tiles": "CartoDB positron", "attr": None},
}

# ---------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_datasets():
    """Load NetCDF datasets and plate boundaries once."""
    success = gsn.try_load_datasets()
    boundaries = gsn.load_plate_boundaries()
    # Store boundaries in the module's global
    gsn._plate_boundaries = boundaries
    return success, boundaries


def ensure_datasets_loaded():
    """Ensure datasets are loaded (call on each run)."""
    if gsn._ga_ds is None or gsn._ct_ds is None:
        gsn.try_load_datasets()
    return gsn._ga_ds is not None and gsn._ct_ds is not None


@st.cache_data
def compute_scan_results(step_deg, top_n, min_sep, alpha, beta, lat_min, lat_max, lon_min, lon_max):
    """Cached computation of top F candidates."""
    try:
        selected, stats = gsn.compute_top_F_candidates(
            step_deg=step_deg,
            top_N=top_n,
            min_sep=min_sep,
            alpha=alpha,
            beta=beta,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            return_stats=True,
        )
        return selected, stats, None
    except Exception as e:
        return [], {}, str(e)


def compute_heatmap_data(step_deg, alpha, beta, use_ml=False):
    """Compute F values on a grid for heatmap visualization."""
    try:
        # Check if datasets are loaded
        if gsn._ga_ds is None:
            return None, "Gravity dataset not loaded - please reload the page"
        
        # Get coordinates from the gravity dataset
        ga_var = gsn._ga_ds[gsn.GA_VAR_NAME]
        native_lats = ga_var.coords["lat"].values
        native_lons = ga_var.coords["lon"].values
        
        # Subsample to manageable size
        factor = max(1, int(round(step_deg / 0.033)))  # Approximate native resolution
        lats = native_lats[::factor]
        lons = native_lons[::factor]
        
        # Limit to reasonable size for browser
        if len(lats) * len(lons) > 50000:
            # Further subsample
            lat_factor = max(1, len(lats) // 200)
            lon_factor = max(1, len(lons) // 250)
            lats = lats[::lat_factor]
            lons = lons[::lon_factor]
        
        # Use ML-based scoring if enabled and available
        if use_ml and hasattr(gsn, 'compute_F_grid_ml'):
            result = gsn.compute_F_grid_ml(lats, lons, fallback_linear=True)
        else:
            result = gsn.compute_F_grid(lats, lons, alpha=alpha, beta=beta)
        
        return result, None
    except Exception as e:
        return None, str(e)


def get_constellation_visibility_summary(lat: float, lon: float) -> list:
    """
    Get human-friendly constellation visibility summary for a location.
    
    Returns list of dicts with constellation info, visibility score, and status.
    """
    if not HAS_ASTRONOMY:
        return []
    
    try:
        from gsn_astronomy import get_constellation_visibility, CONSTELLATIONS
        
        visibility = get_constellation_visibility(lat, lon)
        
        # Sacred constellations with display names and icons
        sacred_consts = {
            "orion": ("Orion", "⭐"),
            "orion_belt": ("Orion's Belt", "✨"),
            "pleiades": ("Pleiades", "🌟"),
            "ursa_major": ("Ursa Major", "🐻"),
            "draco": ("Draco", "🐉"),
            "cygnus": ("Cygnus", "🦢"),
        }
        
        results = []
        for const_key, (display_name, icon) in sacred_consts.items():
            score = visibility.get(const_key, 0.0)
            
            # Determine status and color
            if score >= 0.7:
                status = "Excellent"
                color = "🟢"
            elif score >= 0.4:
                status = "Good"
                color = "🟡"
            elif score > 0:
                status = "Low"
                color = "🟠"
            else:
                status = "Below horizon"
                color = "🔴"
            
            results.append({
                "name": display_name,
                "icon": icon,
                "score": score,
                "status": status,
                "color": color,
                "visible": score > 0,
            })
        
        # Sort by visibility score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
        
    except Exception as e:
        return []


def get_nearest_known_nodes(lat: float, lon: float, top_n: int = 3) -> list:
    """
    Get the nearest known GSN nodes to a location with distances.
    
    Returns list of dicts with node name, distance in km, and coordinates.
    """
    import math
    
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Get known nodes
    if HAS_EXTENDED_NODES:
        nodes = KNOWN_NODES_EXTENDED
    else:
        nodes = gsn.KNOWN_NODES
    
    # Calculate distances
    distances = []
    for name, data in nodes.items():
        if "coords" in data:
            node_lat, node_lon = data["coords"]
        else:
            node_lat, node_lon = data.get("lat", 0), data.get("lon", 0)
        
        dist = haversine_km(lat, lon, node_lat, node_lon)
        distances.append({
            "name": name.replace("_", " ").title(),
            "distance_km": dist,
            "lat": node_lat,
            "lon": node_lon,
            "category": data.get("category", "unknown"),
        })
    
    # Sort by distance and return top N
    distances.sort(key=lambda x: x["distance_km"])
    return distances[:top_n]


def get_geology_summary(lat: float, lon: float) -> dict:
    """
    Get human-friendly geology summary for a location.
    
    Returns dict with nearest volcano, boundary distance, seismic assessment.
    """
    summary = {
        "volcano_dist": None,
        "volcano_name": None,
        "boundary_dist": None,
        "seismic_level": "Unknown",
    }
    
    try:
        # Get boundary distance from core module
        _, _, boundary_dist = gsn.get_geophysical_inputs(lat, lon)
        summary["boundary_dist"] = boundary_dist
        
        # Get volcano info if available
        if HAS_VOLCANIC:
            try:
                from gsn_volcanic import get_nearest_volcano
                vol = get_nearest_volcano(lat, lon)
                if vol:
                    summary["volcano_dist"] = vol.get("distance_km")
                    summary["volcano_name"] = vol.get("name", "Unknown")
            except Exception:
                pass
        
        # Assess seismic level based on boundary distance
        if boundary_dist is not None:
            if boundary_dist < 200:
                summary["seismic_level"] = "High (near boundary)"
            elif boundary_dist < 500:
                summary["seismic_level"] = "Moderate"
            elif boundary_dist < 1000:
                summary["seismic_level"] = "Low"
            else:
                summary["seismic_level"] = "Very Low (stable)"
                
    except Exception:
        pass
    
    return summary


def evaluate_single_point(lat, lon, config=None):
    """
    Evaluate F/G/H for a single point with configurable scoring.
    
    Args:
        lat, lon: Coordinates
        config: Configuration dict with h_method, alpha, beta, etc.
    """
    if config is None:
        config = {"alpha": 1.0, "beta": 1.0, "h_method": "full"}
    
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 1.0)
    h_method = config.get("h_method", "full")
    
    try:
        # Compute H based on selected method
        H_basic = gsn.compute_H(lat, lon)
        
        if h_method == "basic":
            H = H_basic
        elif h_method == "weighted":
            if hasattr(gsn, 'compute_H_weighted'):
                H = gsn.compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True)
            else:
                H = H_basic
        elif h_method == "combined":
            if hasattr(gsn, 'compute_H_combined'):
                H = gsn.compute_H_combined(lat, lon)
            else:
                H = H_basic
        elif h_method == "geometric":
            if HAS_GEOMETRY:
                H = compute_H_geometric(lat, lon)
            else:
                H = H_basic
        else:  # "full"
            if hasattr(gsn, 'compute_H_full'):
                H = gsn.compute_H_full(lat, lon)
            elif hasattr(gsn, 'compute_H_weighted'):
                H = gsn.compute_H_weighted(lat, lon, distance_scale=30.0, use_extended=True)
            else:
                H = H_basic
        
        ga, ct, dist_km = gsn.get_geophysical_inputs(lat, lon)
        G, comps = gsn.compute_G(ga, ct, dist_km)
        F = alpha * G + beta * H
        
        # Get extended components if enabled and available
        extended_comps = {}
        if HAS_DATA_SOURCES and config.get("use_extended_data", True):
            try:
                extended_comps = get_all_components(lat, lon)
            except Exception:
                pass
        
        # Get geometric H breakdown if enabled and available
        geometric_scores = {}
        if HAS_GEOMETRY and config.get("use_geometry", True):
            try:
                geometric_scores = compute_H_all_geometric(lat, lon)
            except Exception:
                pass
        
        # Compute A (astronomical) score if Celestial layer is enabled
        gamma = config.get("gamma", 0.0)
        A = 0.0
        astronomy_details = {}
        use_astronomy = config.get("use_astronomy", False) or config.get("layers", {}).get("Celestial", False)
        
        if HAS_ASTRONOMY and use_astronomy:
            try:
                A_breakdown = compute_A_breakdown(lat, lon)
                A = A_breakdown["A"]
                astronomy_details = A_breakdown
            except Exception as e:
                astronomy_details = {"error": str(e)}
        
        # Compute N (network) score if Network layer is enabled
        delta = config.get("delta", 0.0)
        N = 0.0
        network_details = {}
        use_network = config.get("layers", {}).get("Network", False)
        
        if HAS_NETWORK and use_network:
            try:
                # Get nodes for network construction
                if HAS_EXTENDED_NODES:
                    nodes = KNOWN_NODES_EXTENDED
                else:
                    nodes = gsn.KNOWN_NODES
                N = compute_N_score(lat, lon, nodes)
                network_details = {"N": N, "available": True}
            except Exception as e:
                network_details = {"error": str(e)}
        
        # Compute T (temporal) score if Temporal layer is enabled
        epsilon = config.get("epsilon", 0.0)
        T = 0.0
        temporal_details = {}
        use_temporal = config.get("layers", {}).get("Temporal", False)
        
        if HAS_TEMPORAL and use_temporal:
            try:
                T = compute_T_score(lat, lon)
                analyzer = get_analyzer()
                temporal_details = analyzer.compute_T_breakdown(lat, lon)
            except Exception as e:
                temporal_details = {"error": str(e)}
        
        # Compute final F with all components
        F = alpha * G + beta * H
        if gamma > 0 and A > 0:
            F += gamma * A
        if delta > 0 and N > 0:
            F += delta * N
        if epsilon > 0 and T > 0:
            F += epsilon * T
        
        return {
            "F": F,
            "G": G,
            "H": H,
            "A": A,
            "N": N,
            "T": T,
            "H_basic": H_basic,
            "h_method": h_method,
            "ga": ga,
            "ct": ct,
            "dist_boundary_km": dist_km,
            "ga_norm": comps["ga_norm"],
            "ct_norm": comps["ct_norm"],
            "tb": comps["tb"],
            "extended": extended_comps,
            "geometric": geometric_scores,
            "astronomy": astronomy_details,
            "network": network_details,
            "temporal": temporal_details,
            "classification": gsn.classify_F(F),
            "region": gsn.approximate_region(lat, lon),
        }, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Map Creation Functions
# ---------------------------------------------------------
def create_base_map(center_lat=20, center_lon=0, zoom=2, basemap="OpenStreetMap",
                    enable_draw=False, enable_fullscreen=True, enable_minimap=True):
    """Create a base Folium map with configurable basemap and plugins."""
    
    basemap_config = BASEMAP_OPTIONS.get(basemap, BASEMAP_OPTIONS["OpenStreetMap"])
    
    # Handle built-in tile names vs custom URLs
    tiles = basemap_config["tiles"]
    attr = basemap_config.get("attr")
    
    if attr:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles=tiles,
            attr=attr,
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles=tiles,
        )
    
    # Add fullscreen control
    if enable_fullscreen:
        Fullscreen(position='topleft').add_to(m)
    
    # Add minimap
    if enable_minimap:
        MiniMap(toggle_display=True, position='bottomright').add_to(m)
    
    # Add drawing tools
    if enable_draw:
        Draw(
            export=True,
            position='topleft',
            draw_options={
                'rectangle': True,
                'polygon': True,
                'circle': True,
                'polyline': False,
                'marker': False,
                'circlemarker': False,
            }
        ).add_to(m)
    
    # Add geocoder search
    Geocoder(position='topright', collapsed=True).add_to(m)
    
    # Add locate control (find my location)
    LocateControl(auto_start=False, position='topleft').add_to(m)
    
    return m


def add_known_nodes(m, show_labels=True, use_extended=True):
    """Add known GSN nodes with modern styling."""
    node_color = LAYER_COLORS["nodes"]
    
    # Use extended nodes if available
    if use_extended and HAS_EXTENDED_NODES:
        nodes = KNOWN_NODES_EXTENDED
        for name, data in nodes.items():
            lat, lon = data["coords"]
            category = data.get("category", "unknown")
            age = data.get("age_bp", "unknown")
            country = data.get("country", "")
            
            popup_html = f"""
            <b>{name}</b><br>
            Category: {category}<br>
            Age: {age} BP<br>
            Country: {country}<br>
            Lat: {lat:.4f}, Lon: {lon:.4f}
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fillColor=node_color,
                fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=name if show_labels else None,
            ).add_to(m)
    else:
        # Fallback to original nodes
        for name, (lat, lon) in gsn.KNOWN_NODES.items():
            popup_html = f"<b>{name}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fillColor=node_color,
                fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=name if show_labels else None,
            ).add_to(m)
    return m


def add_known_nodes_clustered(m, show_labels=True, use_extended=True):
    """Add known GSN nodes with marker clustering for dense areas."""
    node_color = LAYER_COLORS["nodes"]
    marker_cluster = MarkerCluster(name="Known Nodes").add_to(m)
    
    if use_extended and HAS_EXTENDED_NODES:
        nodes = KNOWN_NODES_EXTENDED
        for name, data in nodes.items():
            lat, lon = data["coords"]
            category = data.get("category", "unknown")
            age = data.get("age_bp", "unknown")
            country = data.get("country", "")
            
            popup_html = f"""
            <b>{name}</b><br>
            Category: {category}<br>
            Age: {age} BP<br>
            Country: {country}<br>
            Lat: {lat:.4f}, Lon: {lon:.4f}
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=name if show_labels else None,
                icon=folium.Icon(color='red', icon='info-sign'),
            ).add_to(marker_cluster)
    else:
        for name, (lat, lon) in gsn.KNOWN_NODES.items():
            popup_html = f"<b>{name}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=name if show_labels else None,
                icon=folium.Icon(color='red', icon='info-sign'),
            ).add_to(marker_cluster)
    
    return m


def add_candidates(m, candidates, top_n_highlight=5):
    """Add candidate markers with modern color gradient based on F score."""
    if not candidates:
        return m
    
    # Get colors from palette
    color_high = LAYER_COLORS["candidates_high"]
    color_mid = LAYER_COLORS["candidates_mid"]
    color_low = LAYER_COLORS["candidates_low"]
    
    f_values = [c["F"] for c in candidates]
    f_min, f_max = min(f_values), max(f_values)
    f_range = f_max - f_min if f_max > f_min else 1.0
    
    for i, cand in enumerate(candidates):
        # Normalize F to 0-1 for color
        norm_f = (cand["F"] - f_min) / f_range
        
        # Color interpolation: brown -> orange -> gold
        if norm_f < 0.5:
            # Low to mid
            t = norm_f * 2
            r = int(139 + (247 - 139) * t)
            g = int(69 + (127 - 69) * t)
            b = int(19 + (0 - 19) * t)
        else:
            # Mid to high
            t = (norm_f - 0.5) * 2
            r = int(247 + (255 - 247) * t)
            g = int(127 + (215 - 127) * t)
            b = int(0)
        color = f"#{r:02x}{g:02x}{b:02x}"
        
        # Larger markers for top candidates
        radius = 9 if i < top_n_highlight else 5
        
        popup_html = f"""
        <b>Rank #{i+1}</b><br>
        F: {cand['F']:.3f}<br>
        G: {cand['G']:.3f}<br>
        H: {cand['H']:.3f}<br>
        Region: {cand['region']}<br>
        <a href="https://www.google.com/maps?q={cand['lat']},{cand['lon']}" target="_blank">Google Maps</a>
        """
        
        folium.CircleMarker(
            location=[cand["lat"], cand["lon"]],
            radius=radius,
            color="#333333",
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"#{i+1} F={cand['F']:.2f}",
        ).add_to(m)
    
    return m


def add_plate_boundaries(m, boundaries):
    """Add plate boundary polylines with modern styling."""
    if not boundaries:
        return m
    
    boundary_color = LAYER_COLORS["boundaries"]
    
    for poly in boundaries:
        if len(poly) < 2:
            continue
        coords = [[p[0], p[1]] for p in poly]  # [lat, lon]
        folium.PolyLine(
            locations=coords,
            color=boundary_color,
            weight=1.5,
            opacity=0.4,
            dash_array="5, 5",  # Dashed line for subtle appearance
        ).add_to(m)
    
    return m


def add_heatmap_layer(m, result, intensity_scale=1.0):
    """Add F heatmap overlay to the map."""
    if result is None:
        return m
    
    F_grid = result["F"]
    lats = result["lats"]
    lons = result["lons"]
    
    # Normalize F values for heatmap intensity
    F_min, F_max = F_grid.min(), F_grid.max()
    F_range = F_max - F_min if F_max > F_min else 1.0
    F_norm = (F_grid - F_min) / F_range
    
    # Create heatmap data: [[lat, lon, intensity], ...]
    # Subsample further if needed for performance
    heat_data = []
    step = max(1, len(lats) // 100)
    for i in range(0, len(lats), step):
        for j in range(0, len(lons), step):
            intensity = float(F_norm[i, j]) * intensity_scale
            if intensity > 0.1:  # Skip very low values
                heat_data.append([float(lats[i]), float(lons[j]), intensity])
    
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=10,
            radius=15,
            blur=10,
            gradient=LAYER_COLORS["heatmap"],
        ).add_to(m)
    
    return m


def add_volcano_markers(m, max_volcanoes=500):
    """Add volcano locations with modern styling."""
    if not HAS_VOLCANIC:
        return m
    
    volcano_color = LAYER_COLORS["volcanoes"]
    volcano_fill = LAYER_COLORS["volcanoes_fill"]
    
    try:
        from gsn_volcanic import load_volcano_data
        volcanoes = load_volcano_data()
        
        if not volcanoes:
            return m
        
        volcano_group = folium.FeatureGroup(name="Volcanoes")
        for v in volcanoes[:max_volcanoes]:
            folium.CircleMarker(
                [v["lat"], v["lon"]],
                radius=4,
                color=volcano_color,
                weight=1,
                fill=True,
                fillColor=volcano_fill,
                fillOpacity=0.75,
                popup=f"<b>{v['name']}</b><br>{v['country']}<br>Type: {v.get('type', 'Unknown')}",
                tooltip=v['name'],
            ).add_to(volcano_group)
        
        volcano_group.add_to(m)
    except Exception as e:
        print(f"[WARN] Could not add volcano markers: {e}")
    
    return m


def add_great_circle_lines(m, top_n=5):
    """Draw great circle alignment lines on the map."""
    if not HAS_GEOMETRY:
        return m
    
    try:
        from gsn_geometry import get_cached_alignments, xyz_to_latlon
        
        # Get nodes for alignment detection
        if HAS_EXTENDED_NODES:
            nodes = KNOWN_NODES_EXTENDED
        else:
            nodes = gsn.KNOWN_NODES
        
        alignments = get_cached_alignments(nodes, min_sites=4)
        
        if not alignments:
            return m
        
        # Purple-based colors for alignments
        base_color = LAYER_COLORS["alignments"]
        colors = [base_color, "#9D4EDD", "#C77DFF", "#E0AAFF", "#7B2CBF"]
        
        for idx, align in enumerate(alignments[:top_n]):
            pole = align["pole"]
            strength = align["strength"]
            sites = align.get("sites", [])
            
            # Generate points along the great circle
            points = []
            for theta in np.linspace(0, 2 * np.pi, 180):
                # Create orthonormal basis for the plane perpendicular to pole
                if abs(pole[2]) < 0.9:
                    u = np.cross(pole, [0, 0, 1])
                else:
                    u = np.cross(pole, [1, 0, 0])
                u = u / np.linalg.norm(u)
                v = np.cross(pole, u)
                
                # Point on great circle
                point = np.cos(theta) * u + np.sin(theta) * v
                lat, lon = xyz_to_latlon(point)
                points.append([lat, lon])
            
            color = colors[idx % len(colors)]
            
            # Split polyline at date line to avoid wrapping issues
            segments = []
            current_segment = [points[0]]
            for i in range(1, len(points)):
                if abs(points[i][1] - points[i-1][1]) > 180:
                    # Date line crossing
                    segments.append(current_segment)
                    current_segment = [points[i]]
                else:
                    current_segment.append(points[i])
            segments.append(current_segment)
            
            for segment in segments:
                if len(segment) >= 2:
                    folium.PolyLine(
                        segment,
                        weight=2,
                        color=color,
                        opacity=0.6,
                        popup=f"Alignment #{idx+1}: {strength} sites<br>{', '.join(sites[:5])}...",
                        tooltip=f"Alignment: {strength} sites",
                    ).add_to(m)
        
    except Exception as e:
        print(f"[WARN] Could not add great circle lines: {e}")
    
    return m


def add_network_overlay(m):
    """Add network edges and hub markers to map."""
    if not HAS_NETWORK:
        return m
    
    try:
        # Get nodes for network
        if HAS_EXTENDED_NODES:
            nodes = KNOWN_NODES_EXTENDED
        else:
            nodes = gsn.KNOWN_NODES
        
        network = get_network(nodes)
        if network is None:
            return m
        
        # Get edges
        edges = network.get_edges()
        
        # Draw edges by type
        for edge in edges:
            from_coords = edge["from_coords"]
            to_coords = edge["to_coords"]
            
            if from_coords is None or to_coords is None:
                continue
            
            edge_type = edge.get("type", "distance")
            
            if edge_type == "delaunay":
                color = LAYER_COLORS["network_delaunay"]
                weight = 1
                opacity = 0.3
            elif edge_type == "golden":
                color = LAYER_COLORS["network_golden"]
                weight = 2
                opacity = 0.5
            else:
                color = LAYER_COLORS["network_distance"]
                weight = 1
                opacity = 0.25
            
            folium.PolyLine(
                [from_coords, to_coords],
                color=color,
                weight=weight,
                opacity=opacity,
                popup=f"{edge['from']} → {edge['to']}<br>Type: {edge_type}<br>Distance: {edge.get('distance', 0):.0f} km",
            ).add_to(m)
        
        # Highlight hub nodes
        hubs = network.identify_hubs(top_n=5)
        for hub in hubs:
            if hub in network._node_coords:
                lat, lon = network._node_coords[hub]
                folium.CircleMarker(
                    [lat, lon],
                    radius=12,
                    color=LAYER_COLORS["network_hub"],
                    weight=3,
                    fill=True,
                    fillColor=LAYER_COLORS["network_hub"],
                    fillOpacity=0.3,
                    popup=f"<b>Hub: {hub}</b>",
                    tooltip=f"Hub: {hub}",
                ).add_to(m)
        
    except Exception as e:
        print(f"[WARN] Could not add network overlay: {e}")
    
    return m


def cluster_candidates(candidates, eps_deg=5.0, min_samples=2):
    """Cluster candidates using DBSCAN."""
    if len(candidates) < 2:
        return candidates, []
    
    try:
        from sklearn.cluster import DBSCAN
        
        coords = np.array([[c["lat"], c["lon"]] for c in candidates])
        
        # Convert eps to radians for haversine metric
        eps_rad = np.radians(eps_deg)
        
        clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
        labels = clustering.fit_predict(np.radians(coords))
        
        # Add cluster label to each candidate
        for i, c in enumerate(candidates):
            c["cluster"] = int(labels[i])
        
        # Compute cluster statistics
        clusters = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            members = [c for c in candidates if c["cluster"] == label]
            center_lat = np.mean([c["lat"] for c in members])
            center_lon = np.mean([c["lon"] for c in members])
            clusters.append({
                "label": label,
                "center": (center_lat, center_lon),
                "size": len(members),
                "avg_F": np.mean([c["F"] for c in members]),
                "max_F": max([c["F"] for c in members]),
            })
        
        return candidates, clusters
        
    except ImportError:
        print("[WARN] scikit-learn required for clustering")
        return candidates, []
    except Exception as e:
        print(f"[WARN] Clustering failed: {e}")
        return candidates, []


def get_drawn_bounds(map_data):
    """Extract bounds from drawn shapes on the map.
    
    Returns: dict with lat_min, lat_max, lon_min, lon_max or None
    """
    if not map_data:
        return None
    
    all_drawings = map_data.get("all_drawings", [])
    if not all_drawings:
        return None
    
    try:
        # Get the first (most recent) drawing
        drawing = all_drawings[0]
        geom = drawing.get("geometry", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])
        
        if not coords:
            return None
        
        # Handle different geometry types
        if geom_type == "Polygon":
            # Polygon: [[lon, lat], [lon, lat], ...]
            ring = coords[0] if coords else []
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
        elif geom_type == "Point":
            # Point: [lon, lat] - create small box around it
            lon, lat = coords
            return {
                "lat_min": lat - 5,
                "lat_max": lat + 5,
                "lon_min": lon - 5,
                "lon_max": lon + 5,
            }
        else:
            return None
        
        if not lons or not lats:
            return None
        
        return {
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lon_min": min(lons),
            "lon_max": max(lons),
        }
    except Exception as e:
        print(f"[WARN] Could not extract drawn bounds: {e}")
        return None


def add_cluster_boundaries(m, clusters):
    """Draw cluster boundaries with modern styling."""
    if not clusters:
        return m
    
    colors = LAYER_COLORS["clusters"]
    
    for cluster in clusters:
        lat, lon = cluster["center"]
        color = colors[cluster["label"] % len(colors)]
        size = cluster["size"]
        
        # Radius based on cluster size (100-500km)
        radius = min(500000, max(100000, size * 50000))
        
        folium.Circle(
            [lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillOpacity=0.12,
            weight=2,
            dash_array="10, 5",  # Dashed border
            popup=f"<b>Cluster {cluster['label']}</b><br>"
                  f"Size: {size} candidates<br>"
                  f"Avg F: {cluster['avg_F']:.3f}<br>"
                  f"Max F: {cluster['max_F']:.3f}",
            tooltip=f"Cluster: {size} candidates",
        ).add_to(m)
    
    return m


# ---------------------------------------------------------
# Main App
# ---------------------------------------------------------
def main():
    st.title("🌍 Global Stabilization Network Node Predictor")
    
    # Load data
    with st.spinner("Loading datasets..."):
        datasets_loaded, boundaries = load_datasets()
        # Ensure datasets are actually loaded (not just cached)
        if datasets_loaded:
            datasets_loaded = ensure_datasets_loaded()
    
    if not datasets_loaded:
        st.error("⚠️ NetCDF datasets could not be loaded. Please ensure gravity_model.nc and crust_model.nc are present.")
        st.info("The app requires NetCDF gravity and crustal thickness data files.")
        return
    
    # ---------------------------------------------------------
    # Sidebar Controls
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ GSN Predictor")
        
        mode = st.radio(
            "Mode",
            ["View Candidates", "Run Scan", "Click to Evaluate", "Data Status"],
            help="Select how you want to interact with the map"
        )
        
        st.divider()
        
        # Tabbed controls - Combined Settings tab
        tab_settings, tab_data, tab_ml = st.tabs(["⚙️ Settings", "📁 Data", "🤖 ML"])
        
        with tab_settings:
            # Preset selector (always visible at top)
            import os
            ml_model_available = HAS_ML_SCORER and os.path.exists("gsn_ml_grid_scorer.pth")
            available_presets = list(PRESETS.keys())
            
            preset_name = st.selectbox(
                "Preset",
                options=available_presets,
                index=1,  # Default to "Balanced"
                help="Choose a preset or select Custom for full control"
            )
            preset = PRESETS[preset_name]
            st.caption(f"_{preset['description']}_")
            
            # Handle ML Scoring preset
            if preset_name == "ML Scoring":
                if ml_model_available:
                    st.session_state.use_ml_scoring = True
                    st.success("🧠 ML scoring enabled")
                else:
                    st.warning("⚠️ ML model not trained. Go to ML tab to train one.")
                    st.session_state.use_ml_scoring = False
            
            # Get preset values first (needed for expander headers)
            ml_weights = st.session_state.get("applied_ml_weights", {})
            if preset_name == "Custom":
                _alpha = float(ml_weights.get("alpha", 1.0))
                _beta = float(ml_weights.get("beta", 1.0))
            else:
                _alpha = preset["alpha"]
                _beta = preset["beta"]
            
            # --- WEIGHTS EXPANDER ---
            with st.expander(f"⚖️ Weights (α={_alpha:.1f}, β={_beta:.1f})", expanded=(preset_name == "Custom")):
                if preset_name == "Custom":
                    alpha = st.slider("Alpha (G)", 0.0, 3.0, _alpha, 0.1, key="alpha_slider")
                    beta = st.slider("Beta (H)", 0.0, 3.0, _beta, 0.1, key="beta_slider")
                    gamma = st.slider("Gamma (A)", 0.0, 2.0, 
                                      float(ml_weights.get("gamma", 0.0)), 0.1,
                                      disabled=not HAS_ASTRONOMY, key="gamma_slider")
                    delta = st.slider("Delta (N)", 0.0, 2.0, 
                                      float(ml_weights.get("delta", 0.0)), 0.1,
                                      disabled=not HAS_NETWORK, key="delta_slider")
                    epsilon = st.slider("Epsilon (T)", 0.0, 2.0, 
                                        float(ml_weights.get("epsilon", 0.0)), 0.1,
                                        disabled=not HAS_TEMPORAL, key="epsilon_slider")
                else:
                    alpha = preset["alpha"]
                    beta = preset["beta"]
                    gamma = preset.get("gamma", 0.0)
                    delta = preset.get("delta", 0.0)
                    epsilon = preset.get("epsilon", 0.0)
                    st.caption(f"α={alpha:.1f} | β={beta:.1f} | γ={gamma:.1f} | δ={delta:.1f} | ε={epsilon:.1f}")
            
            # --- SCAN SETTINGS EXPANDER ---
            _step = preset["step_deg"] if preset_name != "Custom" else 5.0
            _topn = preset["top_n"] if preset_name != "Custom" else 50
            with st.expander(f"🔍 Scan ({_step}° grid, top {_topn})", expanded=(preset_name == "Custom")):
                # H Scoring Method
                if preset_name == "Custom":
                    h_method = st.selectbox(
                        "H Method",
                        options=list(H_METHODS.keys()),
                        format_func=lambda x: H_METHODS[x],
                        index=3,
                    )
                    step_deg = st.select_slider(
                        "Resolution (°)",
                        options=[1.0, 2.0, 5.0, 10.0, 15.0, 20.0],
                        value=5.0,
                    )
                    top_n = st.slider("Top N", 5, 100, 50)
                    min_sep = st.slider("Min Separation (°)", 1.0, 20.0, 8.0, 0.5)
                else:
                    h_method = preset["h_method"]
                    step_deg = preset["step_deg"]
                    top_n = preset["top_n"]
                    min_sep = preset["min_sep"]
                    st.caption(f"H: {H_METHODS[h_method]}")
                    st.caption(f"Resolution: {step_deg}° | Top {top_n} | Sep: {min_sep}°")
                
                # Region filter
                use_region_filter = st.checkbox("Limit to Region", value=False)
                if use_region_filter:
                    drawn_region = st.session_state.get("drawn_region", None)
                    if drawn_region:
                        use_drawn = st.checkbox("Use Drawn Region", value=True)
                        if use_drawn:
                            lat_min, lat_max = drawn_region["lat_min"], drawn_region["lat_max"]
                            lon_min, lon_max = drawn_region["lon_min"], drawn_region["lon_max"]
                            st.caption(f"📐 [{lat_min:.0f}°,{lat_max:.0f}°] × [{lon_min:.0f}°,{lon_max:.0f}°]")
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                lat_min = st.number_input("Lat Min", -90.0, 90.0, -60.0)
                                lon_min = st.number_input("Lon Min", -180.0, 180.0, -180.0)
                            with col2:
                                lat_max = st.number_input("Lat Max", -90.0, 90.0, 60.0)
                                lon_max = st.number_input("Lon Max", -180.0, 180.0, 180.0)
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            lat_min = st.number_input("Lat Min", -90.0, 90.0, -60.0)
                            lon_min = st.number_input("Lon Min", -180.0, 180.0, -180.0)
                        with col2:
                            lat_max = st.number_input("Lat Max", -90.0, 90.0, 60.0)
                            lon_max = st.number_input("Lon Max", -180.0, 180.0, 180.0)
                else:
                    lat_min = lat_max = lon_min = lon_max = None
            
            # --- MAP OPTIONS EXPANDER ---
            with st.expander("🗺️ Map Options", expanded=False):
                basemap_style = st.selectbox(
                    "Basemap",
                    options=list(BASEMAP_OPTIONS.keys()),
                    index=0,
                    key="basemap_style",
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    cluster_markers = st.checkbox("Cluster Markers", value=False, key="cluster_markers")
                    show_heatmap = st.toggle("🔥 Heatmap", value=False)
                with col2:
                    enable_draw = st.checkbox("Drawing Tools", value=False, key="enable_draw")
                    show_clusters = st.toggle("🎯 Clustering", value=False)
            
            # --- LAYERS EXPANDER ---
            with st.expander("📐 Layers", expanded=False):
                layers = {}
                layers["Base"] = st.toggle("🗺️ Base", value=True, help="Known nodes, boundaries")
                
                geo_available = HAS_VOLCANIC or HAS_HEATFLOW or HAS_SEISMIC
                layers["Geophysical"] = st.toggle("🌋 Geophysical", value=False, disabled=not geo_available)
                layers["Geometric"] = st.toggle("📐 Geometric", value=False, disabled=not HAS_GEOMETRY)
                layers["Celestial"] = st.toggle("✨ Celestial", value=False, disabled=not HAS_ASTRONOMY)
                layers["Network"] = st.toggle("🕸️ Network", value=False, disabled=not HAS_NETWORK)
                layers["Temporal"] = st.toggle("⏳ Temporal", value=False, disabled=not HAS_TEMPORAL)
                
                # Epoch selector
                if layers.get("Temporal", False) and HAS_TEMPORAL:
                    selected_epoch = st.selectbox(
                        "Epoch",
                        options=list(EPOCHS.keys()),
                        format_func=lambda x: EPOCHS[x]["name"],
                    )
                else:
                    selected_epoch = "current"
                
                # Fine-tune options
                st.caption("**Fine-tune:**")
                if layers["Base"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_known_nodes = st.checkbox("Nodes", value=True, key="ft_nodes")
                    with col2:
                        show_boundaries = st.checkbox("Boundaries", value=True, key="ft_bounds")
                else:
                    show_known_nodes = show_boundaries = False
                
                if layers["Geophysical"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_volcanic_layer = st.checkbox("Volcanoes", value=HAS_VOLCANIC, disabled=not HAS_VOLCANIC, key="ft_volc")
                    with col2:
                        show_heatflow_layer = st.checkbox("Heat Flow", value=HAS_HEATFLOW, disabled=not HAS_HEATFLOW, key="ft_hf")
                else:
                    show_volcanic_layer = show_heatflow_layer = False
                
                if layers["Geometric"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_alignments_layer = st.checkbox("Alignments", value=True, key="ft_align")
                    with col2:
                        show_voronoi_layer = st.checkbox("Voronoi", value=False, key="ft_vor")
                else:
                    show_alignments_layer = show_voronoi_layer = False
                
                if layers["Celestial"]:
                    show_constellation_overlay = st.checkbox("Constellations", value=False, key="ft_const")
                else:
                    show_constellation_overlay = False
                
                if layers.get("Network", False):
                    col1, col2 = st.columns(2)
                    with col1:
                        show_network_edges = st.checkbox("Edges", value=True, key="ft_edges")
                    with col2:
                        show_network_hubs = st.checkbox("Hubs", value=True, key="ft_hubs")
                else:
                    show_network_edges = show_network_hubs = False
                
                if layers.get("Temporal", False):
                    show_epoch_info = st.checkbox("Epoch info", value=True, key="ft_epoch")
        
        with tab_data:
            import pandas as pd
            
            # Module & Data Status as tables
            st.subheader("📦 Module Status")
            
            # Build module status data
            module_data = [
                {
                    "Module": "Known Nodes",
                    "Status": "✅ Active" if HAS_EXTENDED_NODES else "❌ Missing",
                    "Version": f"{len(KNOWN_NODES_EXTENDED)} sites" if HAS_EXTENDED_NODES else "—",
                    "Source": "known_nodes_extended.py"
                },
                {
                    "Module": "Data Sources",
                    "Status": "✅ Active" if HAS_DATA_SOURCES else "❌ Missing",
                    "Version": "v1.0",
                    "Source": "gsn_data_sources.py"
                },
                {
                    "Module": "Geometry",
                    "Status": "✅ Active" if HAS_GEOMETRY else "❌ Missing",
                    "Version": "v1.0",
                    "Source": "gsn_geometry.py"
                },
                {
                    "Module": "Astronomy",
                    "Status": "✅ Active" if HAS_ASTRONOMY else "❌ Missing",
                    "Version": "Skyfield 1.x",
                    "Source": "gsn_astronomy.py"
                },
                {
                    "Module": "Network",
                    "Status": "✅ Active" if HAS_NETWORK else "❌ Missing",
                    "Version": "NetworkX 3.x",
                    "Source": "gsn_network.py"
                },
                {
                    "Module": "Temporal",
                    "Status": "✅ Active" if HAS_TEMPORAL else "❌ Missing",
                    "Version": "v1.0",
                    "Source": "gsn_temporal.py"
                },
                {
                    "Module": "Validation",
                    "Status": "✅ Active" if HAS_VALIDATION else "❌ Missing",
                    "Version": "v1.0",
                    "Source": "gsn_validation.py"
                },
                {
                    "Module": "ML Scorer",
                    "Status": "✅ Active" if HAS_ML_SCORER else "❌ Missing",
                    "Version": "PyTorch 2.x",
                    "Source": "gsn_ml_grid_scorer.py"
                },
            ]
            
            df_modules = pd.DataFrame(module_data)
            st.dataframe(df_modules, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Data sources table with version/date info
            st.subheader("📊 Data Sources")
            
            data_sources = [
                {
                    "Dataset": "Gravity Anomaly",
                    "Status": "✅" if HAS_DATA_SOURCES else "❌",
                    "Source": "WGM2012 (BGI)",
                    "Resolution": "2 arc-min (~3.7 km)",
                    "Date": "2012"
                },
                {
                    "Dataset": "Crustal Thickness",
                    "Status": "✅" if HAS_DATA_SOURCES else "❌",
                    "Source": "CRUST1.0 (UCSD)",
                    "Resolution": "1° (~111 km)",
                    "Date": "2013"
                },
                {
                    "Dataset": "Plate Boundaries",
                    "Status": "✅",
                    "Source": "USGS/Literature",
                    "Resolution": "Vector",
                    "Date": "2020"
                },
                {
                    "Dataset": "Volcanoes",
                    "Status": "✅" if HAS_VOLCANIC else "❌",
                    "Source": "GVP Smithsonian",
                    "Resolution": "Point (~1,400)",
                    "Date": "2024"
                },
                {
                    "Dataset": "Heat Flow",
                    "Status": "✅" if HAS_HEATFLOW else "❌",
                    "Source": "IHFC Database",
                    "Resolution": "Point (~70,000)",
                    "Date": "2024"
                },
                {
                    "Dataset": "Seismic Events",
                    "Status": "✅" if HAS_SEISMIC else "❌",
                    "Source": "USGS Earthquake Catalog",
                    "Resolution": "Point",
                    "Date": "1900-2024"
                },
                {
                    "Dataset": "Magnetic Anomaly",
                    "Status": "✅" if HAS_DATA_SOURCES else "❌",
                    "Source": "EMAG2 (NOAA)",
                    "Resolution": "2 arc-min",
                    "Date": "2017"
                },
                {
                    "Dataset": "World Stress Map",
                    "Status": "✅" if HAS_STRESS_MAP else "❌",
                    "Source": "WSM (GFZ Potsdam)",
                    "Resolution": "Point",
                    "Date": "2016"
                },
                {
                    "Dataset": "Archaeological Sites",
                    "Status": "✅" if HAS_ARCHAEOLOGY else "❌",
                    "Source": "Curated Database",
                    "Resolution": "Point (~500+)",
                    "Date": "2024"
                },
            ]
            
            df_data = pd.DataFrame(data_sources)
            st.dataframe(df_data, use_container_width=True, hide_index=True)
            
            st.caption("💡 Data sources are controlled via Map Layers in the Basic tab.")
        
        # Helper function for ML training
        def _run_ml_training(preset, ml_model_path):
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            
            try:
                # Get known nodes
                if HAS_EXTENDED_NODES:
                    known_nodes = KNOWN_NODES_EXTENDED
                else:
                    known_nodes = gsn.KNOWN_NODES
                
                status_text.text(f"Training with {len(known_nodes)} known nodes...")
                progress_bar.progress(10, text="Preparing data...")
                
                config = TrainingConfig(
                    epochs=preset["epochs"],
                    n_augmentations=preset["augmentations"],
                    n_random_negatives=preset["negatives"],
                    patience=preset["patience"],
                )
                
                progress_bar.progress(20, text="Training neural network...")
                
                # Train using the function from gsn_node_predictor
                scorer = gsn.train_ml_scorer(known_nodes, config, ml_model_path, verbose=False)
                
                progress_bar.progress(100, text="Training complete!")
                
                if scorer is not None:
                    st.success(f"✓ Model trained and saved!")
                    st.session_state.ml_scorer_trained = True
                    st.rerun()
                else:
                    st.error("Training failed - check console for errors")
                    
            except Exception as e:
                st.error(f"Training error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        with tab_ml:
            # ML Grid Scorer section - Simplified
            st.subheader("🧠 ML Grid Scoring")
            
            if not HAS_ML_SCORER:
                st.warning("ML Grid Scorer requires PyTorch. Install with: pip install torch")
            else:
                # Check for existing model
                import os
                ml_model_path = "gsn_ml_grid_scorer.pth"
                ml_model_exists = os.path.exists(ml_model_path)
                
                # Training presets
                ML_TRAIN_PRESETS = {
                    "Quick (~1 min)": {"epochs": 100, "negatives": 1000, "augmentations": 5, "patience": 15},
                    "Standard (~3 min)": {"epochs": 300, "negatives": 2000, "augmentations": 10, "patience": 30},
                    "Thorough (~10 min)": {"epochs": 500, "negatives": 3000, "augmentations": 15, "patience": 50},
                }
                
                # Consolidated toggle + status on one line
                toggle_col, status_col = st.columns([2, 1])
                with toggle_col:
                    use_ml_scoring = st.toggle(
                        "Use ML-based scoring",
                        value=st.session_state.get("use_ml_scoring", False),
                        disabled=not ml_model_exists,
                        help="Use neural network instead of linear formula" if ml_model_exists else "Train a model first"
                    )
                    st.session_state.use_ml_scoring = use_ml_scoring
                with status_col:
                    if ml_model_exists:
                        if use_ml_scoring:
                            st.success("🧠 Active")
                        else:
                            st.caption("✓ Model ready")
                    else:
                        st.caption("No model")
                
                # Training section - collapsed when model exists
                if ml_model_exists:
                    # Only show as collapsed expander for retraining
                    with st.expander("🔄 Retrain Model", expanded=False):
                        train_preset = st.selectbox(
                            "Training intensity",
                            options=list(ML_TRAIN_PRESETS.keys()),
                            index=1,  # Default to Standard
                            key="ml_train_preset"
                        )
                        
                        if st.button("🚀 Retrain", type="secondary", use_container_width=True):
                            preset = ML_TRAIN_PRESETS[train_preset]
                            _run_ml_training(preset, ml_model_path)
                else:
                    # No model - show training prominently
                    st.divider()
                    st.write("**Train a model to enable ML scoring:**")
                    
                    train_preset = st.selectbox(
                        "Training intensity",
                        options=list(ML_TRAIN_PRESETS.keys()),
                        index=1,  # Default to Standard
                        key="ml_train_preset_new"
                    )
                    
                    if st.button("🚀 Train ML Model", type="primary", use_container_width=True):
                        preset = ML_TRAIN_PRESETS[train_preset]
                        _run_ml_training(preset, ml_model_path)
                
        # Derive settings from layer groups for config
        use_ext = layers.get("Geophysical", False)
        use_geo = layers.get("Geometric", False)
        use_astro = layers.get("Celestial", False)
        
        # Store config in session state for use by evaluation functions
        st.session_state.config = {
            "preset": preset_name,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta if preset_name == "Custom" else preset.get("delta", 0.0),
            "epsilon": epsilon if preset_name == "Custom" else preset.get("epsilon", 0.0),
            "h_method": h_method if preset_name == "Custom" else preset["h_method"],
            "step_deg": step_deg if preset_name == "Custom" else preset["step_deg"],
            "top_n": top_n if preset_name == "Custom" else preset["top_n"],
            "min_sep": min_sep if preset_name == "Custom" else preset["min_sep"],
            "use_extended_data": use_ext,
            "use_geometry": use_geo,
            "use_astronomy": use_astro,
            "layers": layers,
            "selected_epoch": selected_epoch if layers.get("Temporal", False) else "current",
        }
    
    # ---------------------------------------------------------
    # Main Content Area
    # ---------------------------------------------------------
    
    # Initialize session state for click handling
    if "last_clicked" not in st.session_state:
        st.session_state.last_clicked = None
    if "candidates" not in st.session_state:
        st.session_state.candidates = []
    if "scan_stats" not in st.session_state:
        st.session_state.scan_stats = {}
    
    # Run scan if in scan mode or first load
    if mode == "Run Scan":
        if st.sidebar.button("🔍 Run Scan", type="primary", use_container_width=True):
            with st.spinner(f"Scanning globe at {step_deg}° resolution..."):
                candidates, stats, error = compute_scan_results(
                    step_deg, top_n, min_sep, alpha, beta,
                    lat_min, lat_max, lon_min, lon_max
                )
                if error:
                    st.error(f"Scan failed: {error}")
                else:
                    st.session_state.candidates = candidates
                    st.session_state.scan_stats = stats
                    st.success(f"Found {len(candidates)} candidates!")
    
    elif mode == "View Candidates":
        # Auto-run scan with current parameters if no candidates
        if not st.session_state.candidates:
            with st.spinner("Computing initial candidates..."):
                candidates, stats, error = compute_scan_results(
                    step_deg, top_n, min_sep, alpha, beta,
                    lat_min, lat_max, lon_min, lon_max
                )
                if not error:
                    st.session_state.candidates = candidates
                    st.session_state.scan_stats = stats
    
    # Compute heatmap if enabled
    heatmap_result = None
    if show_heatmap:
        use_ml = st.session_state.get("use_ml_scoring", False)
        with st.spinner("Computing heatmap..." + (" (ML)" if use_ml else "")):
            heatmap_result, hm_error = compute_heatmap_data(
                max(2.0, step_deg), alpha, beta, use_ml=use_ml
            )
            if hm_error:
                st.warning(f"Heatmap computation failed: {hm_error}")
    
    # Get map options from session state
    basemap_style = st.session_state.get("basemap_style", "OpenStreetMap")
    enable_draw = st.session_state.get("enable_draw", False)
    use_marker_clustering = st.session_state.get("cluster_markers", False)
    
    # Create map with configured options
    m = create_base_map(
        basemap=basemap_style,
        enable_draw=enable_draw,
        enable_fullscreen=True,
        enable_minimap=True
    )
    
    # Add base layers
    if show_boundaries and boundaries:
        m = add_plate_boundaries(m, boundaries)
    
    if show_heatmap and heatmap_result:
        m = add_heatmap_layer(m, heatmap_result)
        # Show scoring method indicator
        method = heatmap_result.get("method", "linear")
        if method == "ml":
            st.info("🧠 Heatmap using ML-based scoring")
    
    # Add data layers
    if show_volcanic_layer:
        m = add_volcano_markers(m)
    
    if show_alignments_layer:
        m = add_great_circle_lines(m, top_n=5)
    
    # Add network overlay if enabled
    if layers.get("Network", False) and HAS_NETWORK:
        m = add_network_overlay(m)
    
    # Add known nodes (with optional clustering)
    if show_known_nodes:
        if use_marker_clustering:
            m = add_known_nodes_clustered(m)
        else:
            m = add_known_nodes(m)
    
    # Handle candidate clustering if enabled
    candidates_to_show = st.session_state.candidates
    clusters = []
    if show_clusters and candidates_to_show:
        candidates_to_show, clusters = cluster_candidates(candidates_to_show)
        if clusters:
            m = add_cluster_boundaries(m, clusters)
    
    if candidates_to_show:
        m = add_candidates(m, candidates_to_show)
    
    # Add layer control for in-map toggling
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
    # Display map with tabs for 2D/3D, analytics, validation, H comparison, and help
    map_tab, globe_tab, analytics_tab, validation_tab, compare_tab, help_tab = st.tabs(
        ["🗺️ 2D Map", "🌐 3D Globe", "📊 Analytics", "📈 Validation", "🔬 H Compare", "📖 Help"]
    )
    
    with map_tab:
        # Full width map
        map_data = st_folium(
            m,
            width=None,
            height=550,
            returned_objects=["last_clicked", "all_drawings"],
        )
        
        # Handle drawn region bounds
        if st.session_state.get("enable_draw", False):
            drawn_bounds = get_drawn_bounds(map_data)
            if drawn_bounds:
                st.info(
                    f"📐 Drawn region: Lat [{drawn_bounds['lat_min']:.2f}, {drawn_bounds['lat_max']:.2f}], "
                    f"Lon [{drawn_bounds['lon_min']:.2f}, {drawn_bounds['lon_max']:.2f}]"
                )
                # Store in session state for scan region
                st.session_state["drawn_region"] = drawn_bounds
    
    with globe_tab:
        st.write("**3D Globe View**")
        if st.session_state.candidates:
            try:
                import pydeck as pdk
                
                # Prepare candidate data
                candidate_data = []
                for c in st.session_state.candidates:
                    candidate_data.append({
                        "lat": c["lat"],
                        "lon": c["lon"],
                        "F": c["F"],
                        "name": f"F={c['F']:.2f}",
                    })
                
                # Prepare known nodes data
                node_data = []
                if HAS_EXTENDED_NODES:
                    for name, data in KNOWN_NODES_EXTENDED.items():
                        node_data.append({
                            "lat": data["coords"][0],
                            "lon": data["coords"][1],
                            "name": name,
                        })
                else:
                    for name, coords in gsn.KNOWN_NODES.items():
                        node_data.append({
                            "lat": coords[0],
                            "lon": coords[1],
                            "name": name,
                        })
                
                # Candidate layer (yellow)
                candidate_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=candidate_data,
                    get_position=["lon", "lat"],
                    get_radius=100000,
                    get_fill_color=[255, 200, 0, 180],
                    pickable=True,
                    auto_highlight=True,
                )
                
                # Known nodes layer (red)
                node_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=node_data,
                    get_position=["lon", "lat"],
                    get_radius=80000,
                    get_fill_color=[255, 0, 0, 200],
                    pickable=True,
                )
                
                # Create deck with free map style
                deck = pdk.Deck(
                    layers=[candidate_layer, node_layer],
                    initial_view_state=pdk.ViewState(
                        latitude=20,
                        longitude=0,
                        zoom=1.2,
                        pitch=45,  # Tilted view for 3D effect
                    ),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={"text": "{name}"},
                )
                
                st.pydeck_chart(deck, use_container_width=True, height=500)
                st.caption("🟡 Candidates | 🔴 Known GSN Nodes")
                
            except ImportError:
                st.warning("pydeck required for 3D globe. Install with: pip install pydeck")
            except Exception as e:
                st.error(f"3D globe error: {e}")
        else:
            # Show just known nodes when no candidates
            st.info("Run a scan to see candidates. Showing known GSN nodes only.")
            try:
                import pydeck as pdk
                
                node_data = []
                if HAS_EXTENDED_NODES:
                    for name, data in KNOWN_NODES_EXTENDED.items():
                        node_data.append({
                            "lat": data["coords"][0],
                            "lon": data["coords"][1],
                            "name": name,
                        })
                else:
                    for name, coords in gsn.KNOWN_NODES.items():
                        node_data.append({
                            "lat": coords[0],
                            "lon": coords[1],
                            "name": name,
                        })
                
                node_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=node_data,
                    get_position=["lon", "lat"],
                    get_radius=80000,
                    get_fill_color=[255, 0, 0, 200],
                    pickable=True,
                )
                
                deck = pdk.Deck(
                    layers=[node_layer],
                    initial_view_state=pdk.ViewState(
                        latitude=20,
                        longitude=0,
                        zoom=1.2,
                        pitch=45,
                    ),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={"text": "{name}"},
                )
                
                st.pydeck_chart(deck, use_container_width=True, height=500)
                st.caption("🔴 Known GSN Nodes")
            except Exception as e:
                st.error(f"Could not load 3D view: {e}")
    
    with analytics_tab:
        st.write("**Candidate Analytics**")
        if st.session_state.candidates:
            try:
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame(st.session_state.candidates)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # F Score distribution
                    fig1 = px.histogram(
                        df, x="F", nbins=20, 
                        title="F Score Distribution",
                        labels={"F": "Node Index (F)", "count": "Count"},
                    )
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # G vs H scatter
                    fig2 = px.scatter(
                        df, x="G", y="H", color="F",
                        title="G vs H Scores",
                        labels={"G": "Geophysical (G)", "H": "Geometric (H)"},
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Geographic scatter
                fig3 = px.scatter_geo(
                    df, lat="lat", lon="lon", color="F",
                    projection="natural earth",
                    title="Geographic Distribution of Candidates",
                    color_continuous_scale="YlOrRd",
                    size="F",
                    size_max=15,
                )
                fig3.update_layout(
                    geo=dict(
                        showland=True,
                        landcolor="rgb(243, 243, 243)",
                        showocean=True,
                        oceancolor="rgb(204, 229, 255)",
                    )
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Summary stats
                st.divider()
                st.write("**Summary Statistics**")
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Total Candidates", len(df))
                with stat_cols[1]:
                    st.metric("Mean F", f"{df['F'].mean():.3f}")
                with stat_cols[2]:
                    st.metric("Max F", f"{df['F'].max():.3f}")
                with stat_cols[3]:
                    st.metric("Std F", f"{df['F'].std():.3f}")
                
            except ImportError:
                st.warning("plotly required for analytics. Install with: pip install plotly")
            except Exception as e:
                st.error(f"Analytics error: {e}")
        else:
            st.info("Run a scan first to see analytics.")
        
        # Constellation Visibility Panel (when Celestial layer enabled)
        if layers.get("Celestial", False) and HAS_ASTRONOMY:
            st.divider()
            st.write("**✨ Constellation Visibility**")
            
            # Location selector for visibility check
            vis_col1, vis_col2 = st.columns([2, 1])
            with vis_col1:
                # Default to top candidate or Giza
                if st.session_state.candidates:
                    default_lat = st.session_state.candidates[0]["lat"]
                    default_lon = st.session_state.candidates[0]["lon"]
                else:
                    default_lat, default_lon = 29.9792, 31.1342
                
                vis_lat = st.number_input("Location Lat", -90.0, 90.0, default_lat, 
                                          format="%.2f", key="vis_lat")
                vis_lon = st.number_input("Location Lon", -180.0, 180.0, default_lon,
                                          format="%.2f", key="vis_lon")
            
            with vis_col2:
                st.write("")
                st.caption(f"Showing visibility from\n{vis_lat:.2f}°, {vis_lon:.2f}°")
            
            # Get and display constellation visibility
            const_data = get_constellation_visibility_summary(vis_lat, vis_lon)
            
            if const_data:
                for const in const_data:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.write(f"{const['icon']} **{const['name']}**")
                    with col2:
                        st.write(f"{const['color']} {const['status']}")
                    with col3:
                        st.progress(const['score'], text=f"{const['score']*100:.0f}%")
                
                # Summary
                visible_count = sum(1 for c in const_data if c['visible'])
                avg_score = sum(c['score'] for c in const_data) / len(const_data) if const_data else 0
                st.caption(f"📊 {visible_count}/{len(const_data)} constellations visible | Average visibility: {avg_score*100:.0f}%")
            else:
                st.warning("Could not compute constellation visibility. Check gsn_astronomy module.")
    
    # Statistical Validation tab
    with validation_tab:
        st.subheader("📈 Statistical Validation")
        st.write("Test whether observed patterns are statistically significant or coincidental.")
        
        # Check if validation module is available
        if not HAS_STATISTICAL_VALIDATION:
            st.warning("Statistical validation module not available. Install scipy for full functionality.")
        else:
            # Get known nodes for validation
            if HAS_EXTENDED_NODES:
                val_nodes = KNOWN_NODES_EXTENDED
            else:
                val_nodes = gsn.KNOWN_NODES
            
            st.info(f"Using {len(val_nodes)} known nodes for validation.")
            
            # Validation options
            col1, col2 = st.columns(2)
            with col1:
                n_sims = st.slider("Monte Carlo Simulations", 100, 5000, 1000, 100,
                                   help="More simulations = more precise p-values (but slower)")
            with col2:
                run_validation = st.button("🔬 Run Full Validation", type="primary",
                                          help="Run CSR test, Ripley's K, and alignment significance")
            
            if run_validation:
                with st.spinner("Running statistical validation tests..."):
                    try:
                        summary = run_full_validation(val_nodes, n_simulations=n_sims)
                        
                        # Store in session state
                        st.session_state["validation_summary"] = summary
                        st.success("Validation complete!")
                    except Exception as e:
                        st.error(f"Validation failed: {e}")
            
            # Display results if available
            if "validation_summary" in st.session_state:
                summary = st.session_state["validation_summary"]
                
                # Overall result banner
                if summary.overall_significant:
                    st.success(f"✅ Pattern is STATISTICALLY SIGNIFICANT (p = {summary.overall_p_value:.4f})")
                else:
                    st.warning(f"⚠️ Pattern is NOT significant (p = {summary.overall_p_value:.4f})")
                
                # Detailed results in expanders
                with st.expander("📊 Complete Spatial Randomness (CSR) Test", expanded=True):
                    csr = summary.csr_result
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Observed Coherence", f"{csr.observed_coherence:.4f}")
                    with col2:
                        st.metric("Expected (Random)", f"{csr.null_mean:.4f} ± {csr.null_std:.4f}")
                    with col3:
                        color = "green" if csr.is_significant_05 else "orange"
                        st.metric("P-value", f"{csr.p_value:.4f}", 
                                 delta="Significant" if csr.is_significant_05 else "Not Significant",
                                 delta_color="normal" if csr.is_significant_05 else "off")
                    
                    st.caption(csr.interpretation)
                
                with st.expander("🎯 Ripley's K-Function (Spatial Clustering)", expanded=False):
                    ripley = summary.ripley_result
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if ripley.significant_clustering:
                            st.success("✅ Significant CLUSTERING detected")
                        else:
                            st.info("➖ No significant clustering")
                    with col2:
                        if ripley.significant_dispersion:
                            st.success("✅ Significant DISPERSION (regular spacing)")
                        else:
                            st.info("➖ No significant dispersion")
                    
                    st.metric("Max Deviation from CSR", f"{ripley.max_deviation:.2f}")
                
                with st.expander("🌐 Great Circle Alignment Test", expanded=False):
                    align = summary.alignment_result
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Observed Alignments", align.n_alignments_observed)
                    with col2:
                        st.metric("Expected (Random)", f"{align.n_alignments_expected:.1f}")
                    with col3:
                        st.metric("Excess Ratio", f"{align.excess_ratio:.2f}x")
                    
                    if align.is_significant:
                        st.success(f"✅ More alignments than expected by chance (p={align.p_value:.4f})")
                    else:
                        st.info(f"Alignments consistent with random placement (p={align.p_value:.4f})")
                
                # Recommendations
                with st.expander("💡 Recommendations", expanded=True):
                    for rec in summary.recommendations:
                        if "significant" in rec.lower() and "not" not in rec.lower():
                            st.success(f"• {rec}")
                        elif "not" in rec.lower() or "no " in rec.lower():
                            st.warning(f"• {rec}")
                        else:
                            st.info(f"• {rec}")
        
        # Bayesian Analysis Section
        st.divider()
        st.subheader("🎲 Bayesian Analysis")
        
        if not HAS_BAYESIAN:
            st.warning("Bayesian scorer module not available.")
        else:
            st.write("Compute proper posterior probabilities instead of arbitrary F-scores.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                bayes_lat = st.number_input("Latitude", -90.0, 90.0, 29.9792, format="%.4f",
                                           key="bayes_lat")
            with col2:
                bayes_lon = st.number_input("Longitude", -180.0, 180.0, 31.1342, format="%.4f",
                                           key="bayes_lon")
            with col3:
                if st.button("🔮 Compute Posterior", key="bayes_compute"):
                    try:
                        scorer = BayesianScorer()
                        result = scorer.compute_posterior(bayes_lat, bayes_lon)
                        st.session_state["bayes_result"] = result
                    except Exception as e:
                        st.error(f"Bayesian computation failed: {e}")
            
            if "bayes_result" in st.session_state:
                result = st.session_state["bayes_result"]
                
                col1, col2 = st.columns(2)
                with col1:
                    # Display probability as percentage
                    prob_pct = result.probability * 100
                    st.metric("P(is GSN node | data)", f"{prob_pct:.4f}%")
                    st.caption(f"95% CI: ({result.credible_interval_95[0]*100:.4f}%, {result.credible_interval_95[1]*100:.4f}%)")
                
                with col2:
                    st.metric("Likelihood Ratio", f"{result.likelihood_ratio:.2f}")
                    st.caption(f"Log-odds: {result.log_odds:.2f}")
                
                # Interpretation
                if result.probability > 0.5:
                    st.success(f"📍 This location has elevated probability of being a GSN node.")
                elif result.probability > 0.01:
                    st.info(f"📍 Location has slightly elevated probability.")
                else:
                    st.warning(f"📍 Location has low probability of being a GSN node.")
        
        # Leave-One-Out Validation
        st.divider()
        st.subheader("🔄 Leave-One-Out Cross-Validation")
        
        if HAS_BAYESIAN:
            if st.button("Run LOOCV on Known Nodes", key="run_loocv"):
                with st.spinner("Running leave-one-out validation (this may take a moment)..."):
                    try:
                        scorer = BayesianScorer()
                        loocv_result = scorer.leave_one_out_validation()
                        st.session_state["loocv_result"] = loocv_result
                    except Exception as e:
                        st.error(f"LOOCV failed: {e}")
            
            if "loocv_result" in st.session_state:
                loocv = st.session_state["loocv_result"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Posterior", f"{loocv['mean_posterior']:.4f}")
                with col2:
                    st.metric("Min Posterior", f"{loocv['min_posterior']:.4f}")
                with col3:
                    st.metric("Max Posterior", f"{loocv['max_posterior']:.4f}")
                
                st.info(loocv['interpretation'])
    
    # H Method Comparison tab
    with compare_tab:
        st.subheader("🔬 H Method Comparison")
        st.write("Compare how different H scoring methods evaluate a location.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            comp_lat = st.number_input("Latitude", -90.0, 90.0, 29.9792, format="%.4f",
                                       key="compare_lat",
                                       help="Latitude of location to evaluate")
        with col2:
            comp_lon = st.number_input("Longitude", -180.0, 180.0, 31.1342, format="%.4f",
                                       key="compare_lon",
                                       help="Longitude of location to evaluate")
        with col3:
            st.write("")  # Spacer
            st.write("")
            run_comparison = st.button("🔍 Compare All Methods", type="primary", use_container_width=True)
        
        # Quick location buttons
        st.write("**Quick Locations:**")
        loc_cols = st.columns(5)
        quick_locs = [
            ("Giza", 29.9792, 31.1342),
            ("Stonehenge", 51.1789, -1.8262),
            ("Göbekli Tepe", 37.2231, 38.9225),
            ("Machu Picchu", -13.1631, -72.5450),
            ("New York", 40.7128, -74.0060),
        ]
        for i, (name, lat, lon) in enumerate(quick_locs):
            with loc_cols[i]:
                if st.button(name, key=f"compare_loc_{name}"):
                    st.session_state.comp_lat = lat
                    st.session_state.comp_lon = lon
                    run_comparison = True
        
        if run_comparison or st.session_state.get("comp_lat"):
            if st.session_state.get("comp_lat"):
                comp_lat = st.session_state.comp_lat
                comp_lon = st.session_state.comp_lon
                st.session_state.comp_lat = None
                st.session_state.comp_lon = None
            
            st.divider()
            st.write(f"**Location:** {comp_lat:.4f}°, {comp_lon:.4f}°")
            
            # Compute all H methods efficiently (only H scores, not full evaluation)
            with st.spinner("Computing all H methods..."):
                results = {}
                
                # Compute G once (shared across all methods)
                try:
                    ga, ct, dist_km = gsn.get_geophysical_inputs(comp_lat, comp_lon)
                    G, _ = gsn.compute_G(ga, ct, dist_km)
                except Exception:
                    G = 0.0
                
                # Compute each H method directly
                for method in H_METHODS.keys():
                    try:
                        if method == "basic":
                            H = gsn.compute_H(comp_lat, comp_lon)
                        elif method == "weighted":
                            if hasattr(gsn, 'compute_H_weighted'):
                                H = gsn.compute_H_weighted(comp_lat, comp_lon, distance_scale=30.0, use_extended=True)
                            else:
                                H = gsn.compute_H(comp_lat, comp_lon)
                        elif method == "combined":
                            if hasattr(gsn, 'compute_H_combined'):
                                H = gsn.compute_H_combined(comp_lat, comp_lon)
                            else:
                                H = gsn.compute_H(comp_lat, comp_lon)
                        elif method == "geometric":
                            if HAS_GEOMETRY:
                                H = compute_H_geometric(comp_lat, comp_lon)
                            else:
                                H = gsn.compute_H(comp_lat, comp_lon)
                        else:  # "full"
                            if hasattr(gsn, 'compute_H_full'):
                                H = gsn.compute_H_full(comp_lat, comp_lon)
                            elif hasattr(gsn, 'compute_H_weighted'):
                                H = gsn.compute_H_weighted(comp_lat, comp_lon, distance_scale=30.0, use_extended=True)
                            else:
                                H = gsn.compute_H(comp_lat, comp_lon)
                        
                        F = 1.0 * G + 1.0 * H
                        results[method] = {"H": H, "G": G, "F": F}
                    except Exception:
                        # Skip methods that fail
                        pass
                
                if results:
                    # Create comparison dataframe
                    import pandas as pd
                    
                    comparison_data = []
                    for method, result in results.items():
                        comparison_data.append({
                            "Method": H_METHODS[method],
                            "H Score": result["H"],
                            "G Score": result["G"],
                            "F Score": result["F"],
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    
                    # Display table
                    st.write("**Comparison Results:**")
                    st.dataframe(df.style.highlight_max(subset=["H Score", "F Score"], color="lightgreen"),
                                use_container_width=True)
                    
                    # Bar chart
                    st.write("**H Score Comparison:**")
                    chart_data = pd.DataFrame({
                        "Method": [H_METHODS[m] for m in results.keys()],
                        "H Score": [r["H"] for r in results.values()],
                    })
                    st.bar_chart(chart_data.set_index("Method"))
                    
                    # Best method recommendation
                    best_method = max(results.items(), key=lambda x: x[1]["F"])
                    st.success(f"**Recommended:** {H_METHODS[best_method[0]]} (F = {best_method[1]['F']:.3f})")
    
    # Help tab content
    with help_tab:
        st.subheader("📖 GSN Node Predictor Documentation")
        
        help_section = st.selectbox(
            "Select topic:",
            ["🏠 Introduction", "🧮 Scoring Formula", "🗺️ Feature Guide", "🚀 Quick Start", "❓ FAQ"],
            key="help_topic"
        )
        
        st.divider()
        
        if help_section == "🏠 Introduction":
            st.markdown("""
            ### What is the Global Stabilization Network (GSN)?
            
            The **Global Stabilization Network** is a theoretical framework proposing that 
            ancient sacred sites, megalithic structures, and significant archaeological 
            locations around the world are positioned according to geometric and geophysical 
            patterns rather than random placement.
            
            **This tool helps you:**
            - **Analyze locations** - Evaluate any point for GSN node characteristics
            - **Discover candidates** - Scan regions to find high-potential locations
            - **Visualize patterns** - See heatmaps and geometric alignments
            - **Compare methods** - Test different scoring approaches
            - **Train ML models** - Use machine learning to discover patterns
            
            **Known Reference Nodes include:** Giza, Stonehenge, Göbekli Tepe, Angkor Wat, 
            Machu Picchu, Teotihuacan, Petra, Baalbek, and 30+ other validated ancient sites.
            """)
            
        elif help_section == "🧮 Scoring Formula":
            st.markdown("""
            ### The Node Index Formula
            
            Each location is assigned a **Node Index (F)** based on multiple components:
            """)
            
            st.info("**F = αG + βH + γA + δN + εT**")
            
            st.markdown("""
            | Component | Description |
            |-----------|-------------|
            | **G** (Geophysical) | Gravity anomaly, crustal thickness, plate boundary proximity |
            | **H** (Geometric) | Penrose angle coherence with known nodes |
            | **A** (Astronomical) | Constellation visibility and celestial alignments |
            | **N** (Network) | Graph centrality and connectivity |
            | **T** (Temporal) | Historical epoch alignments |
            
            ---
            
            ### H Scoring Methods
            
            | Method | Description |
            |--------|-------------|
            | **Basic** | Original Penrose angles (36°, 72°, 108°, 144°) |
            | **Weighted** | Distance-weighted with extended sacred angles |
            | **Combined** | Includes archaeological site coherence |
            | **Geometric** | Alignments, Voronoi, golden ratio patterns |
            | **Full** | All methods combined |
            
            ---
            
            ### Recommended Weights
            
            | Focus | α (G) | β (H) | γ (A) | δ (N) | ε (T) |
            |-------|-------|-------|-------|-------|-------|
            | Balanced | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
            | Geophysical | 1.5 | 0.8 | 0.0 | 0.0 | 0.0 |
            | Geometric | 0.8 | 1.5 | 0.0 | 0.0 | 0.0 |
            | Full Analysis | 1.0 | 1.0 | 0.5 | 0.3 | 0.2 |
            """)
            
        elif help_section == "🗺️ Feature Guide":
            st.markdown("""
            ### Configuration Presets
            
            | Preset | Description |
            |--------|-------------|
            | **Fast Scan** | Coarse 10° grid for quick overview |
            | **Balanced** | 5° grid with geometry enabled |
            | **Accurate** | 2° grid with all features |
            | **ML Scoring** | Neural network based scoring |
            
            ---
            
            ### Map Layers
            
            | Layer | Contents |
            |-------|----------|
            | **Base** | Known GSN nodes, plate boundaries |
            | **Geophysical** | Volcanoes, heat flow, seismic activity |
            | **Geometric** | Great circle alignments, Voronoi cells |
            | **Celestial** | Constellation visibility patterns |
            | **Network** | Node connections, hub identification |
            | **Temporal** | Epoch-specific analysis |
            
            ---
            
            ### Analysis Options
            
            - **🔥 F Heatmap** - Color gradient showing F scores (red=high, blue=low)
            - **🎯 Clustering** - Groups nearby candidates using DBSCAN
            - **Drawing Tools** - Draw rectangles to define scan regions
            """)
            
        elif help_section == "🚀 Quick Start":
            st.markdown("""
            ### Workflow 1: Basic Global Scan
            1. Select **"Balanced"** preset
            2. Set Mode to **"Run Scan"**
            3. Click **"🔍 Run Scan"**
            4. View candidates on map and in results
            
            ---
            
            ### Workflow 2: Evaluate a Location
            1. Set Mode to **"Click to Evaluate"**
            2. Click any location on the map
            3. View detailed scores (F, G, H, etc.)
            
            ---
            
            ### Workflow 3: Compare H Methods
            1. Go to **"🔬 H Compare"** tab
            2. Enter coordinates or click a quick location
            3. Click **"Compare All Methods"**
            4. See which method scores highest
            
            ---
            
            ### Tips
            - Start with coarse scans (10°), then zoom in with finer resolution
            - Enable heatmap only after finding interesting regions
            - Use the 3D Globe tab for a different perspective
            """)
            
        elif help_section == "❓ FAQ":
            with st.expander("What does a high F score mean?"):
                st.markdown("""
                - **F ≥ 2.0**: Strong candidate - multiple factors align
                - **F ≥ 1.0**: Moderate candidate - some GSN characteristics
                - **F ≥ 0.3**: Weak candidate - marginal signals
                - **F < 0.3**: Unlikely - background/random
                """)
            
            with st.expander("Why are some features disabled?"):
                st.markdown("""
                Features are disabled when their required modules aren't available:
                - **Astronomy**: Requires `skyfield`
                - **Network**: Requires `networkx`
                - **ML Scoring**: Requires `torch` (PyTorch)
                
                Install missing dependencies: `pip install skyfield networkx torch`
                """)
            
            with st.expander("How do I add my own sites?"):
                st.markdown("""
                Edit `known_nodes_extended.py`:
                ```python
                "Your_Site": {
                    "coords": (latitude, longitude),
                    "category": "megalithic",
                    "age_bp": 3000,
                    "country": "Country Name",
                },
                ```
                Then restart the app.
                """)
            
            with st.expander("Why is scanning slow?"):
                st.markdown("""
                Scan time depends on:
                - **Grid resolution**: 2° is ~16x slower than 10°
                - **Features enabled**: More layers = more computation
                - **Region size**: Full globe takes longer
                
                **Tips:** Start with "Fast Scan" preset, use regional scans for detail.
                """)
    
    # Handle click-to-evaluate mode and results (inside map tab context)
    with map_tab:
        # Results section below map
        st.divider()
        
        if mode == "Click to Evaluate":
            st.subheader("📍 Click Evaluation")
            st.info("Click anywhere on the map to evaluate that location's node potential.")
            
            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                click_lat = clicked["lat"]
                click_lon = clicked["lng"]
                
                # Show location and results in columns
                eval_col1, eval_col2 = st.columns([1, 2])
                
                with eval_col1:
                    st.write(f"**Selected Location:**")
                    st.write(f"Lat: {click_lat:.4f}°")
                    st.write(f"Lon: {click_lon:.4f}°")
                    
                    gmaps_url = f"https://www.google.com/maps?q={click_lat},{click_lon}"
                    st.link_button("🗺️ Open in Google Maps", gmaps_url)
                
                with eval_col2:
                    # Get config from session state
                    config = st.session_state.get("config", {"alpha": alpha, "beta": beta, "h_method": "full"})
                    
                    with st.spinner("Evaluating..."):
                        result, error = evaluate_single_point(click_lat, click_lon, config)
                    
                    if error:
                        st.error(f"Evaluation failed: {error}")
                    elif result:
                        # Main scores in a row
                        score_cols = st.columns(5)
                        with score_cols[0]:
                            st.metric("F", f"{result['F']:.2f}")
                        with score_cols[1]:
                            st.metric("G", f"{result['G']:.2f}")
                        with score_cols[2]:
                            st.metric("H", f"{result['H']:.2f}")
                        if result.get("A", 0) > 0:
                            with score_cols[3]:
                                st.metric("A", f"{result['A']:.2f}")
                        if result.get("N", 0) > 0 or result.get("T", 0) > 0:
                            with score_cols[4]:
                                if result.get("N", 0) > 0:
                                    st.metric("N", f"{result['N']:.2f}")
                                elif result.get("T", 0) > 0:
                                    st.metric("T", f"{result['T']:.2f}")
                        
                        st.caption(f"{result['classification']} | {result['region']}")
                        
                        # Details expander
                        with st.expander("📊 Detailed Scores"):
                            detail_cols = st.columns(3)
                            with detail_cols[0]:
                                st.write(f"Gravity: {result['ga']:.2f} mGal")
                                st.write(f"Crustal Thickness: {result['ct']:.2f} km")
                            with detail_cols[1]:
                                st.write(f"Boundary Dist: {result['dist_boundary_km']:.1f} km")
                                st.write(f"H method: {H_METHODS.get(result.get('h_method', 'full'), 'Full')}")
                            with detail_cols[2]:
                                st.write(f"GA norm: {result['ga_norm']:.3f}")
                                st.write(f"CT norm: {result['ct_norm']:.3f}")
        
        else:
            # Show stats and candidate list
            st.subheader("📊 Results")
            
            if st.session_state.scan_stats:
                stats = st.session_state.scan_stats
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Grid Points", f"{stats.get('count', 0):,}")
                with stat_cols[1]:
                    st.metric("F Range", f"{stats.get('min', 0):.1f} to {stats.get('max', 0):.1f}")
                with stat_cols[2]:
                    st.metric("Median F", f"{stats.get('median', 0):.2f}")
                with stat_cols[3]:
                    st.metric("90th %ile", f"{stats.get('p90', 0):.2f}")
            
            if st.session_state.candidates:
                import pandas as pd
                
                # Build table data
                table_data = []
                for i, cand in enumerate(st.session_state.candidates):
                    table_data.append({
                        "Rank": i + 1,
                        "F": round(cand['F'], 2),
                        "G": round(cand['G'], 2),
                        "H": round(cand['H'], 2),
                        "Lat": round(cand['lat'], 2),
                        "Lon": round(cand['lon'], 2),
                        "Region": cand['region'][:30],
                    })
                
                df_candidates = pd.DataFrame(table_data)
                
                # Display table
                st.write(f"**Top {len(st.session_state.candidates)} Candidates:**")
                st.dataframe(
                    df_candidates,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "F": st.column_config.NumberColumn("F Score", format="%.2f"),
                        "G": st.column_config.NumberColumn("G", format="%.2f"),
                        "H": st.column_config.NumberColumn("H", format="%.2f"),
                        "Lat": st.column_config.NumberColumn("Lat", format="%.2f°"),
                        "Lon": st.column_config.NumberColumn("Lon", format="%.2f°"),
                    }
                )
                
                # Dropdown to select candidate for detailed breakdown
                st.write("**Score Reasoning:**")
                selected_rank = st.selectbox(
                    "Select candidate to view details",
                    options=list(range(1, len(st.session_state.candidates) + 1)),
                    format_func=lambda x: f"#{x}: F={st.session_state.candidates[x-1]['F']:.2f} @ {st.session_state.candidates[x-1]['lat']:.1f}°, {st.session_state.candidates[x-1]['lon']:.1f}°",
                    key="candidate_detail_select"
                )
                
                # Show detailed breakdown for selected candidate
                selected_cand = st.session_state.candidates[selected_rank - 1]
                
                with st.expander(f"Detailed Score Breakdown for #{selected_rank}", expanded=True):
                    detail_cols = st.columns([1, 1, 1])
                    
                    with detail_cols[0]:
                        st.write("**G Score Factors:**")
                        g_score = selected_cand['G']
                        
                        # Get actual geophysical data for this location
                        try:
                            ga, ct, boundary_dist = gsn.get_geophysical_inputs(
                                selected_cand['lat'], selected_cand['lon']
                            )
                            has_geo_data = True
                        except Exception:
                            ga, ct, boundary_dist = 0, 35, 1000
                            has_geo_data = False
                        
                        # Show score with color coding
                        if g_score > 5:
                            st.success(f"G = {g_score:.2f} (Very High)")
                        elif g_score > 2:
                            st.info(f"G = {g_score:.2f} (High)")
                        elif g_score > 0:
                            st.warning(f"G = {g_score:.2f} (Low)")
                        else:
                            st.error(f"G = {g_score:.2f} (Negative)")
                        
                        # Show identified features
                        if has_geo_data:
                            st.caption("**Identified features:**")
                            # Gravity anomaly
                            if abs(ga) > 50:
                                st.caption(f"⚡ Gravity: {ga:.0f} mGal (strong anomaly)")
                            elif abs(ga) > 20:
                                st.caption(f"📊 Gravity: {ga:.0f} mGal (moderate)")
                            else:
                                st.caption(f"○ Gravity: {ga:.0f} mGal (weak)")
                            
                            # Crustal thickness
                            if ct < 25:
                                st.caption(f"🔥 Crust: {ct:.0f} km (thin - oceanic/rift)")
                            elif ct < 35:
                                st.caption(f"📏 Crust: {ct:.0f} km (normal)")
                            else:
                                st.caption(f"🏔️ Crust: {ct:.0f} km (thick - continental)")
                            
                            # Boundary distance
                            if boundary_dist < 200:
                                st.caption(f"🌋 Boundary: {boundary_dist:.0f} km (very close!)")
                            elif boundary_dist < 500:
                                st.caption(f"⚠️ Boundary: {boundary_dist:.0f} km (near)")
                            else:
                                st.caption(f"○ Boundary: {boundary_dist:.0f} km (far)")
                    
                    with detail_cols[1]:
                        st.write("**H Score Factors:**")
                        h_score = selected_cand['H']
                        
                        # Show score with color coding
                        if h_score > 0.5:
                            st.success(f"H = {h_score:.2f} (Strong)")
                        elif h_score > 0.3:
                            st.info(f"H = {h_score:.2f} (Moderate)")
                        elif h_score > 0.1:
                            st.warning(f"H = {h_score:.2f} (Weak)")
                        else:
                            st.error(f"H = {h_score:.2f} (Minimal)")
                        
                        # Show matched patterns
                        try:
                            matches = gsn.get_H_pattern_matches(
                                selected_cand['lat'], selected_cand['lon'],
                                KNOWN_NODES_EXTENDED if HAS_EXTENDED_NODES else None,
                                top_n=3
                            )
                            if matches:
                                st.caption("**Matched angles:**")
                                for m in matches:
                                    if m['match_score'] >= 0.7:
                                        icon = "✓"
                                    elif m['match_score'] >= 0.4:
                                        icon = "◐"
                                    else:
                                        icon = "○"
                                    st.caption(f"{icon} {m['matched_angle']:.0f}° ({m['angle_type']}) → {m['node_name']}")
                        except Exception:
                            st.caption("Angular pattern matching with known nodes")
                    
                    with detail_cols[2]:
                        st.write("**Overall Assessment:**")
                        f_score = selected_cand['F']
                        classification = gsn.classify_F(f_score)
                        
                        if "Strong" in classification or f_score > 5:
                            st.success(f"**{classification}**")
                        elif "Moderate" in classification or f_score > 2:
                            st.info(f"**{classification}**")
                        else:
                            st.warning(f"**{classification}**")
                        
                        st.caption(f"Region: {selected_cand['region']}")
                        
                        # Google Maps link
                        gmaps_url = f"https://www.google.com/maps?q={selected_cand['lat']},{selected_cand['lon']}"
                        st.link_button("🗺️ View on Google Maps", gmaps_url, use_container_width=True)
                    
                    # Second row: Additional Context
                    st.divider()
                    context_cols = st.columns(3)
                    
                    cand_lat = selected_cand['lat']
                    cand_lon = selected_cand['lon']
                    
                    # Column 1: Nearby Known Nodes (always shown)
                    with context_cols[0]:
                        st.write("**📍 Nearby Known Sites:**")
                        nearby_nodes = get_nearest_known_nodes(cand_lat, cand_lon, top_n=3)
                        for node in nearby_nodes:
                            dist = node['distance_km']
                            if dist < 1000:
                                dist_str = f"{dist:.0f} km"
                            else:
                                dist_str = f"{dist/1000:.1f}k km"
                            st.caption(f"• {node['name']}: {dist_str}")
                    
                    # Column 2: Constellation Visibility (always shown if available)
                    with context_cols[1]:
                        if HAS_ASTRONOMY:
                            st.write("**✨ Constellations:**")
                            const_vis = get_constellation_visibility_summary(cand_lat, cand_lon)[:3]
                            if const_vis:
                                for c in const_vis:
                                    st.caption(f"{c['color']} {c['name']}: {c['score']*100:.0f}%")
                            else:
                                st.caption("• Data unavailable")
                        else:
                            st.write("**📌 Coordinates:**")
                            st.caption(f"• Lat: {cand_lat:.4f}°")
                            st.caption(f"• Lon: {cand_lon:.4f}°")
                    
                    # Column 3: Geology (always shown)
                    with context_cols[2]:
                        st.write("**🌋 Geology:**")
                        geo = get_geology_summary(cand_lat, cand_lon)
                        if geo["boundary_dist"] is not None:
                            st.caption(f"• Plate boundary: {geo['boundary_dist']:.0f} km")
                        st.caption(f"• Seismic: {geo['seismic_level']}")
                        if geo["volcano_dist"]:
                            st.caption(f"• Volcano: {geo['volcano_dist']:.0f} km")
                    
                    # Third row: Geometric Patterns
                    st.divider()
                    pattern_cols = st.columns(2)
                    
                    with pattern_cols[0]:
                        st.write("**📐 Great Circle Alignments:**")
                        try:
                            alignments = gsn.compute_great_circle_alignments(
                                cand_lat, cand_lon,
                                KNOWN_NODES_EXTENDED if HAS_EXTENDED_NODES else None,
                                tolerance=2.0
                            )
                            if alignments["num_alignments"] > 0:
                                st.caption(f"🎯 Lies on {alignments['num_alignments']} alignment lines")
                                for a in alignments["alignments"][:3]:
                                    st.caption(f"• {a['node1']} ↔ {a['node2']}")
                            else:
                                st.caption("○ No major alignments detected")
                        except Exception:
                            st.caption("○ Alignment data unavailable")
                    
                    with pattern_cols[1]:
                        st.write("**🔷 Symmetry Patterns:**")
                        try:
                            symmetry = gsn.compute_symmetry_patterns(
                                cand_lat, cand_lon,
                                KNOWN_NODES_EXTENDED if HAS_EXTENDED_NODES else None
                            )
                            if symmetry["num_patterns"] > 0:
                                for p in symmetry["patterns"][:2]:
                                    st.caption(f"✓ {p['type']}")
                                    st.caption(f"  with {', '.join(p['nodes'][:2])}")
                            else:
                                st.caption("○ No sacred geometry patterns")
                        except Exception:
                            st.caption("○ Pattern data unavailable")
                    
                    # Fourth row: Confidence Intervals (if uncertainty module available)
                    if HAS_UNCERTAINTY and HAS_BAYESIAN:
                        st.divider()
                        st.write("**📊 Confidence Analysis:**")
                        
                        conf_cols = st.columns(3)
                        
                        with conf_cols[0]:
                            try:
                                # Bayesian posterior
                                scorer = BayesianScorer()
                                posterior = scorer.compute_posterior(cand_lat, cand_lon)
                                prob_pct = posterior.probability * 100
                                ci_low = posterior.credible_interval_95[0] * 100
                                ci_high = posterior.credible_interval_95[1] * 100
                                
                                st.metric(
                                    "P(node|data)", 
                                    f"{prob_pct:.3f}%",
                                    help="Bayesian posterior probability"
                                )
                                st.caption(f"95% CI: ({ci_low:.3f}%, {ci_high:.3f}%)")
                            except Exception:
                                st.caption("Bayesian analysis unavailable")
                        
                        with conf_cols[1]:
                            try:
                                # Classify confidence
                                f_score = selected_cand['F']
                                g_score = selected_cand['G']
                                h_score = selected_cand['H']
                                
                                # Rough uncertainty estimate
                                est_std = abs(f_score) * 0.15  # Assume 15% CV
                                
                                quality = compute_data_quality_score(
                                    cand_lat, cand_lon,
                                    {"gravity": True, "crust": True, "boundary": True}
                                )
                                
                                conf = classify_confidence(
                                    f_score, est_std, 
                                    quality['quality_score']
                                )
                                
                                level = conf['confidence_level']
                                if level == "HIGH":
                                    st.success(f"🟢 {level} Confidence")
                                elif level == "MODERATE":
                                    st.info(f"🟡 {level} Confidence")
                                else:
                                    st.warning(f"🔴 {level} Confidence")
                                
                                st.caption(conf['description'])
                            except Exception:
                                st.caption("Confidence unavailable")
                        
                        with conf_cols[2]:
                            st.caption("**Score Components:**")
                            st.caption(f"• G (geophysical): {selected_cand['G']:.3f}")
                            st.caption(f"• H (geometric): {selected_cand['H']:.3f}")
                            st.caption(f"• F (combined): {selected_cand['F']:.3f}")
    
    # Handle Data Status mode
    if mode == "Data Status":
        st.subheader("📊 Data Sources & System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Core Modules**")
            st.write(f"{'✓' if HAS_EXTENDED_NODES else '✗'} Extended Nodes ({len(KNOWN_NODES_EXTENDED) if HAS_EXTENDED_NODES else 0} sites)")
            st.write(f"{'✓' if HAS_DATA_SOURCES else '✗'} Data Sources Module")
            st.write(f"{'✓' if HAS_VALIDATION else '✗'} Validation Framework")
            st.write(f"{'✓' if HAS_UNCERTAINTY else '✗'} Uncertainty Quantification")
            st.write(f"{'✓' if HAS_STATISTICAL_VALIDATION else '✗'} Statistical Validation")
            st.write(f"{'✓' if HAS_BAYESIAN else '✗'} Bayesian Scoring")
        
        with col2:
            st.write("**New Data Sources**")
            st.write(f"{'✓' if HAS_SEISMIC else '✗'} Seismic Density (USGS)")
            st.write(f"{'✓' if HAS_HEATFLOW else '✗'} Heat Flow (IHFC)")
            st.write(f"{'✓' if HAS_VOLCANIC else '✗'} Volcanoes (GVP)")
            st.write(f"{'✓' if HAS_ARCHAEOLOGY else '✗'} Archaeological Sites")
            st.write(f"{'✓' if HAS_STRESS_MAP else '✗'} World Stress Map")
        
        with col3:
            st.write("**Data Files**")
            if HAS_DATA_SOURCES:
                availability = check_data_availability()
                for name, available in list(availability.items())[:6]:
                    st.write(f"{'✓' if available else '✗'} {name}")
            else:
                st.write("Data sources module not loaded")
        
        with col4:
            st.write("**Enhanced Features**")
            st.write(f"{'✓' if hasattr(gsn, 'compute_H_weighted') else '✗'} Distance-Weighted H")
            st.write(f"{'✓' if hasattr(gsn, 'compute_H_full') else '✗'} Full H Scoring")
            st.write(f"{'✓' if HAS_GEOMETRY else '✗'} Geometric Analysis")
            st.write(f"{'✓' if hasattr(gsn, 'EXTENDED_ANGLES') else '✗'} Extended Angles")
            if HAS_VALIDATION:
                st.write("✓ Geographic CV")
            
            # ML Scorer status
            import os
            ml_model_exists = os.path.exists("gsn_ml_grid_scorer.pth")
            if HAS_ML_SCORER:
                if ml_model_exists:
                    st.write("✓ ML Grid Scorer (trained)")
                else:
                    st.write("◐ ML Grid Scorer (not trained)")
            else:
                st.write("✗ ML Grid Scorer (PyTorch required)")
        
        st.divider()
        
        # New data source statistics
        st.write("**Data Source Statistics**")
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            if HAS_HEATFLOW:
                hf_stats = get_heatflow_stats()
                if hf_stats.get("available"):
                    st.metric("Heat Flow Measurements", f"{hf_stats['n_measurements']:,}")
                    st.caption(f"Range: {hf_stats['min_heatflow']:.0f}-{hf_stats['max_heatflow']:.0f} mW/m²")
        
        with stat_cols[1]:
            if HAS_VOLCANIC:
                vol_stats = get_volcano_stats()
                if vol_stats.get("available"):
                    st.metric("Holocene Volcanoes", f"{vol_stats['n_volcanoes']:,}")
                    st.caption(f"Countries: {vol_stats['n_countries']}")
        
        with stat_cols[2]:
            if HAS_ARCHAEOLOGY:
                arch_stats = get_site_stats()
                if arch_stats.get("available"):
                    st.metric("Archaeological Sites", f"{arch_stats['n_sites']:,}")
                    st.caption(f"Categories: {arch_stats['n_categories']}")
        
        with stat_cols[3]:
            if HAS_GEOMETRY:
                geo_stats = get_geometry_stats()
                if geo_stats.get("available"):
                    st.metric("Great Circle Alignments", f"{geo_stats.get('n_alignments_4plus', 0)}")
                    st.caption(f"Max strength: {geo_stats.get('max_alignment_strength', 0)} sites")
            elif HAS_SEISMIC:
                st.metric("Seismic Data", "✓ Available")
                st.caption("USGS Earthquake Catalog")
        
        st.divider()
        
        # Show node categories
        if HAS_EXTENDED_NODES:
            st.write("**Reference Node Categories**")
            cols = st.columns(3)
            cat_items = list(CATEGORIES.items())
            for i, (cat, desc) in enumerate(cat_items):
                from known_nodes_extended import get_nodes_by_category
                nodes = get_nodes_by_category(cat)
                with cols[i % 3]:
                    st.write(f"• {desc}: {len(nodes)}")
        
        # Archaeological site categories
        if HAS_ARCHAEOLOGY:
            st.write("**Archaeological Site Weights**")
            weight_cols = st.columns(5)
            for i, (cat, weight) in enumerate(sorted(SITE_WEIGHTS.items(), key=lambda x: -x[1])):
                with weight_cols[i % 5]:
                    st.write(f"{cat}: {weight:.1f}")
        
        st.divider()
        
        # Geographic regions
        if HAS_VALIDATION:
            st.write("**Geographic Regions for Stratified CV**")
            region_names = list(GEOGRAPHIC_REGIONS.keys())
            st.write(", ".join(region_names))
        
        st.divider()
        
        # Validation metrics if candidates exist
        if HAS_VALIDATION and st.session_state.candidates:
            st.write("**Validation Metrics**")
            known = get_coords_dict() if HAS_EXTENDED_NODES else gsn.KNOWN_NODES
            
            recall_50 = compute_recall_at_k(
                st.session_state.candidates, known, 
                threshold_km=100, k=50
            )
            
            dist_stats = compute_average_distance_to_nearest(
                st.session_state.candidates, known
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recall@50", f"{recall_50['recall']:.1%}")
                st.write(f"({recall_50['n_hits']}/{recall_50['n_nodes']} nodes matched)")
            with col2:
                st.metric("Mean Distance", f"{dist_stats['mean_dist_km']:.0f} km")
                st.write(f"Median: {dist_stats['median_dist_km']:.0f} km")
    
    # Footer
    st.divider()
    st.caption("GSN Node Predictor (Enhanced) | Powered by Streamlit & Leaflet/OpenStreetMap")


if __name__ == "__main__":
    main()

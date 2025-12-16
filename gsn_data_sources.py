#!/usr/bin/env python
"""
GSN Data Sources Module

Unified data loading and management for multiple geophysical datasets:
- Gravity anomaly (NetCDF)
- Crustal thickness (NetCDF)
- Magnetic anomaly (EMAG2 CSV)
- Plate boundaries (KMZ)
- Topography/Elevation (ETOPO NetCDF)
- Seismic density (computed from earthquake catalogs)
- Heat flow (interpolated from point measurements)

Each data source is loaded lazily and cached for performance.
"""

import os
import math
import zipfile
from xml.etree import ElementTree as ET
from typing import Dict, Optional, Tuple, List, Any

import numpy as np

# Optional imports
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from scipy import ndimage, interpolate
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------
# Configuration: Data file paths
# ---------------------------------------------------------
DATA_CONFIG = {
    "gravity": {
        "file": "gravity_model.nc",
        "var_name": "ga",
        "description": "Gravity anomaly (mGal)",
    },
    "crust": {
        "file": "crust_model.nc",
        "var_name": "thickness",
        "description": "Crustal thickness (km)",
    },
    "magnetic": {
        "file": "EMAG2_V3_20170530.csv",
        "description": "Magnetic anomaly (nT)",
    },
    "boundaries": {
        "file": "gsn_data/plate-boundaries.kmz",
        "description": "Plate boundary polylines",
    },
    "topography": {
        "file": "gsn_data/Global_grids___World_Gravity_Map/WGM2012_ETOPO1_ponc_2min.grd",
        "description": "Topography/Bathymetry (m)",
    },
    "bouguer": {
        "file": "gsn_data/Global_grids___World_Gravity_Map/WGM2012_Bouguer_ponc_2min.grd",
        "description": "Bouguer gravity anomaly (mGal)",
    },
    "freeair": {
        "file": "gsn_data/Global_grids___World_Gravity_Map/WGM2012_Freeair_ponc_2min.grd",
        "description": "Free-air gravity anomaly (mGal)",
    },
    "isostatic": {
        "file": "gsn_data/Global_grids___World_Gravity_Map/WGM2012_Isostatic_ponc_2min.grd",
        "description": "Isostatic gravity anomaly (mGal)",
    },
    "ocean_age": {
        "file": "crustal age/age.3.6.tif",
        "description": "Ocean floor age (Ma)",
        "nodata": 32767,
    },
    "heatflow": {
        "file": "GHFDB-R2024/IHFC_2024_GHFDB.xlsx",
        "description": "Global heat flow (mW/m²)",
    },
    "volcanoes": {
        "file": "GVP_Volcano_List_Holocene_202512122258.xls",
        "description": "Holocene volcanoes (GVP)",
    },
}


# ---------------------------------------------------------
# Extended G Component Configuration
# ---------------------------------------------------------
G_COMPONENTS = {
    "ga": {
        "weight": 1.0,
        "scale": 30.0,
        "normalization": "divide",
        "description": "Gravity anomaly",
        "source": "gravity",
    },
    "ct": {
        "weight": 1.0,
        "mean": 35.0,
        "std": 10.0,
        "normalization": "zscore",
        "description": "Crustal thickness",
        "source": "crust",
    },
    "tb": {
        "weight": 1.0,
        "L": 800.0,
        "normalization": "gaussian_decay",
        "description": "Tectonic boundary proximity",
        "source": "boundaries",
    },
    "ma": {
        "weight": 0.5,
        "scale": 100.0,
        "normalization": "divide",
        "description": "Magnetic anomaly",
        "source": "magnetic",
    },
    "el": {
        "weight": 0.2,
        "scale": 1000.0,
        "normalization": "divide",
        "description": "Elevation/Topography",
        "source": "topography",
    },
    "bg": {
        "weight": 0.3,
        "scale": 50.0,
        "normalization": "divide",
        "description": "Bouguer anomaly",
        "source": "bouguer",
    },
    "iso": {
        "weight": 0.2,
        "scale": 30.0,
        "normalization": "divide",
        "description": "Isostatic anomaly",
        "source": "isostatic",
    },
    # Gradient features (derived)
    "ga_grad": {
        "weight": 0.3,
        "scale": 5.0,
        "normalization": "divide",
        "description": "Gravity gradient magnitude",
        "source": "gravity_gradient",
    },
    "ga_lap": {
        "weight": 0.2,
        "scale": 10.0,
        "normalization": "divide_abs",
        "description": "Gravity Laplacian",
        "source": "gravity_laplacian",
    },
    "ma_grad": {
        "weight": 0.2,
        "scale": 20.0,
        "normalization": "divide",
        "description": "Magnetic gradient magnitude",
        "source": "magnetic_gradient",
    },
    # Seismic density
    "sd": {
        "weight": 0.4,
        "scale": 3.0,
        "normalization": "divide",
        "description": "Seismic density (log)",
        "source": "seismic",
    },
    # Heat flow
    "hf": {
        "weight": 0.4,
        "scale": 80.0,
        "normalization": "divide",
        "description": "Heat flow anomaly",
        "source": "heatflow",
    },
    # Volcanic proximity
    "vd": {
        "weight": 0.3,
        "L": 200.0,
        "normalization": "gaussian_decay",
        "description": "Volcanic proximity",
        "source": "volcanic",
    },
    # Ocean floor age
    "oa": {
        "weight": 0.2,
        "scale": 100.0,
        "normalization": "divide",
        "description": "Ocean floor age (Ma)",
        "source": "ocean_age",
    },
}


# ---------------------------------------------------------
# Data Cache
# ---------------------------------------------------------
class DataCache:
    """Singleton cache for loaded datasets."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._loaded = set()
        return cls._instance
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
        self._loaded.add(key)
    
    def is_loaded(self, key):
        return key in self._loaded
    
    def clear(self):
        self._cache.clear()
        self._loaded.clear()


_cache = DataCache()


# ---------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------

def get_base_path():
    """Get the base path for data files."""
    return os.path.dirname(os.path.abspath(__file__))


def load_netcdf_dataset(name: str) -> Optional[Any]:
    """Load a NetCDF dataset by config name."""
    if not HAS_XARRAY:
        print(f"[ERROR] xarray required to load {name}")
        return None
    
    if _cache.is_loaded(name):
        return _cache.get(name)
    
    config = DATA_CONFIG.get(name)
    if not config:
        print(f"[ERROR] Unknown dataset: {name}")
        return None
    
    filepath = os.path.join(get_base_path(), config["file"])
    if not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return None
    
    try:
        ds = xr.open_dataset(filepath)
        _cache.set(name, ds)
        print(f"[INFO] Loaded {name}: {config['description']}")
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return None


def load_grd_file(name: str) -> Optional[Any]:
    """Load a GMT .grd file (NetCDF format)."""
    if not HAS_XARRAY:
        print(f"[ERROR] xarray required to load {name}")
        return None
    
    if _cache.is_loaded(name):
        return _cache.get(name)
    
    config = DATA_CONFIG.get(name)
    if not config:
        return None
    
    filepath = os.path.join(get_base_path(), config["file"])
    if not os.path.exists(filepath):
        print(f"[WARN] GRD file not found: {filepath}")
        return None
    
    try:
        ds = xr.open_dataset(filepath)
        _cache.set(name, ds)
        print(f"[INFO] Loaded {name}: {config['description']}")
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load GRD {name}: {e}")
        return None


def load_magnetic_data() -> Optional[Dict]:
    """
    Load EMAG2 magnetic anomaly data from CSV.
    Returns a dict with 'lats', 'lons', 'values' arrays and a KDTree for lookup.
    """
    if _cache.is_loaded("magnetic"):
        return _cache.get("magnetic")
    
    if not HAS_PANDAS or not HAS_SCIPY:
        print("[ERROR] pandas and scipy required for magnetic data")
        return None
    
    filepath = os.path.join(get_base_path(), DATA_CONFIG["magnetic"]["file"])
    if not os.path.exists(filepath):
        print(f"[WARN] Magnetic data file not found: {filepath}")
        return None
    
    try:
        print("[INFO] Loading EMAG2 magnetic anomaly data (this may take a moment)...")
        
        # CSV columns: index, ?, lon, lat, ?, magnetic_anomaly, ?, ?
        df = pd.read_csv(filepath, header=None, 
                         names=["idx", "n", "lon", "lat", "flag", "mag", "src", "age"])
        
        # Filter out no-data values
        df = df[df["flag"] != 99999]
        
        # Subsample for memory efficiency (every 10th point)
        df = df.iloc[::10].copy()
        
        lons = df["lon"].values
        lats = df["lat"].values
        mags = df["mag"].values
        
        # Build KDTree for fast nearest-neighbor lookup
        # Convert to radians for spherical distance
        coords_rad = np.column_stack([
            np.radians(lats),
            np.radians(lons)
        ])
        tree = cKDTree(coords_rad)
        
        result = {
            "lons": lons,
            "lats": lats,
            "values": mags,
            "tree": tree,
        }
        _cache.set("magnetic", result)
        print(f"[INFO] Loaded magnetic data: {len(mags):,} points")
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to load magnetic data: {e}")
        return None


def load_plate_boundaries() -> List[List[Tuple[float, float]]]:
    """Load plate boundary polylines from KMZ file."""
    if _cache.is_loaded("boundaries"):
        return _cache.get("boundaries")
    
    filepath = os.path.join(get_base_path(), DATA_CONFIG["boundaries"]["file"])
    if not os.path.exists(filepath):
        print(f"[WARN] Plate boundary file not found: {filepath}")
        return []
    
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
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
        
        _cache.set("boundaries", polylines)
        print(f"[INFO] Loaded {len(polylines)} plate boundary polylines")
        return polylines
        
    except Exception as e:
        print(f"[ERROR] Failed to load plate boundaries: {e}")
        return []


# ---------------------------------------------------------
# Data Query Functions
# ---------------------------------------------------------

def get_gravity_anomaly(lat: float, lon: float) -> Optional[float]:
    """Get gravity anomaly value at a point."""
    ds = load_netcdf_dataset("gravity")
    if ds is None:
        return None
    
    var_name = DATA_CONFIG["gravity"]["var_name"]
    try:
        value = ds[var_name].sel(lat=lat, lon=lon, method="nearest")
        return float(value.values)
    except Exception:
        return None


def get_crustal_thickness(lat: float, lon: float) -> Optional[float]:
    """Get crustal thickness value at a point."""
    ds = load_netcdf_dataset("crust")
    if ds is None:
        return None
    
    var_name = DATA_CONFIG["crust"]["var_name"]
    try:
        value = ds[var_name].sel(lat=lat, lon=lon, method="nearest")
        return float(value.values)
    except Exception:
        return None


def get_magnetic_anomaly(lat: float, lon: float) -> Optional[float]:
    """Get magnetic anomaly value at a point using nearest-neighbor."""
    data = load_magnetic_data()
    if data is None:
        return None
    
    # Query KDTree
    query_rad = np.array([[np.radians(lat), np.radians(lon)]])
    _, idx = data["tree"].query(query_rad, k=1)
    return float(data["values"][idx[0]])


def get_topography(lat: float, lon: float) -> Optional[float]:
    """Get elevation/topography value at a point."""
    ds = load_grd_file("topography")
    if ds is None:
        return None
    
    try:
        # GRD files typically have 'z' as the variable name
        var_names = list(ds.data_vars)
        if var_names:
            value = ds[var_names[0]].sel(lat=lat, lon=lon, method="nearest")
            return float(value.values)
    except Exception:
        pass
    return None


def get_bouguer_anomaly(lat: float, lon: float) -> Optional[float]:
    """Get Bouguer gravity anomaly at a point."""
    ds = load_grd_file("bouguer")
    if ds is None:
        return None
    
    try:
        var_names = list(ds.data_vars)
        if var_names:
            value = ds[var_names[0]].sel(lat=lat, lon=lon, method="nearest")
            return float(value.values)
    except Exception:
        pass
    return None


def get_isostatic_anomaly(lat: float, lon: float) -> Optional[float]:
    """Get isostatic gravity anomaly at a point."""
    ds = load_grd_file("isostatic")
    if ds is None:
        return None
    
    try:
        var_names = list(ds.data_vars)
        if var_names:
            value = ds[var_names[0]].sel(lat=lat, lon=lon, method="nearest")
            return float(value.values)
    except Exception:
        pass
    return None


def get_boundary_distance(lat: float, lon: float, boundaries: List = None) -> float:
    """
    Calculate distance to nearest plate boundary in km.
    Uses great-circle distance approximation.
    """
    if boundaries is None:
        boundaries = load_plate_boundaries()
    
    if not boundaries:
        return 500.0  # Default fallback
    
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    min_dist = float("inf")
    for poly in boundaries:
        for pt in poly:
            d = haversine_km(lat, lon, pt[0], pt[1])
            if d < min_dist:
                min_dist = d
    
    return min_dist if min_dist < float("inf") else 500.0


# ---------------------------------------------------------
# Gradient and Derivative Features
# ---------------------------------------------------------

# Cached gradient grids
_gradient_cache = {}


def compute_gradient_features(grid: np.ndarray, name: str = "unnamed") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial gradient features from a 2D grid.
    
    Args:
        grid: 2D numpy array of values
        name: Name for caching
    
    Returns:
        Tuple of (gradient_magnitude, laplacian)
        - gradient_magnitude: Edge strength (first derivative)
        - laplacian: Second derivative (highlights boundaries/anomalies)
    """
    cache_key = f"{name}_gradient"
    if cache_key in _gradient_cache:
        return _gradient_cache[cache_key]
    
    if not HAS_SCIPY:
        print("[WARN] scipy required for gradient features")
        return np.zeros_like(grid), np.zeros_like(grid)
    
    from scipy.ndimage import sobel, laplace
    
    # Handle NaN values
    grid_clean = np.nan_to_num(grid, nan=0.0)
    
    # Sobel gradient (first derivative) in both directions
    grad_y = sobel(grid_clean, axis=0)  # latitude direction
    grad_x = sobel(grid_clean, axis=1)  # longitude direction
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian (second derivative)
    laplacian = laplace(grid_clean)
    
    _gradient_cache[cache_key] = (gradient_magnitude, laplacian)
    
    return gradient_magnitude, laplacian


def get_gravity_gradient(lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Get gravity gradient magnitude and Laplacian at a point.
    
    Returns:
        Tuple of (gradient_magnitude, laplacian)
    """
    ds = load_netcdf_dataset("gravity")
    if ds is None:
        return None, None
    
    var_name = DATA_CONFIG["gravity"]["var_name"]
    
    try:
        # Get the full grid for gradient computation
        cache_key = "gravity_gradient_grid"
        if cache_key not in _gradient_cache:
            grid = ds[var_name].values
            grad_mag, lap = compute_gradient_features(grid, "gravity")
            lats = ds.coords["lat"].values
            lons = ds.coords["lon"].values
            _gradient_cache[cache_key] = (grad_mag, lap, lats, lons)
        else:
            grad_mag, lap, lats, lons = _gradient_cache[cache_key]
        
        # Find nearest indices
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        return float(grad_mag[lat_idx, lon_idx]), float(lap[lat_idx, lon_idx])
        
    except Exception as e:
        print(f"[WARN] Gradient computation failed: {e}")
        return None, None


def get_magnetic_gradient(lat: float, lon: float) -> Optional[float]:
    """
    Get magnetic anomaly gradient at a point.
    
    For CSV-based magnetic data, we compute local gradient from nearby points.
    """
    data = load_magnetic_data()
    if data is None:
        return None
    
    try:
        # Find nearby points and compute local gradient
        query_rad = np.array([[np.radians(lat), np.radians(lon)]])
        distances, indices = data["tree"].query(query_rad, k=5)
        
        if len(indices[0]) < 3:
            return 0.0
        
        # Get values at nearby points
        nearby_values = data["values"][indices[0]]
        nearby_lats = data["lats"][indices[0]]
        nearby_lons = data["lons"][indices[0]]
        
        # Simple gradient estimate: std of nearby values / mean distance
        value_range = np.std(nearby_values)
        mean_dist = np.mean(distances[0]) * 6371  # Convert to km (approximate)
        
        if mean_dist > 0:
            gradient = value_range / max(mean_dist, 1.0)
        else:
            gradient = 0.0
        
        return float(gradient)
        
    except Exception:
        return None


def get_seismic_density(lat: float, lon: float) -> Optional[float]:
    """
    Get seismic density at a point.
    
    Returns log-scaled earthquake density from the seismic module.
    """
    try:
        from gsn_seismic import get_seismic_density as _get_sd, get_cached_seismic_grid
        grid_data = get_cached_seismic_grid()
        return _get_sd(lat, lon, grid_data)
    except ImportError:
        return None
    except Exception as e:
        print(f"[WARN] Seismic density lookup failed: {e}")
        return None


def get_heatflow(lat: float, lon: float) -> Optional[float]:
    """
    Get heat flow value at a point (mW/m²).
    
    Returns interpolated heat flow from IHFC Global Heat Flow Database.
    """
    try:
        from gsn_heatflow import get_heatflow as _get_hf
        return _get_hf(lat, lon)
    except ImportError:
        return None
    except Exception as e:
        print(f"[WARN] Heat flow lookup failed: {e}")
        return None


def get_volcanic_distance(lat: float, lon: float) -> Optional[float]:
    """
    Get distance to nearest volcano in km.
    
    Returns distance from Smithsonian GVP Holocene volcano database.
    """
    try:
        from gsn_volcanic import get_distance_to_nearest_volcano
        return get_distance_to_nearest_volcano(lat, lon)
    except ImportError:
        return None
    except Exception as e:
        print(f"[WARN] Volcanic distance lookup failed: {e}")
        return None


# Ocean age raster cache
_ocean_age_cache = None


def load_ocean_age_raster():
    """Load ocean floor age GeoTIFF."""
    global _ocean_age_cache
    
    if _ocean_age_cache is not None:
        return _ocean_age_cache
    
    try:
        import rasterio
    except ImportError:
        print("[WARN] rasterio required for ocean age data. Install with: pip install rasterio")
        return None
    
    filepath = os.path.join(get_base_path(), DATA_CONFIG["ocean_age"]["file"])
    if not os.path.exists(filepath):
        print(f"[WARN] Ocean age file not found: {filepath}")
        return None
    
    try:
        src = rasterio.open(filepath)
        data = src.read(1)
        nodata = DATA_CONFIG["ocean_age"].get("nodata", 32767)
        
        # Replace nodata with NaN
        data = data.astype(float)
        data[data >= nodata] = np.nan
        
        # Get transform for coordinate conversion
        transform = src.transform
        bounds = src.bounds
        
        _ocean_age_cache = {
            "data": data,
            "transform": transform,
            "bounds": bounds,
            "shape": data.shape,
        }
        
        print(f"[INFO] Loaded ocean age raster: {data.shape}")
        return _ocean_age_cache
        
    except Exception as e:
        print(f"[ERROR] Failed to load ocean age data: {e}")
        return None


def get_ocean_age(lat: float, lon: float) -> Optional[float]:
    """
    Get ocean floor age at a point in Ma (million years).
    
    Returns None for continental crust (no age data).
    """
    cache = load_ocean_age_raster()
    if cache is None:
        return None
    
    data = cache["data"]
    bounds = cache["bounds"]
    
    # Check if point is within bounds
    if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
        return None
    
    # Convert lat/lon to pixel indices
    # The raster covers -180 to 180, -90 to 90
    nrows, ncols = cache["shape"]
    
    # Calculate pixel indices (assuming regular grid)
    col = int((lon - bounds.left) / (bounds.right - bounds.left) * ncols)
    row = int((bounds.top - lat) / (bounds.top - bounds.bottom) * nrows)
    
    # Clamp to valid range
    row = max(0, min(nrows - 1, row))
    col = max(0, min(ncols - 1, col))
    
    value = data[row, col]
    
    if np.isnan(value):
        return None
    
    return float(value)


# ---------------------------------------------------------
# Extended G Scoring
# ---------------------------------------------------------

def normalize_component(value: float, config: Dict) -> float:
    """Normalize a component value based on its configuration."""
    norm_type = config.get("normalization", "divide")
    
    if value is None:
        return 0.0
    
    if norm_type == "divide":
        scale = config.get("scale", 1.0)
        return value / scale
    
    elif norm_type == "zscore":
        mean = config.get("mean", 0.0)
        std = config.get("std", 1.0)
        return (value - mean) / std
    
    elif norm_type == "gaussian_decay":
        L = config.get("L", 1.0)
        return math.exp(-(value ** 2) / (2.0 * L * L))
    
    elif norm_type == "divide_abs":
        # For values that can be negative (like Laplacian)
        scale = config.get("scale", 1.0)
        return abs(value) / scale
    
    else:
        return value


def get_all_components(lat: float, lon: float, boundaries: List = None) -> Dict[str, float]:
    """
    Get all geophysical component values at a point.
    Returns a dict of {component_name: raw_value}.
    """
    components = {}
    
    # Gravity anomaly
    ga = get_gravity_anomaly(lat, lon)
    components["ga"] = ga if ga is not None else 0.0
    
    # Crustal thickness
    ct = get_crustal_thickness(lat, lon)
    components["ct"] = ct if ct is not None else 35.0
    
    # Tectonic boundary distance
    tb = get_boundary_distance(lat, lon, boundaries)
    components["tb"] = tb
    
    # Magnetic anomaly
    ma = get_magnetic_anomaly(lat, lon)
    components["ma"] = ma if ma is not None else 0.0
    
    # Topography
    el = get_topography(lat, lon)
    components["el"] = el if el is not None else 0.0
    
    # Bouguer anomaly
    bg = get_bouguer_anomaly(lat, lon)
    components["bg"] = bg if bg is not None else 0.0
    
    # Isostatic anomaly
    iso = get_isostatic_anomaly(lat, lon)
    components["iso"] = iso if iso is not None else 0.0
    
    # Gradient features
    ga_grad, ga_lap = get_gravity_gradient(lat, lon)
    components["ga_grad"] = ga_grad if ga_grad is not None else 0.0
    components["ga_lap"] = ga_lap if ga_lap is not None else 0.0
    
    # Magnetic gradient
    ma_grad = get_magnetic_gradient(lat, lon)
    components["ma_grad"] = ma_grad if ma_grad is not None else 0.0
    
    # Seismic density
    sd = get_seismic_density(lat, lon)
    components["sd"] = sd if sd is not None else 0.0
    
    # Heat flow
    hf = get_heatflow(lat, lon)
    components["hf"] = hf if hf is not None else 65.0  # Global average ~65 mW/m²
    
    # Volcanic distance
    vd = get_volcanic_distance(lat, lon)
    components["vd"] = vd if vd is not None else 500.0  # Default far from volcanoes
    
    # Ocean floor age
    oa = get_ocean_age(lat, lon)
    components["oa"] = oa if oa is not None else 0.0  # 0 for continental crust
    
    return components


def compute_G_extended(
    lat: float, 
    lon: float,
    component_config: Dict = None,
    boundaries: List = None,
) -> Tuple[float, Dict]:
    """
    Compute extended geophysical suitability score G.
    
    Args:
        lat, lon: Coordinates
        component_config: Override default G_COMPONENTS config
        boundaries: Pre-loaded plate boundaries
    
    Returns:
        (G_score, components_dict) where components_dict has raw and normalized values
    """
    config = component_config or G_COMPONENTS
    
    # Get raw component values
    raw = get_all_components(lat, lon, boundaries)
    
    # Normalize and weight each component
    G = 0.0
    normalized = {}
    
    for comp_name, comp_config in config.items():
        if comp_name not in raw:
            continue
        
        raw_val = raw[comp_name]
        norm_val = normalize_component(raw_val, comp_config)
        weight = comp_config.get("weight", 1.0)
        
        normalized[comp_name] = norm_val
        G += weight * norm_val
    
    result = {
        "raw": raw,
        "normalized": normalized,
    }
    
    return G, result


# ---------------------------------------------------------
# Data Availability Check
# ---------------------------------------------------------

def check_data_availability() -> Dict[str, bool]:
    """Check which data sources are available."""
    available = {}
    base = get_base_path()
    
    for name, config in DATA_CONFIG.items():
        filepath = os.path.join(base, config["file"])
        available[name] = os.path.exists(filepath)
    
    return available


def print_data_status():
    """Print status of all data sources."""
    available = check_data_availability()
    
    print("\n=== GSN Data Sources Status ===")
    for name, config in DATA_CONFIG.items():
        status = "✓ Available" if available[name] else "✗ Missing"
        print(f"  {name:12s}: {status} - {config['description']}")
    print()


# ---------------------------------------------------------
# Grid Operations
# ---------------------------------------------------------

def get_grid_coordinates(source: str = "gravity") -> Tuple[np.ndarray, np.ndarray]:
    """Get lat/lon coordinate arrays from a gridded dataset."""
    ds = load_netcdf_dataset(source)
    if ds is None:
        ds = load_grd_file(source)
    
    if ds is None:
        return None, None
    
    lats = ds.coords.get("lat", ds.coords.get("y"))
    lons = ds.coords.get("lon", ds.coords.get("x"))
    
    if lats is not None and lons is not None:
        return lats.values, lons.values
    
    return None, None


if __name__ == "__main__":
    print_data_status()
    
    # Test point query
    test_lat, test_lon = 29.9792, 31.1342  # Giza
    print(f"\nTest point: Giza ({test_lat}, {test_lon})")
    
    components = get_all_components(test_lat, test_lon)
    print("\nRaw component values:")
    for name, value in components.items():
        if value is not None:
            print(f"  {name}: {value:.2f}")
    
    G, details = compute_G_extended(test_lat, test_lon)
    print(f"\nExtended G score: {G:.4f}")

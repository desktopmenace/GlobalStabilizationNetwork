"""
GSN Temporal Analysis Module

Analyzes astronomical alignments across different historical epochs,
accounting for Earth's axial precession (~25,772 year cycle).

Key features:
- Precession calculation for any historical year
- Historical pole star identification
- Constellation visibility for past epochs
- Orion correlation theory analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from itertools import combinations

# Try to import skyfield for precise calculations
try:
    from skyfield.api import load
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False

# Precession constants
PRECESSION_PERIOD = 25772  # years for one complete cycle
PRECESSION_RATE = 50.3     # arcseconds per year
CURRENT_YEAR = 2024

# Historical epochs of interest (negative = BCE)
EPOCHS = {
    "current": {
        "year": 2024,
        "name": "Current Era",
        "description": "Present day",
    },
    "giza_pyramids": {
        "year": -2560,
        "name": "Giza Pyramids",
        "description": "Construction of Great Pyramid (~2560 BCE)",
    },
    "gobekli_tepe": {
        "year": -9500,
        "name": "Gobekli Tepe",
        "description": "Construction of Gobekli Tepe (~9500 BCE)",
    },
    "younger_dryas_end": {
        "year": -9700,
        "name": "Younger Dryas End",
        "description": "End of Younger Dryas period (~9700 BCE)",
    },
    "younger_dryas_start": {
        "year": -10800,
        "name": "Younger Dryas Start",
        "description": "Start of Younger Dryas period (~10800 BCE)",
    },
    "orion_nadir": {
        "year": -10500,
        "name": "Orion Nadir",
        "description": "Orion's belt at lowest point in precessional cycle",
    },
    "last_glacial_max": {
        "year": -20000,
        "name": "Last Glacial Maximum",
        "description": "Peak of last ice age (~20000 BCE)",
    },
}

# Pole stars through precessional cycle
POLE_STARS = [
    {"year_range": (2000, 4000), "star": "Polaris", "constellation": "Ursa Minor"},
    {"year_range": (1000, 2000), "star": "Polaris", "constellation": "Ursa Minor"},
    {"year_range": (-1000, 1000), "star": "Kochab", "constellation": "Ursa Minor"},
    {"year_range": (-3000, -1000), "star": "Thuban", "constellation": "Draco"},
    {"year_range": (-5000, -3000), "star": "Thuban", "constellation": "Draco"},
    {"year_range": (-8000, -5000), "star": "Tau Herculis", "constellation": "Hercules"},
    {"year_range": (-12000, -8000), "star": "Vega", "constellation": "Lyra"},
    {"year_range": (10000, 14000), "star": "Vega", "constellation": "Lyra"},  # Future
]

# Sacred constellations with J2000 coordinates
SACRED_CONSTELLATIONS = {
    "orion": {
        "center_ra": 83.0,  # degrees
        "center_dec": -1.0,
        "sacred_significance": "Hunter, death/rebirth, pyramid alignment",
    },
    "orion_belt": {
        "center_ra": 84.0,
        "center_dec": -1.2,
        "stars": [
            {"name": "Alnitak", "ra": 85.19, "dec": -1.94},
            {"name": "Alnilam", "ra": 84.05, "dec": -1.20},
            {"name": "Mintaka", "ra": 83.00, "dec": -0.30},
        ],
        "sacred_significance": "Three kings, pyramid correlation",
    },
    "pleiades": {
        "center_ra": 56.6,
        "center_dec": 24.1,
        "sacred_significance": "Seven sisters, agricultural calendar",
    },
    "draco": {
        "center_ra": 260.0,
        "center_dec": 65.0,
        "sacred_significance": "Dragon, pole star (Thuban), cosmic serpent",
    },
    "cygnus": {
        "center_ra": 305.0,
        "center_dec": 40.0,
        "sacred_significance": "Northern cross, soul's journey",
    },
    "ursa_major": {
        "center_ra": 180.0,
        "center_dec": 55.0,
        "sacred_significance": "Great bear, circumpolar navigation",
    },
}


class TemporalAnalyzer:
    """
    Analyzes astronomical alignments across different epochs,
    accounting for precession of the equinoxes.
    """
    
    def __init__(self):
        self._timescale = None
        self._ephemeris = None
        
        if HAS_SKYFIELD:
            try:
                self._timescale = load.timescale()
                self._ephemeris = load('de421.bsp')
            except Exception as e:
                print(f"[WARN] Could not load ephemeris: {e}")
    
    def get_precession_offset(self, year: int) -> float:
        """
        Calculate precession offset in degrees for a given year.
        
        Precession causes the celestial pole to trace a circle
        over ~25,772 years, and shifts the vernal equinox.
        
        Args:
            year: Calendar year (negative for BCE)
        
        Returns:
            Offset in degrees to apply to RA coordinates
        """
        years_from_j2000 = year - 2000
        
        # Precession rate: ~50.3 arcseconds per year
        offset_arcsec = PRECESSION_RATE * years_from_j2000
        offset_deg = offset_arcsec / 3600
        
        return offset_deg
    
    def get_historical_pole_star(self, year: int) -> Dict:
        """
        Determine which star was closest to celestial north pole
        in a given year.
        
        Args:
            year: Calendar year (negative for BCE)
        
        Returns:
            Dict with star name and constellation
        """
        for pole_data in POLE_STARS:
            low, high = pole_data["year_range"]
            if low <= year <= high:
                return {
                    "star": pole_data["star"],
                    "constellation": pole_data["constellation"],
                    "year": year,
                }
        
        # Default for very ancient times
        return {
            "star": "Unknown",
            "constellation": "Unknown",
            "year": year,
        }
    
    def adjust_coordinates_for_precession(self, ra: float, dec: float, 
                                          target_year: int) -> Tuple[float, float]:
        """
        Adjust J2000 coordinates for precession to a target year.
        
        This is a simplified model - full precession also affects declination
        and involves nutation, but for visualization RA shift is dominant.
        
        Args:
            ra: Right ascension in degrees (J2000)
            dec: Declination in degrees (J2000)
            target_year: Target year for coordinates
        
        Returns:
            Adjusted (ra, dec) tuple
        """
        offset = self.get_precession_offset(target_year)
        
        # Apply RA offset (primary effect of precession)
        adjusted_ra = (ra + offset) % 360
        
        # Declination is also affected but less dramatically for most stars
        # Full calculation would use precession matrix
        adjusted_dec = dec
        
        return adjusted_ra, adjusted_dec
    
    def compute_constellation_visibility(self, lat: float, lon: float,
                                         target_year: int = CURRENT_YEAR) -> Dict[str, float]:
        """
        Compute visibility of sacred constellations for a given location and epoch.
        
        Args:
            lat, lon: Observer location
            target_year: Historical year
        
        Returns:
            Dict mapping constellation name to visibility score (0-1)
        """
        visibility = {}
        
        for const_name, const_data in SACRED_CONSTELLATIONS.items():
            ra = const_data.get("center_ra", 0)
            dec = const_data.get("center_dec", 0)
            
            # Adjust for precession
            adj_ra, adj_dec = self.adjust_coordinates_for_precession(ra, dec, target_year)
            
            # Simple visibility based on declination and latitude
            # A constellation is visible if its declination is within range
            max_visible_dec = lat + 90
            min_visible_dec = lat - 90
            
            if min_visible_dec <= adj_dec <= max_visible_dec:
                # Calculate how high it gets (max altitude)
                max_altitude = 90 - abs(lat - adj_dec)
                visibility[const_name] = min(1.0, max(0, max_altitude / 45))
            else:
                visibility[const_name] = 0.0
        
        return visibility
    
    def compute_solstice_azimuth(self, lat: float, target_year: int = CURRENT_YEAR) -> Dict:
        """
        Compute summer/winter solstice sunrise/sunset azimuths.
        
        The obliquity of Earth's axis changes slightly over time,
        affecting solstice positions.
        
        Args:
            lat: Observer latitude
            target_year: Historical year
        
        Returns:
            Dict with solstice azimuth information
        """
        # Obliquity varies between ~22.1° and ~24.5° over ~41,000 year cycle
        # Current: ~23.44°
        # Simplified model
        years_from_present = abs(target_year - CURRENT_YEAR)
        obliquity_variation = 1.2 * np.sin(2 * np.pi * years_from_present / 41000)
        obliquity = 23.44 + obliquity_variation
        
        # Sunrise azimuth at summer solstice
        lat_rad = np.radians(lat)
        obliquity_rad = np.radians(obliquity)
        
        # Simplified sunrise azimuth formula
        cos_az = np.sin(obliquity_rad) / np.cos(lat_rad)
        cos_az = np.clip(cos_az, -1, 1)
        
        summer_sunrise_az = 90 - np.degrees(np.arcsin(cos_az))
        winter_sunrise_az = 180 - summer_sunrise_az
        
        return {
            "summer_solstice_sunrise": summer_sunrise_az,
            "summer_solstice_sunset": 360 - summer_sunrise_az,
            "winter_solstice_sunrise": winter_sunrise_az,
            "winter_solstice_sunset": 360 - winter_sunrise_az,
            "obliquity": obliquity,
            "year": target_year,
        }
    
    def compute_historical_alignment(self, lat: float, lon: float,
                                     target_year: int) -> Dict:
        """
        Compute comprehensive celestial alignment data for a historical year.
        
        Args:
            lat, lon: Location coordinates
            target_year: Historical year
        
        Returns:
            Dict with alignment information
        """
        precession_offset = self.get_precession_offset(target_year)
        pole_star = self.get_historical_pole_star(target_year)
        visibility = self.compute_constellation_visibility(lat, lon, target_year)
        solstice = self.compute_solstice_azimuth(lat, target_year)
        
        # Sacred constellation average visibility
        sacred_consts = ["orion", "orion_belt", "pleiades", "draco", "cygnus"]
        sacred_visibility = np.mean([visibility.get(c, 0) for c in sacred_consts])
        
        return {
            "year": target_year,
            "precession_offset_deg": precession_offset,
            "pole_star": pole_star,
            "constellation_visibility": visibility,
            "sacred_visibility_avg": sacred_visibility,
            "solstice": solstice,
        }
    
    def analyze_orion_correlation(self, nodes: List[Dict],
                                  target_year: int = -10500) -> Dict:
        """
        Analyze how well a set of nodes matches the Orion belt pattern
        as it appeared in a specific historical epoch.
        
        The Orion Correlation Theory suggests the Giza pyramids
        mirror Orion's belt as it appeared around 10,500 BCE.
        
        Args:
            nodes: List of node dicts with 'lat', 'lon' keys
            target_year: Target year for Orion position
        
        Returns:
            Dict with correlation analysis
        """
        if len(nodes) < 3:
            return {"error": "Need at least 3 nodes for Orion correlation"}
        
        # Get Orion belt positions for target year
        precession_offset = self.get_precession_offset(target_year)
        
        belt_stars = SACRED_CONSTELLATIONS["orion_belt"]["stars"]
        adjusted_belt = []
        for star in belt_stars:
            adj_ra, adj_dec = self.adjust_coordinates_for_precession(
                star["ra"], star["dec"], target_year
            )
            adjusted_belt.append({
                "name": star["name"],
                "ra": adj_ra,
                "dec": adj_dec,
            })
        
        # Convert belt stars to comparable coordinates
        # Use dec as "latitude" and ra/15 as "longitude" equivalent
        belt_coords = [(s["dec"], s["ra"] / 15) for s in adjusted_belt]
        
        # Find best matching triplet of nodes
        best_match = None
        best_score = 0
        
        node_list = list(nodes) if isinstance(nodes, dict) else nodes
        
        for triplet in combinations(node_list, 3):
            if isinstance(triplet[0], dict):
                node_coords = [(n["lat"], n["lon"]) for n in triplet]
                node_names = [n.get("name", f"Node_{i}") for i, n in enumerate(triplet)]
            else:
                node_coords = [triplet[i] for i in range(3)]
                node_names = [f"Node_{i}" for i in range(3)]
            
            score = self._pattern_match_score(node_coords, belt_coords)
            
            if score > best_score:
                best_score = score
                best_match = {
                    "nodes": node_names if isinstance(triplet[0], dict) else triplet,
                    "coords": node_coords,
                }
        
        return {
            "target_year": target_year,
            "belt_positions": adjusted_belt,
            "best_match": best_match,
            "correlation_score": best_score,
            "interpretation": self._interpret_correlation(best_score),
        }
    
    def _pattern_match_score(self, coords1: List[Tuple], 
                            coords2: List[Tuple]) -> float:
        """
        Compute pattern similarity score using normalized distances.
        """
        if len(coords1) != len(coords2) or len(coords1) < 3:
            return 0.0
        
        def pairwise_distances(coords):
            distances = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    distances.append(np.sqrt(dx*dx + dy*dy))
            return sorted(distances)
        
        # Get pairwise distances
        d1 = pairwise_distances(coords1)
        d2 = pairwise_distances(coords2)
        
        # Normalize
        if max(d1) > 0:
            d1 = [d / max(d1) for d in d1]
        if max(d2) > 0:
            d2 = [d / max(d2) for d in d2]
        
        # Compare distance ratios
        diff = sum(abs(a - b) for a, b in zip(d1, d2)) / len(d1)
        
        return max(0, 1 - diff)
    
    def _interpret_correlation(self, score: float) -> str:
        """Interpret a correlation score."""
        if score > 0.9:
            return "Excellent match - strong Orion belt correlation"
        elif score > 0.7:
            return "Good match - notable Orion belt correlation"
        elif score > 0.5:
            return "Moderate match - some similarity to Orion belt"
        elif score > 0.3:
            return "Weak match - limited Orion belt correlation"
        else:
            return "No significant Orion belt correlation"
    
    def compute_T_score(self, lat: float, lon: float,
                       nodes: List[Dict] = None) -> float:
        """
        Compute Temporal score based on alignment quality across epochs.
        
        High T score = location shows significant alignments in
        multiple historical periods.
        
        Args:
            lat, lon: Location coordinates
            nodes: Optional list of known nodes for pattern analysis
        
        Returns:
            T score between 0 and 1
        """
        epoch_scores = []
        
        key_epochs = ["giza_pyramids", "gobekli_tepe", "younger_dryas_start", "orion_nadir"]
        
        for epoch_key in key_epochs:
            if epoch_key not in EPOCHS:
                continue
            
            year = EPOCHS[epoch_key]["year"]
            alignment = self.compute_historical_alignment(lat, lon, year)
            
            # Base score from sacred constellation visibility
            visibility_score = alignment["sacred_visibility_avg"]
            
            # Bonus for good solstice alignments
            solstice = alignment["solstice"]
            # Check if solstice azimuth is "significant" (near cardinal directions)
            summer_az = solstice["summer_solstice_sunrise"]
            cardinal_proximity = min(
                abs(summer_az - 45),   # NE
                abs(summer_az - 60),   # Typical alignment
                abs(summer_az - 90),   # E
            )
            solstice_bonus = max(0, 1 - cardinal_proximity / 30) * 0.2
            
            epoch_score = visibility_score + solstice_bonus
            epoch_scores.append(epoch_score)
        
        # T score is average across epochs
        if epoch_scores:
            T = np.mean(epoch_scores)
        else:
            T = 0.5
        
        return float(np.clip(T, 0, 1))
    
    def compute_T_breakdown(self, lat: float, lon: float) -> Dict:
        """
        Compute T score with detailed breakdown.
        """
        breakdown = {
            "T": 0.0,
            "epochs": {},
        }
        
        for epoch_key, epoch_data in EPOCHS.items():
            year = epoch_data["year"]
            alignment = self.compute_historical_alignment(lat, lon, year)
            
            breakdown["epochs"][epoch_key] = {
                "year": year,
                "name": epoch_data["name"],
                "visibility": alignment["sacred_visibility_avg"],
                "pole_star": alignment["pole_star"]["star"],
            }
        
        breakdown["T"] = self.compute_T_score(lat, lon)
        
        return breakdown


# Cached analyzer instance
_cached_analyzer = None


def get_analyzer() -> TemporalAnalyzer:
    """Get or create cached temporal analyzer."""
    global _cached_analyzer
    
    if _cached_analyzer is None:
        _cached_analyzer = TemporalAnalyzer()
    
    return _cached_analyzer


def compute_T_score(lat: float, lon: float, nodes: List[Dict] = None) -> float:
    """Compute temporal score for a location."""
    analyzer = get_analyzer()
    return analyzer.compute_T_score(lat, lon, nodes)


def get_epochs() -> Dict:
    """Get available historical epochs."""
    return EPOCHS


def get_epoch_alignment(lat: float, lon: float, epoch_key: str) -> Optional[Dict]:
    """Get alignment data for a specific epoch."""
    if epoch_key not in EPOCHS:
        return None
    
    analyzer = get_analyzer()
    year = EPOCHS[epoch_key]["year"]
    return analyzer.compute_historical_alignment(lat, lon, year)

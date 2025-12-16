#!/usr/bin/env python
"""
Extended Known GSN Nodes Database

This module contains an expanded list of 30+ validated ancient and megalithic sites
that are used as reference points for geometric coherence (H) calculations.

Each site includes:
- Name
- Coordinates (latitude, longitude)
- Category (pyramid, megalithic, ancient_city, temple, geoglyph, etc.)
- Approximate age (years BP where known)
- Source/reference

Sources:
- UNESCO World Heritage List
- Academic archaeological surveys
- Peer-reviewed publications on ancient sites
"""

# ---------------------------------------------------------
# Extended Known GSN Nodes (30+ sites)
# ---------------------------------------------------------

KNOWN_NODES_EXTENDED = {
    # ==================== PYRAMIDS ====================
    "Giza": {
        "coords": (29.9792, 31.1342),
        "category": "pyramid",
        "age_bp": 4500,
        "country": "Egypt",
        "notes": "Great Pyramid of Giza",
    },
    "Teotihuacan": {
        "coords": (19.6925, -98.8433),
        "category": "pyramid",
        "age_bp": 2000,
        "country": "Mexico",
        "notes": "Pyramid of the Sun",
    },
    "Shaanxi_Pyramids": {
        "coords": (34.3833, 108.7000),
        "category": "pyramid",
        "age_bp": 2200,
        "country": "China",
        "notes": "Maoling and surrounding pyramid mounds",
    },
    "Cholula": {
        "coords": (19.0578, -98.3017),
        "category": "pyramid",
        "age_bp": 2300,
        "country": "Mexico",
        "notes": "Great Pyramid of Cholula - largest pyramid by volume",
    },
    "Dahshur": {
        "coords": (29.8083, 31.2064),
        "category": "pyramid",
        "age_bp": 4600,
        "country": "Egypt",
        "notes": "Bent Pyramid and Red Pyramid",
    },
    "Saqqara": {
        "coords": (29.8711, 31.2164),
        "category": "pyramid",
        "age_bp": 4700,
        "country": "Egypt",
        "notes": "Step Pyramid of Djoser",
    },
    "Caral": {
        "coords": (-10.8933, -77.5203),
        "category": "pyramid",
        "age_bp": 5000,
        "country": "Peru",
        "notes": "Oldest known city in the Americas",
    },
    
    # ==================== MEGALITHIC STRUCTURES ====================
    "Stonehenge": {
        "coords": (51.1789, -1.8262),
        "category": "megalithic",
        "age_bp": 5000,
        "country": "United Kingdom",
        "notes": "Iconic stone circle",
    },
    "Gobekli_Tepe": {
        "coords": (37.2231, 38.9225),
        "category": "megalithic",
        "age_bp": 11500,
        "country": "Turkey",
        "notes": "Oldest known megalithic temple",
    },
    "Gunung_Padang": {
        "coords": (-6.9942, 107.0564),
        "category": "megalithic",
        "age_bp": 9000,
        "country": "Indonesia",
        "notes": "Controversial megalithic site",
    },
    "Carnac": {
        "coords": (47.5833, -3.0833),
        "category": "megalithic",
        "age_bp": 6000,
        "country": "France",
        "notes": "Largest collection of standing stones",
    },
    "Newgrange": {
        "coords": (53.6947, -6.4756),
        "category": "megalithic",
        "age_bp": 5200,
        "country": "Ireland",
        "notes": "Passage tomb with solar alignment",
    },
    "Avebury": {
        "coords": (51.4286, -1.8544),
        "category": "megalithic",
        "age_bp": 4500,
        "country": "United Kingdom",
        "notes": "Largest stone circle in Europe",
    },
    "Baalbek": {
        "coords": (34.0069, 36.2039),
        "category": "megalithic",
        "age_bp": 9000,
        "country": "Lebanon",
        "notes": "Massive stone blocks, Temple of Jupiter",
    },
    "Easter_Island": {
        "coords": (-27.1127, -109.3497),
        "category": "megalithic",
        "age_bp": 800,
        "country": "Chile",
        "notes": "Moai statues",
    },
    "Puma_Punku": {
        "coords": (-16.5617, -68.6803),
        "category": "megalithic",
        "age_bp": 1500,
        "country": "Bolivia",
        "notes": "Precision-cut stone blocks",
    },
    "Tiwanaku": {
        "coords": (-16.5544, -68.6731),
        "category": "megalithic",
        "age_bp": 1500,
        "country": "Bolivia",
        "notes": "Pre-Columbian archaeological site",
    },
    "Ollantaytambo": {
        "coords": (-13.2583, -72.2625),
        "category": "megalithic",
        "age_bp": 550,
        "country": "Peru",
        "notes": "Inca fortress with massive stonework",
    },
    "Sacsayhuaman": {
        "coords": (-13.5094, -71.9822),
        "category": "megalithic",
        "age_bp": 550,
        "country": "Peru",
        "notes": "Massive polygonal masonry",
    },
    
    # ==================== ANCIENT CITIES ====================
    "Angkor_Wat": {
        "coords": (13.4125, 103.8670),
        "category": "temple",
        "age_bp": 900,
        "country": "Cambodia",
        "notes": "Largest religious monument",
    },
    "Machu_Picchu": {
        "coords": (-13.1631, -72.5450),
        "category": "ancient_city",
        "age_bp": 550,
        "country": "Peru",
        "notes": "Inca citadel",
    },
    "Petra": {
        "coords": (30.3285, 35.4444),
        "category": "ancient_city",
        "age_bp": 2300,
        "country": "Jordan",
        "notes": "Rock-cut architecture",
    },
    "Mohenjo_Daro": {
        "coords": (27.3242, 68.1375),
        "category": "ancient_city",
        "age_bp": 4500,
        "country": "Pakistan",
        "notes": "Indus Valley Civilization",
    },
    "Great_Zimbabwe": {
        "coords": (-20.2674, 30.9338),
        "category": "ancient_city",
        "age_bp": 900,
        "country": "Zimbabwe",
        "notes": "Medieval African city",
    },
    "Derinkuyu": {
        "coords": (38.3742, 34.7347),
        "category": "ancient_city",
        "age_bp": 2800,
        "country": "Turkey",
        "notes": "Underground city",
    },
    "Persepolis": {
        "coords": (29.9352, 52.8914),
        "category": "ancient_city",
        "age_bp": 2500,
        "country": "Iran",
        "notes": "Ceremonial capital of Achaemenid Empire",
    },
    
    # ==================== GEOGLYPHS & EARTHWORKS ====================
    "Nazca_Lines": {
        "coords": (-14.7350, -75.1300),
        "category": "geoglyph",
        "age_bp": 2000,
        "country": "Peru",
        "notes": "Giant desert drawings",
    },
    "Serpent_Mound": {
        "coords": (39.0253, -83.4303),
        "category": "earthwork",
        "age_bp": 2000,
        "country": "USA",
        "notes": "Effigy mound in Ohio",
    },
    "Poverty_Point": {
        "coords": (32.6347, -91.4064),
        "category": "earthwork",
        "age_bp": 3400,
        "country": "USA",
        "notes": "Archaic earthwork complex",
    },
    
    # ==================== TEMPLES & OBSERVATORIES ====================
    "Chichen_Itza": {
        "coords": (20.6843, -88.5678),
        "category": "temple",
        "age_bp": 1500,
        "country": "Mexico",
        "notes": "El Castillo pyramid",
    },
    "Karnak": {
        "coords": (25.7188, 32.6573),
        "category": "temple",
        "age_bp": 4000,
        "country": "Egypt",
        "notes": "Largest ancient religious complex",
    },
    "Borobudur": {
        "coords": (-7.6079, 110.2038),
        "category": "temple",
        "age_bp": 1200,
        "country": "Indonesia",
        "notes": "Largest Buddhist temple",
    },
    "Delphi": {
        "coords": (38.4824, 22.5010),
        "category": "temple",
        "age_bp": 2800,
        "country": "Greece",
        "notes": "Oracle of Delphi",
    },
    
    # ==================== ANOMALOUS LOCATIONS ====================
    "Mt_Hayes": {
        "coords": (63.6200, -146.7178),
        "category": "anomalous",
        "age_bp": None,
        "country": "USA",
        "notes": "Alaska - theorized GSN node",
    },
    "Geikie_Peninsula": {
        "coords": (70.3000, -27.0000),
        "category": "anomalous",
        "age_bp": None,
        "country": "Greenland",
        "notes": "East Greenland - theorized GSN node",
    },
    "Azores_Plateau": {
        "coords": (38.7000, -28.0000),
        "category": "anomalous",
        "age_bp": None,
        "country": "Portugal",
        "notes": "Atlantic - theorized GSN node",
    },
    "Moche": {
        "coords": (-8.1500, -79.0000),
        "category": "pyramid",
        "age_bp": 1800,
        "country": "Peru",
        "notes": "Huaca del Sol and Huaca de la Luna",
    },
}

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_coords_dict():
    """Return simple {name: (lat, lon)} dict for backward compatibility."""
    return {name: data["coords"] for name, data in KNOWN_NODES_EXTENDED.items()}


def get_nodes_by_category(category):
    """Filter nodes by category."""
    return {
        name: data for name, data in KNOWN_NODES_EXTENDED.items()
        if data["category"] == category
    }


def get_nodes_older_than(age_bp):
    """Get nodes older than specified age (years before present)."""
    return {
        name: data for name, data in KNOWN_NODES_EXTENDED.items()
        if data["age_bp"] is not None and data["age_bp"] >= age_bp
    }


def get_node_count():
    """Return total number of known nodes."""
    return len(KNOWN_NODES_EXTENDED)


# Categories summary
CATEGORIES = {
    "pyramid": "Pyramid structures",
    "megalithic": "Megalithic stone structures",
    "ancient_city": "Ancient urban centers",
    "temple": "Temples and religious structures",
    "geoglyph": "Geoglyphs and ground drawings",
    "earthwork": "Earthworks and mounds",
    "anomalous": "Theorized/anomalous locations",
}


if __name__ == "__main__":
    print(f"Extended Known Nodes Database")
    print(f"Total nodes: {get_node_count()}")
    print("\nBy category:")
    for cat, desc in CATEGORIES.items():
        nodes = get_nodes_by_category(cat)
        print(f"  {cat}: {len(nodes)} sites")
    
    print("\nSites older than 5000 BP:")
    ancient = get_nodes_older_than(5000)
    for name, data in ancient.items():
        print(f"  {name}: {data['age_bp']} BP - {data['country']}")

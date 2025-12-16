# GSN Node Predictor

**Global Stabilization Network (GSN) Node Prediction Tool**

A comprehensive Python toolkit for identifying potential Global Stabilization Network nodes using geophysical, geometric, astronomical, and archaeological data analysis.

## Author

**H** - Creator and maintainer

## Features

### 🌍 Multi-Factor Analysis
- **G-Score (Geophysical Suitability)**: Gravity anomalies, crustal thickness, plate boundary proximity
- **H-Score (Geometric Coherence)**: Penrose angles, sacred geometry, great circle alignments, symmetry patterns
- **A-Score (Astronomical)**: Celestial alignments, constellation visibility
- **N-Score (Network)**: Existing node connectivity analysis
- **T-Score (Temporal)**: Time-based pattern analysis

### 🤖 ML-Enhanced Scoring
- Neural network trained on known GSN node locations
- Replaces hand-crafted `F = α*G + β*H` formula with learned optimal weighting
- Data augmentation and hard negative mining for robust training

### 🗺️ Interactive Web Application
- Streamlit-based dashboard with interactive maps
- Real-time heatmap visualization
- Detailed score breakdowns with explanations
- H-method comparison tools
- Constellation visibility overlay

### 📐 Advanced Geometry
- Extended sacred angles (Penrose, Fibonacci, Platonic solids)
- Great circle alignment detection
- Symmetric pattern recognition (Golden triangles, equilateral configurations)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/desktopmenace/GlobalStabilizationNetwork.git
cd GlobalStabilizationNetwork

# Install Git LFS (required for data files)
git lfs install
git lfs pull

# Create virtual environment
python3 -m venv gsn_env
source gsn_env/bin/activate  # On Windows: gsn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run gsn_web_app.py
```

---

## Data Sources

The repository includes most data files via Git LFS. One large file must be downloaded separately.

### ✅ Included via Git LFS (~1.6 GB)

These files are automatically downloaded with `git lfs pull`:

| Dataset | Size | Source |
|---------|------|--------|
| WGM2012 Gravity Grids | 5 × 223 MB | [BGI World Gravity Map](https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/) |
| gravity_model.nc | 223 MB | Preprocessed gravity anomaly grid |
| crust_model.nc | 265 KB | Crustal thickness data |
| de421.bsp | 16 MB | [JPL Ephemeris](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) |
| plate-boundaries.kmz | 321 KB | Tectonic plate boundaries |
| earthquakes.json | 65 MB | USGS earthquake cache |
| GHFDB Heat Flow | 31 MB | [IHFC Database](https://ihfc-iugg.org/products/global-heat-flow-database/data) |
| Volcano List | 1.1 MB | [GVP Smithsonian](https://volcano.si.edu/volcanolist_holocene.cfm) |
| Crustal Age Grids | ~50 MB | [NOAA Ocean Age](https://www.ngdc.noaa.gov/mgg/ocean_age/) |
| ML Model Checkpoint | 71 KB | Pre-trained scorer |

### ⚠️ Manual Download Required

**EMAG2 Magnetic Anomaly Grid (4 GB)** - Exceeds GitHub LFS 2GB limit

1. Visit: https://www.ngdc.noaa.gov/geomag/emag2.html
2. Download: `EMAG2_V3_20170530.csv` (or latest version)
3. Place in repository root folder

---

## Data Source Links

| Data Type | Source | Download Link |
|-----------|--------|---------------|
| **Gravity** | World Gravity Map (BGI) | https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/ |
| **Crustal Thickness** | CRUST1.0 (UCSD) | https://igppweb.ucsd.edu/~gabi/crust1.html |
| **Plate Boundaries** | USGS | https://www.usgs.gov/programs/earthquake-hazards/faults |
| **Magnetic Anomaly** | NOAA EMAG2 | https://www.ngdc.noaa.gov/geomag/emag2.html |
| **Heat Flow** | IHFC Database | https://ihfc-iugg.org/products/global-heat-flow-database/data |
| **Volcanoes** | GVP Smithsonian | https://volcano.si.edu/volcanolist_holocene.cfm |
| **Earthquakes** | USGS Catalog | https://earthquake.usgs.gov/earthquakes/search/ |
| **Ocean Age** | NOAA NCEI | https://www.ngdc.noaa.gov/mgg/ocean_age/ |
| **World Stress Map** | GFZ Potsdam | https://www.world-stress-map.org/download |
| **Ephemeris** | JPL NAIF | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ |
| **Ancient Places** | Pleiades Project | https://pleiades.stoa.org/downloads |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Git with LFS support
- ~2 GB disk space (with LFS data)
- ~6 GB disk space (with EMAG2)

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/desktopmenace/GlobalStabilizationNetwork.git
cd GlobalStabilizationNetwork

# 2. Install Git LFS and pull data files
git lfs install
git lfs pull

# 3. Create and activate virtual environment
python3 -m venv gsn_env
source gsn_env/bin/activate  # Linux/macOS
# gsn_env\Scripts\activate   # Windows

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) Install PyTorch for ML features
pip install torch

# 6. (Optional) Download EMAG2 data
# Visit https://www.ngdc.noaa.gov/geomag/emag2.html
# Download and place EMAG2_V3_20170530.csv in root folder

# 7. Verify data status
python download_data.py --check
```

---

## Usage

### Web Application (Recommended)

```bash
streamlit run gsn_web_app.py
```

Access at:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

### Command Line Interface

```bash
python gsn_node_predictor.py
```

Menu options:
1. **Evaluate single location** - Calculate G, H, F scores for coordinates
2. **Scan for H peaks** - Find geometric coherence hotspots
3. **Scan for F candidates** - Combined geophysical + geometric search
4. **Generate Mollweide map** - Global projection visualization
5. **Validate known nodes** - Test against known GSN locations
6. **Full-resolution scan** - High-detail native grid analysis
7. **ML-based scan** - Neural network enhanced prediction

### Data Management

```bash
# Check data availability
python download_data.py --check

# List all datasets
python download_data.py --list

# Show download instructions for specific dataset
python download_data.py --info wsm
```

---

## Project Structure

```
GlobalStabilizationNetwork/
├── gsn_node_predictor.py    # Main CLI tool and core algorithms
├── gsn_web_app.py           # Streamlit web interface
├── gsn_ml_grid_scorer.py    # PyTorch neural network model
├── gsn_astronomy.py         # Astronomical calculations (Skyfield)
├── gsn_archaeology.py       # Archaeological site analysis
├── gsn_geometry.py          # Sacred geometry & pattern detection
├── gsn_network.py           # Network connectivity analysis
├── gsn_seismic.py           # USGS earthquake integration
├── gsn_volcanic.py          # Volcanic data analysis
├── gsn_heatflow.py          # Heat flow calculations
├── gsn_temporal.py          # Temporal pattern analysis
├── gsn_data_sources.py      # Data loading utilities
├── gsn_validation.py        # Validation framework
├── gsn_uncertainty.py       # Uncertainty quantification
├── known_nodes_extended.py  # Known GSN node database
├── download_data.py         # Data download helper
├── requirements.txt         # Python dependencies
│
├── # Data Files (Git LFS)
├── gravity_model.nc         # Gravity anomaly grid
├── crust_model.nc           # Crustal thickness grid
├── de421.bsp                # JPL ephemeris
├── gsn_data/                # Plate boundaries, gravity grids
├── cache/                   # Earthquake & computed data cache
├── GHFDB-R2024/             # Heat flow database
└── gsn_ml_grid_scorer.pth   # Trained ML model
```

---

## Requirements

### Core Dependencies
- numpy, scipy, pandas
- netCDF4, xarray
- matplotlib, cartopy

### Web Application
- streamlit
- folium, streamlit-folium
- plotly, pydeck

### ML Features (Optional)
- torch (PyTorch 2.0+)
- scikit-learn

### Astronomy (Optional)
- skyfield

See `requirements.txt` for complete list with versions.

---

## License

This project is provided as-is for research and educational purposes.

---

## Acknowledgments

- **Geophysical Data**: BGI, NOAA NCEI, USGS, IHFC
- **Astronomical Data**: JPL NAIF, Skyfield
- **Archaeological Data**: Pleiades Project, GVP Smithsonian
- **Sacred Geometry**: Mathematical frameworks and research community
- **GSN Research**: The Global Stabilization Network research community

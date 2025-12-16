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

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GSN.git
cd GSN

# Create virtual environment
python3 -m venv gsn_env
source gsn_env/bin/activate  # On Windows: gsn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

The tool requires several data files that are not included in the repository due to size. Download and place them in the project root:

1. **Gravity Model** (`gravity_model.nc`) - Global gravity anomaly data
2. **Crust Model** (`crust_model.nc`) - Crustal thickness data  
3. **Ephemeris** (`de421.bsp`) - JPL ephemeris for astronomical calculations
4. **Plate Boundaries** (`gsn_data/plate-boundaries.kmz`) - Tectonic plate boundary data

## Usage

### Command Line Interface

```bash
python gsn_node_predictor.py
```

Select from available modes:
1. Evaluate single location
2. Scan for geometric (H) peaks
3. Scan for F (G+H) candidates
4. Generate Mollweide map
5. Validate known nodes
6. Full-resolution scan
7. **ML-based grid scan** (recommended)

### Web Application

```bash
streamlit run gsn_web_app.py
```

Access at `http://localhost:8501` (or Network URL for LAN access)

## Project Structure

```
GSN/
├── gsn_node_predictor.py    # Main CLI tool
├── gsn_web_app.py           # Streamlit web interface
├── gsn_ml_grid_scorer.py    # ML scoring model
├── gsn_astronomy.py         # Astronomical calculations
├── gsn_archaeology.py       # Archaeological site analysis
├── gsn_geometry.py          # Advanced geometric analysis
├── gsn_network.py           # Network connectivity
├── gsn_seismic.py           # Seismic data integration
├── gsn_temporal.py          # Temporal pattern analysis
├── gsn_volcanic.py          # Volcanic data analysis
├── known_nodes_extended.py  # Known GSN node database
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (for ML features)
- NumPy, SciPy, Pandas
- Streamlit, Folium, Plotly
- Skyfield (astronomy)
- NetCDF4 (data loading)

See `requirements.txt` for full dependency list.

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

- Geophysical data from various open scientific sources
- Sacred geometry research and mathematical frameworks
- The GSN research community


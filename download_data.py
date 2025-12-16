#!/usr/bin/env python
"""
GSN Data Download Script

Downloads and sets up external data sources for the GSN prediction system.

Usage:
    python download_data.py --all           # Download all available datasets
    python download_data.py --heatflow      # Download heat flow data only
    python download_data.py --list          # List available datasets

Note: Some datasets require manual download due to access restrictions.
"""

import os
import sys
import argparse
from typing import Dict

# Data directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Dataset configurations
DATASETS = {
    "heatflow": {
        "name": "Global Heat Flow Database",
        "source": "International Heat Flow Commission",
        "url": "https://ihfc-iugg.org/products/global-heat-flow-database/data",
        "filename": "GHFDB-R2024/IHFC_2024_GHFDB.xlsx",
        "auto_download": False,
        "instructions": """
1. Visit https://ihfc-iugg.org/products/global-heat-flow-database/data
2. Download the latest GHFDB release (xlsx format)
3. Create directory 'GHFDB-R2024' in the GSN folder
4. Place the downloaded file there
""",
    },
    "volcanoes": {
        "name": "Smithsonian Holocene Volcanoes",
        "source": "Global Volcanism Program",
        "url": "https://volcano.si.edu/volcanolist_holocene.cfm",
        "filename": "GVP_Volcano_List_Holocene_*.xls",
        "auto_download": False,
        "instructions": """
1. Visit https://volcano.si.edu/volcanolist_holocene.cfm
2. Click 'Download Excel File'
3. Save to GSN folder (keep default filename)
""",
    },
    "ocean_age": {
        "name": "Ocean Floor Age Grid",
        "source": "NOAA NCEI / Müller et al.",
        "url": "https://www.ngdc.noaa.gov/mgg/ocean_age/",
        "filename": "crustal age/age.3.6.tif",
        "auto_download": False,
        "instructions": """
1. Visit https://www.ngdc.noaa.gov/mgg/ocean_age/
2. Download the age of oceanic lithosphere grid
3. Create directory 'crustal age' in the GSN folder
4. Extract/place the .tif file there
""",
    },
    "wsm": {
        "name": "World Stress Map",
        "source": "WSM Project",
        "url": "https://www.world-stress-map.org/download",
        "filename": "wsm_data.csv",
        "auto_download": False,
        "instructions": """
1. Visit https://www.world-stress-map.org/download
2. Register if required
3. Download the WSM database (CSV format)
4. Save as 'wsm_data.csv' in the GSN folder
""",
    },
    "gravity": {
        "name": "WGM Gravity Grids",
        "source": "World Gravity Map / BGI",
        "url": "https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/",
        "filename": "gsn_data/Global_grids___World_Gravity_Map/WGM2012_*.grd",
        "auto_download": False,
        "instructions": """
1. Visit BGI website
2. Download WGM2012 grids (Bouguer, Free-air, Isostatic, etc.)
3. Place in gsn_data/Global_grids___World_Gravity_Map/
""",
    },
    "magnetic": {
        "name": "EMAG2 Magnetic Anomaly",
        "source": "NOAA NCEI",
        "url": "https://www.ngdc.noaa.gov/geomag/emag2.html",
        "filename": "EMAG2_V3_20170530.csv",
        "auto_download": False,
        "instructions": """
1. Visit https://www.ngdc.noaa.gov/geomag/emag2.html
2. Download EMAG2v3 data (CSV or grid format)
3. Place in GSN folder
""",
    },
    "pleiades": {
        "name": "Pleiades Ancient Places",
        "source": "Pleiades Project",
        "url": "https://pleiades.stoa.org/downloads",
        "filename": "pleiades-places-*.csv",
        "auto_download": True,  # Can be auto-downloaded
        "download_url": "https://atlantides.org/downloads/pleiades/dumps/pleiades-places-latest.csv.gz",
        "instructions": """
1. Visit https://pleiades.stoa.org/downloads
2. Download 'Places' dataset (CSV)
3. Extract and place in GSN folder
""",
    },
}


def check_dataset(name: str) -> bool:
    """Check if a dataset is available."""
    config = DATASETS.get(name)
    if not config:
        return False
    
    filepath = os.path.join(DATA_DIR, config["filename"])
    
    # Handle wildcards
    if "*" in filepath:
        import glob
        matches = glob.glob(filepath)
        return len(matches) > 0
    
    return os.path.exists(filepath)


def list_datasets():
    """List all datasets and their status."""
    print("\nGSN Data Sources Status")
    print("=" * 60)
    
    for name, config in DATASETS.items():
        available = check_dataset(name)
        status = "✓ Available" if available else "✗ Not found"
        
        print(f"\n{config['name']}")
        print(f"  Source: {config['source']}")
        print(f"  Status: {status}")
        print(f"  File: {config['filename']}")


def show_instructions(name: str):
    """Show download instructions for a dataset."""
    config = DATASETS.get(name)
    if not config:
        print(f"Unknown dataset: {name}")
        return
    
    print(f"\n{config['name']}")
    print("-" * 40)
    print(f"Source: {config['source']}")
    print(f"URL: {config['url']}")
    print(f"\nDownload Instructions:")
    print(config["instructions"])


def download_dataset(name: str) -> bool:
    """Attempt to download a dataset."""
    config = DATASETS.get(name)
    if not config:
        print(f"Unknown dataset: {name}")
        return False
    
    if not config.get("auto_download", False):
        print(f"\n{config['name']} requires manual download:")
        print(config["instructions"])
        return False
    
    try:
        import requests
        import gzip
    except ImportError:
        print("requests library required for downloads")
        return False
    
    download_url = config.get("download_url")
    if not download_url:
        print(f"No download URL configured for {name}")
        return False
    
    print(f"\nDownloading {config['name']}...")
    print(f"URL: {download_url}")
    
    try:
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Determine output filename
        filename = config["filename"]
        if "*" in filename:
            # Use a default name
            filename = filename.replace("*", "latest")
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # Handle gzipped files
        if download_url.endswith(".gz"):
            import gzip
            import shutil
            
            gz_path = filepath + ".gz"
            with open(gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with gzip.open(gz_path, "rb") as f_in:
                with open(filepath, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
        else:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"Downloaded to: {filepath}")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def setup_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")


def main():
    parser = argparse.ArgumentParser(description="GSN Data Download Script")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--check", action="store_true", help="Check data availability")
    parser.add_argument("--dataset", type=str, help="Download specific dataset")
    parser.add_argument("--info", type=str, help="Show info for specific dataset")
    
    args = parser.parse_args()
    
    if args.list or args.check:
        list_datasets()
        return
    
    if args.info:
        show_instructions(args.info)
        return
    
    if args.dataset:
        download_dataset(args.dataset)
        return
    
    if args.all:
        setup_cache_dir()
        print("\nAttempting to download all datasets...")
        
        for name in DATASETS:
            if check_dataset(name):
                print(f"\n✓ {DATASETS[name]['name']} already available")
            else:
                download_dataset(name)
        
        print("\n" + "=" * 60)
        print("Download complete. Run --check to verify status.")
        return
    
    # Default: show help
    parser.print_help()
    print("\nExample usage:")
    print("  python download_data.py --list      # Show all datasets")
    print("  python download_data.py --check     # Check availability")
    print("  python download_data.py --info wsm  # Show WSM download instructions")


if __name__ == "__main__":
    main()

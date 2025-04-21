# LST Downscaling Project
![images](images/images.png)

This project implements a Land Surface Temperature (LST) downscaling pipeline using Landsat and Sentinel-2 satellite imagery. The workflow processes geospatial data, calculates spectral indices, trains machine learning models to downscale LST, and visualizes the results.

## Project Structure

The project consists of three main Python scripts:

1. **preprocessing.py**: Handles data preprocessing, including clipping rasters with shapefiles, calibrating satellite imagery, and calculating spectral indices (NDVI, MNDWI, NDBI, ASVI).
2. **lst_downscaling.py**: Implements LST downscaling using Linear Regression and Random Forest models, trained on Landsat data and applied to Sentinel-2 data.
3. **visualization.py**: Generates visualizations of the spectral indices and downscaled LST results using matplotlib.

## Prerequisites

### Data
- **Landsat Data**: Landsat 9 Level-2 Science Product, containing bands B3, B4, B5, B6, and ST_B10.
- **Sentinel-2 Data**: Sentinel-2 Level-2A product with bands B03, B04, B08, and B11.
- **Shapefile**: A shapefile defining the area of interest for clipping rasters.

Place all data in a `data/` directory relative to the scripts.

## Directory Structure

```plaintext
LST_downscaling/
├── data/
│   ├── Landsat data
│   ├── Sentinel-2 data
│   ├── SHP
│   ├── processed/  # Output directory for processed rasters
│   └── models/  # Output directory for trained models
├── preprocessing.py
├── lst_downscaling.py
├── visualization.py
└── README.md
```

## Usage

1. **Preprocessing**:
   Run `preprocessing.py` to clip rasters, calibrate imagery, and calculate spectral indices:
   ```bash
   python preprocessing.py
   ```
   This generates processed rasters (`landsat_lst.tif`, `landsat_ndvi.tif`, `sentinel_ndvi.tif`, etc.) in the `data/processed/` directory.

2. **LST Downscaling**:
   Run `lst_downscaling.py` to train machine learning models and predict high-resolution LST using Sentinel-2 data:
   ```bash
   python lst_downscaling.py
   ```
   This saves trained models in `data/models/` and downscaled LST rasters (`sentinel_lst_lr.tif`, `sentinel_lst_rf.tif`) in `data/processed/`.

3. **Visualization**:
   Run `visualization.py` to generate a spatial distribution map of indices and LST results:
   ```bash
   python visualization.py
   ```
   This saves the visualization as `results_spatial.png` in the `data/` directory.

## Workflow

1. **Preprocessing** (`preprocessing.py`):
   - Reads a shapefile to define the study area.
   - Clips Landsat and Sentinel-2 rasters to the shapefile boundaries.
   - Calibrates Landsat bands (reflectance for B3-B6, temperature for ST_B10) and Sentinel-2 bands (reflectance).
   - Resamples Sentinel-2 SWIR band (B11) to 10m resolution if needed.
   - Calculates NDVI, MNDWI, and NDBI for both datasets.
   - Saves processed rasters as GeoTIFFs.

2. **LST Downscaling** (`lst_downscaling.py`):
   - Loads processed rasters (LST and indices).
   - Filters data to remove NaN values and outliers (1st–99th percentiles).
   - Trains Linear Regression and Random Forest models using Landsat indices as features and LST as the target.
   - Applies trained models to Sentinel-2 indices to predict high-resolution LST.
   - Saves downscaled LST rasters and model files.

3. **Visualization** (`visualization.py`):
   - Loads all processed rasters.
   - Creates a 3x3 grid of plots showing NDVI, MNDWI, NDBI, and LST for Landsat, Sentinel-2, and downscaled results.
   - Uses custom colormaps for each index and LST.

## Output

- **Processed Rasters** (`data/processed/`):
  - `landsat_lst.tif`: Landsat LST
  - `landsat_ndvi.tif`, `landsat_mndwi.tif`, `landsat_ndbi.tif`: Landsat spectral indices
  - `sentinel_ndvi.tif`, `sentinel_mndwi.tif`, `sentinel_ndbi.tif`: Sentinel-2 spectral indices
  - `sentinel_lst_lr.tif`, `sentinel_lst_rf.tif`: Downscaled Sentinel-2 LST (Linear Regression and Random Forest)

- **Models** (`data/models/`)

- **Visualization** (`data/`):
  - `results_spatial.png`: Spatial distribution map of indices and LST

## Notes

- Ensure the input data paths in the scripts match your directory structure.
- The scripts assume Landsat and Sentinel-2 data are from the same date or close temporal proximity for consistency.
- Random Forest model parameters (e.g., `n_estimators=50`, `max_depth=15`) can be tuned for better performance.
- The visualization uses custom colormaps tailored for NDVI, MNDWI, NDBI, and LST; adjust as needed.

## Contact

If you have any questions, please contact geo.wqlin@gmail.com or qlwu@itpcas.ac.cn

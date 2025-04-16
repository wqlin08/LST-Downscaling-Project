import os
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

class ResultVisualization:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        
    def _load_raster(self, filename):
 
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"no files: {filepath}")
        
        with rasterio.open(filepath) as src:
            data = src.read(1)
            data = data.astype(np.float32)
            data[data == -9999] = np.nan
            return data, src.transform, src.crs
            
            
    def plot_indices(self):


        landsat_ndvi, l_transform, l_crs = self._load_raster('landsat_ndvi.tif')
        landsat_mndwi, _, _ = self._load_raster('landsat_mndwi.tif')
        landsat_ndbi, _, _ = self._load_raster('landsat_ndbi.tif')
        landsat_lst, _, _ = self._load_raster('landsat_lst.tif')
        
        sentinel_ndvi, s_transform, _ = self._load_raster('sentinel_ndvi.tif')
        sentinel_mndwi, _, _ = self._load_raster('sentinel_mndwi.tif')
        sentinel_ndbi, _, _ = self._load_raster('sentinel_ndbi.tif')

        sentinel_lst_lr, _, _ = self._load_raster('sentinel_lst_lr.tif')
        sentinel_lst_rf, _, _ = self._load_raster('sentinel_lst_rf.tif')
        
        # color
        ndvi_colors = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32']
        ndvi_cmap = LinearSegmentedColormap.from_list('ndvi', ndvi_colors)
        
        water_colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
        water_cmap = LinearSegmentedColormap.from_list('water', water_colors)
        
        built_colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        built_cmap = LinearSegmentedColormap.from_list('built', built_colors)
        
        lst_colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027']
        lst_cmap = LinearSegmentedColormap.from_list('lst', lst_colors)
        
  
        fig = plt.figure(figsize=(20, 12))
        
        plt.subplot(331)
        plt.imshow(landsat_ndvi, cmap=ndvi_cmap)
        plt.colorbar(label='NDVI')
        plt.title('Landsat NDVI')
        
        plt.subplot(332)
        plt.imshow(landsat_mndwi, cmap=water_cmap)
        plt.colorbar(label='MNDWI')
        plt.title('Landsat MNDWI')
        
        plt.subplot(333)
        plt.imshow(landsat_ndbi, cmap=built_cmap)
        plt.colorbar(label='NDBI')
        plt.title('Landsat NDBI')
        
        plt.subplot(334)
        plt.imshow(sentinel_ndvi, cmap=ndvi_cmap)
        plt.colorbar(label='NDVI')
        plt.title('Sentinel-2 NDVI')
        
        plt.subplot(335)
        plt.imshow(sentinel_mndwi, cmap=water_cmap)
        plt.colorbar(label='MNDWI')
        plt.title('Sentinel-2 MNDWI')
        
        plt.subplot(336)
        plt.imshow(sentinel_ndbi, cmap=built_cmap)
        plt.colorbar(label='NDBI')
        plt.title('Sentinel-2 NDBI')
        
        plt.subplot(337)
        plt.imshow(landsat_lst, cmap=lst_cmap)
        plt.colorbar(label='Temperature (K)')
        plt.title('Landsat LST')
        
        plt.subplot(338)
        plt.imshow(sentinel_lst_lr, cmap=lst_cmap)
        plt.colorbar(label='Temperature (K)')
        plt.title('Downscaled LST (Linear Regression)')
        
        plt.subplot(339)
        plt.imshow(sentinel_lst_rf, cmap=lst_cmap)
        plt.colorbar(label='Temperature (K)')
        plt.title('Downscaled LST (Random Forest)')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'results_spatial.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    visualizer = ResultVisualization("data")
    
    print("Draw a spatial distribution map...")
    visualizer.plot_indices()
    
    print("Visualization is complete! The results are saved in: data/results_spatial.png")
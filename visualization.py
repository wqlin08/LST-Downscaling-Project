import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

class ResultVisualization:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        
    def _load_raster(self, filename):
        """Load raster data"""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with rasterio.open(filepath) as src:
            data = src.read(1)
            data = data.astype(np.float32)
            data[data == -9999] = np.nan
            return data, src.transform, src.crs
            
    def _resample_to_reference(self, data, src_transform, reference_path):
        """Resample data to reference resolution"""
        with rasterio.open(reference_path) as ref:
            # Calculate scale factor
            scale_factor = ref.transform[0] / src_transform[0]
            
            # Calculate target size
            height = int(data.shape[0] / scale_factor)
            width = int(data.shape[1] / scale_factor)
            
            # Create output array
            output = np.zeros((height, width), dtype=np.float32)
            
            # Perform resampling
            reproject(
                source=data,
                destination=output,
                src_transform=src_transform,
                dst_transform=ref.transform,
                src_crs=ref.crs,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear
            )
            
            return output
            
    def plot_indices(self):
        """Plot spatial distribution of all indices"""
        # Load data
        landsat_ndvi, l_transform, l_crs = self._load_raster('landsat_ndvi.tif')
        landsat_mndwi, _, _ = self._load_raster('landsat_mndwi.tif')
        landsat_ndbi, _, _ = self._load_raster('landsat_ndbi.tif')
        landsat_savi, _, _ = self._load_raster('landsat_savi.tif')
        landsat_lst, _, _ = self._load_raster('landsat_lst.tif')
        
        sentinel_ndvi, s_transform, _ = self._load_raster('sentinel_ndvi.tif')
        sentinel_mndwi, _, _ = self._load_raster('sentinel_mndwi.tif')
        sentinel_ndbi, _, _ = self._load_raster('sentinel_ndbi.tif')
        sentinel_savi, _, _ = self._load_raster('sentinel_savi.tif')
        
        # Load predicted LST
        sentinel_lst_lr, _, _ = self._load_raster('sentinel_lst_lr.tif')
        sentinel_lst_rf, _, _ = self._load_raster('sentinel_lst_rf.tif')
        sentinel_lst_xgb, _, _ = self._load_raster('sentinel_lst_xgb.tif')
        
        # Create custom colormaps
        ndvi_colors = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32']
        ndvi_cmap = LinearSegmentedColormap.from_list('ndvi', ndvi_colors)
        
        water_colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
        water_cmap = LinearSegmentedColormap.from_list('water', water_colors)
        
        built_colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        built_cmap = LinearSegmentedColormap.from_list('built', built_colors)
        
        lst_colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027']
        lst_cmap = LinearSegmentedColormap.from_list('lst', lst_colors)
        
        # Create figure
        fig = plt.figure(figsize=(24, 20))  
        gs = plt.GridSpec(4, 4, figure=fig, 
                         hspace=0.15,  
                         wspace=0.4,   
                         left=0.1,  
                         right=0.95,   
                         top=0.95,    
                         bottom=0.05)  
        
     
        plt.subplot(gs[0, 0])
        im1 = plt.imshow(landsat_ndvi, cmap=ndvi_cmap)
        plt.colorbar(im1, label='NDVI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Landsat NDVI', pad=5)
        
        plt.subplot(gs[0, 1])
        im2 = plt.imshow(landsat_mndwi, cmap=water_cmap)
        plt.colorbar(im2, label='MNDWI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Landsat MNDWI', pad=5)
        
        plt.subplot(gs[0, 2])
        im3 = plt.imshow(landsat_ndbi, cmap=built_cmap)
        plt.colorbar(im3, label='NDBI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Landsat NDBI', pad=5)
        
        plt.subplot(gs[0, 3])
        im4 = plt.imshow(landsat_savi, cmap=ndvi_cmap)
        plt.colorbar(im4, label='SAVI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Landsat SAVI', pad=5)
        
      
        plt.subplot(gs[1, 0])
        im5 = plt.imshow(sentinel_ndvi, cmap=ndvi_cmap)
        plt.colorbar(im5, label='NDVI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Sentinel-2 NDVI', pad=5)
        
        plt.subplot(gs[1, 1])
        im6 = plt.imshow(sentinel_mndwi, cmap=water_cmap)
        plt.colorbar(im6, label='MNDWI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Sentinel-2 MNDWI', pad=5)
        
        plt.subplot(gs[1, 2])
        im7 = plt.imshow(sentinel_ndbi, cmap=built_cmap)
        plt.colorbar(im7, label='NDBI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Sentinel-2 NDBI', pad=5)
        
        plt.subplot(gs[1, 3])
        im8 = plt.imshow(sentinel_savi, cmap=ndvi_cmap)
        plt.colorbar(im8, label='SAVI', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Sentinel-2 SAVI', pad=5)
        
      
        plt.subplot(gs[2, 0])
        im9 = plt.imshow(landsat_lst, cmap=lst_cmap)
        plt.colorbar(im9, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Original Landsat LST', pad=5)
        
        plt.subplot(gs[2, 1])
        im10 = plt.imshow(sentinel_lst_lr, cmap=lst_cmap)
        plt.colorbar(im10, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Raw Prediction\n(Linear Regression)', pad=5)
        
        plt.subplot(gs[2, 2])
        im11 = plt.imshow(sentinel_lst_rf, cmap=lst_cmap)
        plt.colorbar(im11, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Raw Prediction\n(Random Forest)', pad=5)
        
        plt.subplot(gs[2, 3])
        im12 = plt.imshow(sentinel_lst_xgb, cmap=lst_cmap)
        plt.colorbar(im12, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Raw Prediction\n(XGBoost)', pad=5)
        
    
        landsat_mean = np.nanmean(landsat_lst)
        lr_residual = np.nanmean(sentinel_lst_lr) - landsat_mean
        rf_residual = np.nanmean(sentinel_lst_rf) - landsat_mean
        xgb_residual = np.nanmean(sentinel_lst_xgb) - landsat_mean
        
        sentinel_lst_lr_adj = sentinel_lst_lr - lr_residual
        sentinel_lst_rf_adj = sentinel_lst_rf - rf_residual
        sentinel_lst_xgb_adj = sentinel_lst_xgb - xgb_residual
        
        plt.subplot(gs[3, 1])
        im14 = plt.imshow(sentinel_lst_lr_adj, cmap=lst_cmap)
        plt.colorbar(im14, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Bias-Corrected\n(Linear Regression)', pad=5)
        
        plt.subplot(gs[3, 2])
        im15 = plt.imshow(sentinel_lst_rf_adj, cmap=lst_cmap)
        plt.colorbar(im15, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Bias-Corrected\n(Random Forest)', pad=5)
        
        plt.subplot(gs[3, 3])
        im16 = plt.imshow(sentinel_lst_xgb_adj, cmap=lst_cmap)
        plt.colorbar(im16, label='Temperature (K)', fraction=0.04, pad=0.02, aspect=23.3)
        plt.title('Bias-Corrected\n(XGBoost)', pad=5)
        
        plt.savefig(self.data_dir / 'results_spatial.png', dpi=300, bbox_inches='tight')
        
    def plot_accuracy(self):
        """Plot LST distribution histograms"""
        # Load data
        landsat_lst, _, _ = self._load_raster('landsat_lst.tif')
        sentinel_lst_lr, _, _ = self._load_raster('sentinel_lst_lr.tif')
        sentinel_lst_rf, _, _ = self._load_raster('sentinel_lst_rf.tif')
        sentinel_lst_xgb, _, _ = self._load_raster('sentinel_lst_xgb.tif')
        
        landsat_mean = np.nanmean(landsat_lst)
        lr_residual = np.nanmean(sentinel_lst_lr) - landsat_mean
        rf_residual = np.nanmean(sentinel_lst_rf) - landsat_mean
        xgb_residual = np.nanmean(sentinel_lst_xgb) - landsat_mean
        
        sentinel_lst_lr_adj = sentinel_lst_lr - lr_residual
        sentinel_lst_rf_adj = sentinel_lst_rf - rf_residual
        sentinel_lst_xgb_adj = sentinel_lst_xgb - xgb_residual
        
        # Prepare data
        x = landsat_lst[~np.isnan(landsat_lst)].flatten()
        y_lr = sentinel_lst_lr[~np.isnan(sentinel_lst_lr)].flatten()
        y_rf = sentinel_lst_rf[~np.isnan(sentinel_lst_rf)].flatten()
        y_xgb = sentinel_lst_xgb[~np.isnan(sentinel_lst_xgb)].flatten()
        y_lr_adj = sentinel_lst_lr_adj[~np.isnan(sentinel_lst_lr_adj)].flatten()
        y_rf_adj = sentinel_lst_rf_adj[~np.isnan(sentinel_lst_rf_adj)].flatten()
        y_xgb_adj = sentinel_lst_xgb_adj[~np.isnan(sentinel_lst_xgb_adj)].flatten()
        
        np.random.seed(42)
        landsat_size = len(x)
        
        def random_sample(data, size):
            if len(data) > size:
                return np.random.choice(data, size, replace=False)
            return data
            
        y_lr_sampled = random_sample(y_lr, landsat_size)
        y_rf_sampled = random_sample(y_rf, landsat_size)
        y_xgb_sampled = random_sample(y_xgb, landsat_size)
        y_lr_adj_sampled = random_sample(y_lr_adj, landsat_size)
        y_rf_adj_sampled = random_sample(y_rf_adj, landsat_size)
        y_xgb_adj_sampled = random_sample(y_xgb_adj, landsat_size)
        
        def print_stats(name, data):
            print(f"\n{name} Statistics:")
            print(f"Count: {len(data)}")
            print(f"Mean: {np.mean(data):.2f}K")
            print(f"Std: {np.std(data):.2f}K")
            print(f"Range: {np.min(data):.2f}K - {np.max(data):.2f}K")
        
        print("\nData Statistics:")
        print_stats("Landsat", x)
        print_stats("Linear Regression (Raw)", y_lr_sampled)
        print_stats("Random Forest (Raw)", y_rf_sampled)
        print_stats("XGBoost (Raw)", y_xgb_sampled)
        print_stats("Linear Regression (Adjusted)", y_lr_adj_sampled)
        print_stats("Random Forest (Adjusted)", y_rf_adj_sampled)
        print_stats("XGBoost (Adjusted)", y_xgb_adj_sampled)
        
        # Create histogram plots
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        
        # Raw predictions
        # Linear Regression
        plt.sca(ax1)
        plt.hist([x, y_lr_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('Linear Regression\n(Raw)')
        plt.legend()
        
        # Random Forest
        plt.sca(ax2)
        plt.hist([x, y_rf_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('Random Forest\n(Raw)')
        plt.legend()
        
        # XGBoost
        plt.sca(ax3)
        plt.hist([x, y_xgb_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('XGBoost\n(Raw)')
        plt.legend()
        
        # Bias-corrected predictions
        # Linear Regression
        plt.sca(ax4)
        plt.hist([x, y_lr_adj_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('Linear Regression\n(Bias-Corrected)')
        plt.legend()
        
        # Random Forest
        plt.sca(ax5)
        plt.hist([x, y_rf_adj_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('Random Forest\n(Bias-Corrected)')
        plt.legend()
        
        # XGBoost
        plt.sca(ax6)
        plt.hist([x, y_xgb_adj_sampled], bins=50, label=['Landsat', 'Predicted'], alpha=0.7)
        plt.xlabel('LST (K)')
        plt.ylabel('Count')
        plt.title('XGBoost\n(Bias-Corrected)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'results_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Initialize visualizer with the correct path
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Initialize visualizer
    visualizer = ResultVisualization(data_dir)
    
    # Plot spatial distribution
    print("Plotting spatial distribution...")
    visualizer.plot_indices()
    
    # Plot accuracy
    print("Plotting accuracy...")
    visualizer.plot_accuracy()
    
    print("Done!")

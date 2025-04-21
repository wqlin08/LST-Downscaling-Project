import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'processed'
        self.output_dir.mkdir(exist_ok=True)
        
    def read_shp(self, shp_path):
        self.shp = gpd.read_file(shp_path)
        return self.shp
    
    def clip_with_shp(self, raster_path, bounds=None):

        with rasterio.open(raster_path) as src:

            if bounds is None:
                out_image, out_transform = mask(src, 
                                             [feature["geometry"] for feature in self.shp.to_dict("records")],
                                             crop=True)
                out_meta = src.meta.copy()
                bounds = rasterio.transform.array_bounds(
                    out_image.shape[1], out_image.shape[2], out_transform
                )
            

            window = src.window(*bounds)
            window = window.round_lengths()
            window = window.round_offsets()
            

            out_image = src.read(1, window=window)
            out_transform = src.window_transform(window)
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform
            })
            
            return out_image, out_transform, out_meta, bounds
    
    def calibrate_landsat(self, array, band_name):

        array = array.astype(np.float32)
        array[array == 0] = np.nan
        
        if 'ST_B10' in band_name:  # lst
            valid_mask = ~np.isnan(array)
            array[valid_mask] = array[valid_mask] * 0.00341802 + 149.0
        else:  
            valid_mask = ~np.isnan(array)

            array[valid_mask] = array[valid_mask] * 0.0000275 - 0.2

            array = np.clip(array, 0, 1)
        return array
            
    def calibrate_sentinel(self, array):
 
        array = array.astype(np.float32)
        array[array == 0] = np.nan
        valid_mask = ~np.isnan(array)
        array[valid_mask] = array[valid_mask] / 10000.0

        array = np.clip(array, 0, 1)
        return array
                
    def resample_sentinel(self, array, transform, src_crs, target_res=10):

        height = array.shape[0]
        width = array.shape[1]
        
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, src_crs, width, height, *rasterio.transform.array_bounds(height, width, transform),
            resolution=target_res
        )
        
        dst_array = np.zeros((1, dst_height, dst_width), dtype=np.float32)
        
        reproject(
            source=array,
            destination=dst_array,
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.bilinear
        )
        
        return dst_array[0], dst_transform
            
    def calculate_indices(self, landsat_dir, sentinel_dir):
        landsat_files = list(Path(landsat_dir).glob("*B*.TIF"))
        lst_file = list(Path(landsat_dir).glob("*ST_B10.TIF"))[0]
        
        lst_array, lst_transform, lst_meta, bounds = self.clip_with_shp(lst_file)
        
        lst_array = self.calibrate_landsat(lst_array, lst_file.name)
        
        lst_output = self.output_dir / 'landsat_lst.tif'
        lst_meta.update({
            'dtype': 'float32',
            'nodata': -9999
        })
        
        with rasterio.open(lst_output, 'w', **lst_meta) as dst:
            lst_array[np.isnan(lst_array)] = -9999
            dst.write(lst_array.astype('float32'), 1)
        
        red_band = [f for f in landsat_files if "B4" in f.name][0]
        nir_band = [f for f in landsat_files if "B5" in f.name][0]
        swir1_band = [f for f in landsat_files if "B6" in f.name][0]
        green_band = [f for f in landsat_files if "B3" in f.name][0]
        
        red, red_transform, red_meta, _ = self.clip_with_shp(red_band, bounds)
        nir, nir_transform, _, _ = self.clip_with_shp(nir_band, bounds)
        swir1, swir1_transform, _, _ = self.clip_with_shp(swir1_band, bounds)
        green, green_transform, _, _ = self.clip_with_shp(green_band, bounds)
        
        red = self.calibrate_landsat(red, red_band.name)
        nir = self.calibrate_landsat(nir, nir_band.name)
        swir1 = self.calibrate_landsat(swir1, swir1_band.name)
        green = self.calibrate_landsat(green, green_band.name)
        
        self._calculate_and_save_indices(
            red, nir, swir1, green, red_meta, red_transform, 'landsat'
        )
        
        sentinel_granule = list(Path(sentinel_dir).glob("GRANULE/*/IMG_DATA/R10m"))[0]
        sentinel_20m = Path(str(sentinel_granule).replace("R10m", "R20m"))
        
        red_band = list(sentinel_granule.glob("*_B04_*.jp2"))[0]
        nir_band = list(sentinel_granule.glob("*_B08_*.jp2"))[0]
        green_band = list(sentinel_granule.glob("*_B03_*.jp2"))[0]
        
        swir1_band = list(sentinel_20m.glob("*_B11_*.jp2"))[0]
        
        red, red_transform, red_meta, sentinel_bounds = self.clip_with_shp(red_band)
        nir, nir_transform, _, _ = self.clip_with_shp(nir_band, sentinel_bounds)
        green, green_transform, _, _ = self.clip_with_shp(green_band, sentinel_bounds)
        swir1, swir1_transform, swir1_meta, _ = self.clip_with_shp(swir1_band, sentinel_bounds)
        
        red = self.calibrate_sentinel(red)
        nir = self.calibrate_sentinel(nir)
        green = self.calibrate_sentinel(green)
        swir1 = self.calibrate_sentinel(swir1)
        
        if swir1_transform[0] != red_transform[0]:
            swir1, swir1_transform = self.resample_sentinel(
                swir1, swir1_transform, swir1_meta['crs'], target_res=red_transform[0]
            )
        
        self._calculate_and_save_indices(
            red, nir, swir1, green, red_meta, red_transform, 'sentinel'
        )
        
    def _calculate_and_save_indices(self, red, nir, swir1, green, meta, transform, sensor_name):
        epsilon = 1e-10
        
        if not (red.shape == nir.shape == swir1.shape == green.shape):
            raise ValueError(f"Inconsistent band shapes: red={red.shape}, nir={nir.shape}, swir1={swir1.shape}, green={green.shape}")
        
        print(f"\n{sensor_name} Band:")
        for name, band in [("Red", red), ("NIR", nir), ("SWIR1", swir1), ("Green", green)]:
            valid_data = band[~np.isnan(band)]
            print(f"{name}: min={np.min(valid_data):.4f}, max={np.max(valid_data):.4f}, mean={np.mean(valid_data):.4f}")
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = np.where(
            ~np.isnan(nir + red),
            (nir - red) / (nir + red + epsilon),
            np.nan
        )
        
        # MNDWI = (Green - SWIR) / (Green + SWIR)
        mndwi = np.where(
            ~np.isnan(green + swir1),
            (green - swir1) / (green + swir1 + epsilon),
            np.nan
        )
        
        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        ndbi = np.where(
            ~np.isnan(swir1 + nir),
            (swir1 - nir) / (swir1 + nir + epsilon),
            np.nan
        )
        
        # SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L), where L is a soil brightness correction factor
        L = 0.5  
        savi = np.where(
            ~np.isnan(nir + red),
            ((nir - red) / (nir + red + L + epsilon)) * (1 + L),
            np.nan
        )

        ndvi = np.clip(ndvi, -1, 1)
        mndwi = np.clip(mndwi, -1, 1)
        ndbi = np.clip(ndbi, -1, 1)
        savi = np.clip(savi, -1, 1)
        
        print(f"\n{sensor_name} Index Statistics:")
        for name, index in [("NDVI", ndvi), ("MNDWI", mndwi), ("NDBI", ndbi), ("SAVI", savi)]:
            valid_data = index[~np.isnan(index)]
            print(f"{name}: min={np.min(valid_data):.4f}, max={np.max(valid_data):.4f}, mean={np.mean(valid_data):.4f}")
        
        indices = {'ndvi': ndvi, 'mndwi': mndwi, 'ndbi': ndbi, 'savi': savi}
        for index_name, index_array in indices.items():
            output_path = self.output_dir / f"{sensor_name}_{index_name}.tif"
            
            meta = meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": index_array.shape[0],
                "width": index_array.shape[1],
                "transform": transform,
                "dtype": 'float32',
                "nodata": -9999
            })

            index_array[np.isnan(index_array)] = -9999
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(index_array.astype('float32'), 1)

if __name__ == "__main__":
    data_dir = ""
    shp_path = os.path.join(data_dir, "SHP", "boundary1.shp")
    landsat_dir = os.path.join(data_dir, "Landsat")
    sentinel_dir = os.path.join(data_dir, "S2")
    
    preprocessor = DataPreprocessor(data_dir)
    
    # read shapefile
    preprocessor.read_shp(shp_path)
    
    preprocessor.calculate_indices(landsat_dir, sentinel_dir) 

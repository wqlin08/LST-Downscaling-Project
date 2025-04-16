import os
import numpy as np
import rasterio
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

class LSTDownscaling:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.model_dir = self.data_dir / 'models'
        
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):

        self.lst = self._load_raster('landsat_lst.tif')
        landsat_ndvi = self._load_raster('landsat_ndvi.tif')
        landsat_mndwi = self._load_raster('landsat_mndwi.tif')
        landsat_ndbi = self._load_raster('landsat_ndbi.tif')
        
        valid_lst = self.lst[~np.isnan(self.lst)]
        print("\nLST:")
        print(f"Min: {np.min(valid_lst):.2f}K")
        print(f"Max: {np.max(valid_lst):.2f}K")
        print(f"Mean: {np.mean(valid_lst):.2f}K")
        
        sentinel_ndvi = self._load_raster('sentinel_ndvi.tif')
        sentinel_mndwi = self._load_raster('sentinel_mndwi.tif')
        sentinel_ndbi = self._load_raster('sentinel_ndbi.tif')
        
        self.sentinel_features = np.vstack((
            sentinel_ndvi.flatten(),
            sentinel_mndwi.flatten(),
            sentinel_ndbi.flatten()
        )).T
        
        self.X = np.vstack((
            landsat_ndvi.flatten(),
            landsat_mndwi.flatten(),
            landsat_ndbi.flatten()
        )).T
        self.y = self.lst.flatten()
        
        # remove nan
        valid_mask = ~np.isnan(self.y)
        for i in range(self.X.shape[1]):
            valid_mask = valid_mask & ~np.isnan(self.X[:, i])
        
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        # use 1%-99% data
        for i in range(self.X.shape[1]):
            p1, p99 = np.percentile(self.X[:, i], [1, 99])
            mask = (self.X[:, i] >= p1) & (self.X[:, i] <= p99)
            self.X = self.X[mask]
            self.y = self.y[mask]
            
        p1, p99 = np.percentile(self.y, [1, 99])
        mask = (self.y >= p1) & (self.y <= p99)
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print("\nTraining Data:")
        print(f"Sample size: {len(self.y)}")
        print(f"LST: {np.min(self.y):.2f}K - {np.max(self.y):.2f}K")
        for i, name in enumerate(['NDVI', 'MNDWI', 'NDBI']):
            print(f"{name}: {np.min(self.X[:, i]):.4f} - {np.max(self.X[:, i]):.4f}")
        
    def _load_raster(self, filename):
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"no files: {filepath}\n"
                f"Please make sure the file exists: {self.processed_dir}"
                f"And the current working directory or script path is correct."
            )
        
        with rasterio.open(filepath) as src:
            data = src.read(1)
            # nodata to nan
            data = data.astype(np.float32)
            data[data == -9999] = np.nan
            return data
            
    def train_models(self):
        """Training multiple linear regression and random forest models"""
        print("\nStart splitting the training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(f"training datasets: {X_train.shape[0]}, test datasets: {X_test.shape[0]}")
        
        print("\nStart training the linear regression model...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        print("Linear regression model training completed")
        
        lr_pred = self.lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        print("\nRandom forest model training completed")
        self.rf_model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=15,     
            min_samples_split=50, 
            random_state=42,
            n_jobs=-1,
            verbose=1 
        )
        self.rf_model.fit(X_train, y_train)
        print("Random forest model training completed")
        
        rf_pred = self.rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        # save models
        joblib.dump(self.lr_model, self.model_dir / 'lr_model.pkl')
        joblib.dump(self.rf_model, self.model_dir / 'rf_model.pkl')
        
        print(f"\nlr model - R2: {lr_r2:.3f}, RMSE: {lr_rmse:.3f}K")
        print(f"rf model - R2: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}K")

        print("\nLinear regression model coefficients:")
        for name, coef in zip(['NDVI', 'MNDWI', 'NDBI'], self.lr_model.coef_):
            print(f"{name}: {coef:.4f}")
        print(f"intercept: {self.lr_model.intercept_:.4f}")
        
    def predict_sentinel_lst(self):
        """Predicting Sentinel LST"""

        valid_mask = np.all(~np.isnan(self.sentinel_features), axis=1)
        valid_features = self.sentinel_features[valid_mask]

        lr_pred = self.lr_model.predict(valid_features)
        rf_pred = self.rf_model.predict(valid_features)

        lr_lst = np.full(self.sentinel_features.shape[0], np.nan)
        rf_lst = np.full(self.sentinel_features.shape[0], np.nan)
        
        lr_lst[valid_mask] = lr_pred
        rf_lst[valid_mask] = rf_pred
        
        original_shape = self._load_raster('sentinel_ndvi.tif').shape
        lr_lst = lr_lst.reshape(original_shape)
        rf_lst = rf_lst.reshape(original_shape)
        
        print("\nForecast result statistics:")
        print("lr model:")
        print(f"LST: {np.nanmin(lr_lst):.2f}K - {np.nanmax(lr_lst):.2f}K")
        print(f"mean: {np.nanmean(lr_lst):.2f}K")
        print("\nrf model:")
        print(f"LST: {np.nanmin(rf_lst):.2f}K - {np.nanmax(rf_lst):.2f}K")
        print(f"mean: {np.nanmean(rf_lst):.2f}K")
        
        self._save_raster(lr_lst, 'sentinel_lst_lr.tif')
        self._save_raster(rf_lst, 'sentinel_lst_rf.tif')
        
    def _save_raster(self, array, filename):
        with rasterio.open(self.processed_dir / 'sentinel_ndvi.tif') as src:
            meta = src.meta.copy()
            meta.update({
                'dtype': 'float32',
                'nodata': -9999
            })
            
            array = array.copy()
            array[np.isnan(array)] = -9999
            
            with rasterio.open(self.processed_dir / filename, 'w', **meta) as dst:
                dst.write(array.astype('float32'), 1)

if __name__ == "__main__":

    script_dir = Path(__file__).parent

    data_dir = script_dir / "data"

    downscaler = LSTDownscaling(data_dir)

    print("load data...")
    downscaler.load_data()
    

    print("\ntraining model...")
    downscaler.train_models()
    
    print("\npredicting Sentinel LST...")
    downscaler.predict_sentinel_lst()
    
    print("\nFinish")
import os
import numpy as np
import rasterio
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

class LSTDownscaling:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.model_dir = self.data_dir / 'models'
        
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load processed index data"""
        self.lst = self._load_raster('landsat_lst.tif')
        landsat_ndvi = self._load_raster('landsat_ndvi.tif')
        landsat_mndwi = self._load_raster('landsat_mndwi.tif')
        landsat_ndbi = self._load_raster('landsat_ndbi.tif')
        landsat_savi = self._load_raster('landsat_savi.tif')
        
        valid_lst = self.lst[~np.isnan(self.lst)]
        print("\nLST Statistics:")
        print(f"Min: {np.min(valid_lst):.2f}K")
        print(f"Max: {np.max(valid_lst):.2f}K")
        print(f"Mean: {np.mean(valid_lst):.2f}K")
        
        sentinel_ndvi = self._load_raster('sentinel_ndvi.tif')
        sentinel_mndwi = self._load_raster('sentinel_mndwi.tif')
        sentinel_ndbi = self._load_raster('sentinel_ndbi.tif')
        sentinel_savi = self._load_raster('sentinel_savi.tif')
        
        self.sentinel_features = np.vstack((
            sentinel_ndvi.flatten(),
            sentinel_mndwi.flatten(),
            sentinel_ndbi.flatten(),
            sentinel_savi.flatten()
        )).T
        
        self.X = np.vstack((
            landsat_ndvi.flatten(),
            landsat_mndwi.flatten(),
            landsat_ndbi.flatten(),
            landsat_savi.flatten()
        )).T
        self.y = self.lst.flatten()
        
        valid_mask = ~np.isnan(self.y)
        for i in range(self.X.shape[1]):
            valid_mask = valid_mask & ~np.isnan(self.X[:, i])
        
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        for i in range(self.X.shape[1]):
            p1, p99 = np.percentile(self.X[:, i], [1, 99])
            mask = (self.X[:, i] >= p1) & (self.X[:, i] <= p99)
            self.X = self.X[mask]
            self.y = self.y[mask]
            
        p1, p99 = np.percentile(self.y, [1, 99])
        mask = (self.y >= p1) & (self.y <= p99)
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print("\nTraining Data Statistics:")
        print(f"Sample size: {len(self.y)}")
        print(f"LST range: {np.min(self.y):.2f}K - {np.max(self.y):.2f}K")
        for i, name in enumerate(['NDVI', 'MNDWI', 'NDBI', 'SAVI']):
            print(f"{name} range: {np.min(self.X[:, i]):.4f} - {np.max(self.X[:, i]):.4f}")
            
    def analyze_shap_values(self, X_test, model, model_name):
        """Calculate and plot SHAP values"""
        print(f"\nCalculating feature importance for {model_name}...")
        
        if isinstance(model, LinearRegression):
           
            feature_importance = np.abs(model.coef_)
            importance_df = pd.DataFrame({
                'Feature': ['NDVI', 'MNDWI', 'NDBI', 'SAVI'],
                'Importance': feature_importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.title(f'Feature Importance - {model_name}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.data_dir / f'feature_importance_{model_name.lower()}.png')
            plt.close()
            
        elif isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
            if len(X_test) > 1000:
                np.random.seed(42)
                sample_idx = np.random.choice(len(X_test), 1000, replace=False)
                X_sample = X_test[sample_idx]
            else:
                X_sample = X_test
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, 
                            feature_names=['NDVI', 'MNDWI', 'NDBI', 'SAVI'],
                            show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(self.data_dir / f'shap_summary_{model_name.lower()}.png')
            plt.close()
            
            # Calculate and print feature importance based on SHAP values
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'Feature': ['NDVI', 'MNDWI', 'NDBI', 'SAVI'],
                'Importance': feature_importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print(f"\n{model_name} Feature Importance:")
        for _, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
            
    def plot_residuals(self, y_true, y_pred, model_name):
        """Plot residuals"""
        if len(y_true) > 10000:
            np.random.seed(42)
            sample_idx = np.random.choice(len(y_true), 10000, replace=False)
            y_true = y_true[sample_idx]
            y_pred = y_pred[sample_idx]
            
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(y_pred, residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted LST (K)')
        plt.ylabel('Residuals (K)')
        plt.title(f'Residuals vs Predicted - {model_name}')
        
        plt.subplot(122)
        plt.hist(residuals, bins=50, density=True)
        plt.xlabel('Residuals (K)')
        plt.ylabel('Density')
        plt.title(f'Residuals Distribution - {model_name}')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / f'residuals_{model_name.lower()}.png')
        plt.close()
        
        print(f"\n{model_name} Residuals Statistics:")
        print(f"Mean: {np.mean(residuals):.4f}K")
        print(f"Std: {np.std(residuals):.4f}K")
        print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.4f}K")
            
    def train_models(self):
        """Train multiple regression models"""
        print("\nSplitting train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        print("\nTraining Linear Regression model...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        print("\nTraining Random Forest model...")
        self.rf_model = RandomForestRegressor(
            n_estimators=50, max_depth=15, min_samples_split=50,
            random_state=42, n_jobs=-1, verbose=1
        )
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        print("\nTraining XGBoost model...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=1
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        
        # save models
        joblib.dump(self.lr_model, self.model_dir / 'lr_model.pkl')
        joblib.dump(self.rf_model, self.model_dir / 'rf_model.pkl')
        joblib.dump(self.xgb_model, self.model_dir / 'xgb_model.pkl')
        

        print("\nModel Results:")
        print(f"Linear Regression - R2: {lr_r2:.3f}, RMSE: {lr_rmse:.3f}K")
        print(f"Random Forest - R2: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}K")
        print(f"XGBoost - R2: {xgb_r2:.3f}, RMSE: {xgb_rmse:.3f}K")
        
        self.analyze_shap_values(X_test, self.lr_model, "Linear Regression")
        self.analyze_shap_values(X_test, self.rf_model, "Random Forest")
        self.analyze_shap_values(X_test, self.xgb_model, "XGBoost")
        
        self.plot_residuals(y_test, lr_pred, "Linear Regression")
        self.plot_residuals(y_test, rf_pred, "Random Forest")
        self.plot_residuals(y_test, xgb_pred, "XGBoost")
        
    def predict_sentinel_lst(self):
        """Predict Sentinel LST using trained models"""
        valid_mask = np.all(~np.isnan(self.sentinel_features), axis=1)
        valid_features = self.sentinel_features[valid_mask]
        
        lr_pred = self.lr_model.predict(valid_features)
        rf_pred = self.rf_model.predict(valid_features)
        xgb_pred = self.xgb_model.predict(valid_features)
        
        landsat_mean = np.nanmean(self.lst)
        lr_residual = np.mean(lr_pred) - landsat_mean
        rf_residual = np.mean(rf_pred) - landsat_mean
        xgb_residual = np.mean(xgb_pred) - landsat_mean
        
        lr_pred_adj = lr_pred - lr_residual
        rf_pred_adj = rf_pred - rf_residual
        xgb_pred_adj = xgb_pred - xgb_residual
        
        lr_lst = np.full(self.sentinel_features.shape[0], np.nan)
        rf_lst = np.full(self.sentinel_features.shape[0], np.nan)
        xgb_lst = np.full(self.sentinel_features.shape[0], np.nan)
        lr_lst_adj = np.full(self.sentinel_features.shape[0], np.nan)
        rf_lst_adj = np.full(self.sentinel_features.shape[0], np.nan)
        xgb_lst_adj = np.full(self.sentinel_features.shape[0], np.nan)
        
        lr_lst[valid_mask] = lr_pred
        rf_lst[valid_mask] = rf_pred
        xgb_lst[valid_mask] = xgb_pred
        lr_lst_adj[valid_mask] = lr_pred_adj
        rf_lst_adj[valid_mask] = rf_pred_adj
        xgb_lst_adj[valid_mask] = xgb_pred_adj
        
        original_shape = self._load_raster('sentinel_ndvi.tif').shape
        lr_lst = lr_lst.reshape(original_shape)
        rf_lst = rf_lst.reshape(original_shape)
        xgb_lst = xgb_lst.reshape(original_shape)
        lr_lst_adj = lr_lst_adj.reshape(original_shape)
        rf_lst_adj = rf_lst_adj.reshape(original_shape)
        xgb_lst_adj = xgb_lst_adj.reshape(original_shape)

        self._save_raster(lr_lst, 'sentinel_lst_lr.tif')
        self._save_raster(rf_lst, 'sentinel_lst_rf.tif')
        self._save_raster(xgb_lst, 'sentinel_lst_xgb.tif')
        self._save_raster(lr_lst_adj, 'sentinel_lst_lr_adj.tif')
        self._save_raster(rf_lst_adj, 'sentinel_lst_rf_adj.tif')
        self._save_raster(xgb_lst_adj, 'sentinel_lst_xgb_adj.tif')
        
        print("\nPrediction Statistics (Raw):")
        print("Linear Regression:")
        print(f"LST range: {np.nanmin(lr_lst):.2f}K - {np.nanmax(lr_lst):.2f}K")
        print(f"Mean: {np.nanmean(lr_lst):.2f}K")
        print(f"Residual: {lr_residual:.2f}K")
        print("\nRandom Forest:")
        print(f"LST range: {np.nanmin(rf_lst):.2f}K - {np.nanmax(rf_lst):.2f}K")
        print(f"Mean: {np.nanmean(rf_lst):.2f}K")
        print(f"Residual: {rf_residual:.2f}K")
        print("\nXGBoost:")
        print(f"LST range: {np.nanmin(xgb_lst):.2f}K - {np.nanmax(xgb_lst):.2f}K")
        print(f"Mean: {np.nanmean(xgb_lst):.2f}K")
        print(f"Residual: {xgb_residual:.2f}K")
        
        print("\nPrediction Statistics (Bias-Corrected):")
        print("Linear Regression:")
        print(f"LST range: {np.nanmin(lr_lst_adj):.2f}K - {np.nanmax(lr_lst_adj):.2f}K")
        print(f"Mean: {np.nanmean(lr_lst_adj):.2f}K")
        print("\nRandom Forest:")
        print(f"LST range: {np.nanmin(rf_lst_adj):.2f}K - {np.nanmax(rf_lst_adj):.2f}K")
        print(f"Mean: {np.nanmean(rf_lst_adj):.2f}K")
        print("\nXGBoost:")
        print(f"LST range: {np.nanmin(xgb_lst_adj):.2f}K - {np.nanmax(xgb_lst_adj):.2f}K")
        print(f"Mean: {np.nanmean(xgb_lst_adj):.2f}K")
        
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

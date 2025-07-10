import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class WeatherModelEvaluator:
    def __init__(self, model_base_path='sri_lanka_weather_model'):
        """
        Initialize the model evaluator
        
        Args:
            model_base_path: Base path/name for saved model files
        """
        self.model_base_path = model_base_path
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.model_weights = {}
        
        # Weather parameter descriptions
        self.parameter_descriptions = {
            'temperature_2m_max': 'Maximum Temperature (°C)',
            'temperature_2m_min': 'Minimum Temperature (°C)',
            'temperature_2m_mean': 'Mean Temperature (°C)',
            'apparent_temperature_max': 'Maximum Apparent Temperature (°C)',
            'apparent_temperature_min': 'Minimum Apparent Temperature (°C)',
            'apparent_temperature_mean': 'Mean Apparent Temperature (°C)',
            'precipitation_sum': 'Total Precipitation (mm)',
            'rain_sum': 'Total Rain (mm)',
            'windspeed_10m_max': 'Maximum Wind Speed (km/h)',
            'windgusts_10m_max': 'Maximum Wind Gusts (km/h)',
            'shortwave_radiation_sum': 'Solar Radiation (MJ/m²)',
            'et0_fao_evapotranspiration': 'Evapotranspiration (mm)',
            'precipitation_hours': 'Precipitation Hours'
        }
        
    def load_models(self):
        """Load all saved models and preprocessing components"""
        print("Loading saved models and preprocessors...")
        
        try:
            # Load sklearn models
            model_names = ['xgboost', 'lightgbm', 'random_forest']
            for name in model_names:
                try:
                    self.models[name] = joblib.load(f'{self.model_base_path}_{name}.pkl')
                    print(f"✓ Loaded {name} model")
                except FileNotFoundError:
                    print(f"✗ {name} model not found")
            
            # Load neural network
            try:
                self.models['neural_network'] = keras.models.load_model(
                    f'{self.model_base_path}_neural_network.h5'
                )
                print("✓ Loaded neural network model")
            except FileNotFoundError:
                print("✗ Neural network model not found")
            
            # Load preprocessors
            self.scalers = joblib.load(f'{self.model_base_path}_scalers.pkl')
            self.label_encoders = joblib.load(f'{self.model_base_path}_label_encoders.pkl')
            self.feature_columns = joblib.load(f'{self.model_base_path}_feature_columns.pkl')
            self.target_columns = joblib.load(f'{self.model_base_path}_target_columns.pkl')
            self.model_weights = joblib.load(f'{self.model_base_path}_weights.pkl')
            
            print("✓ All preprocessors loaded successfully")
            print(f"✓ Loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_test_data(self, file_path):
        """Load and preprocess test data"""
        print("Loading and preprocessing test data...")
        
        df = pd.read_csv(file_path)
        
        # Apply same preprocessing as training
        df = self._preprocess_data(df)
        
        # Prepare features
        X = df[self.feature_columns]
        y = df[self.target_columns]
        
        print(f"Test data shape: {X.shape}")
        return X, y, df
    
    def _preprocess_data(self, df):
        """Apply same preprocessing as training data"""
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        if 'sunrise' in df.columns:
            df['sunrise'] = pd.to_datetime(df['sunrise'])
        if 'sunset' in df.columns:
            df['sunset'] = pd.to_datetime(df['sunset'])
        
        # Extract time features
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['day_of_year'] = df['time'].dt.dayofyear
        df['week_of_year'] = df['time'].dt.isocalendar().week
        df['season'] = df['month'].apply(self._get_season)
        
        # Calculate daylight hours (if sunrise/sunset available)
        if 'sunrise' in df.columns and 'sunset' in df.columns:
            df['daylight_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
        else:
            df['daylight_hours'] = 12  # Default for Sri Lanka
        
        # Add cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Encode categorical variables
        if 'city' in df.columns:
            df['city_encoded'] = self.label_encoders['city'].transform(df['city'])
        else:
            df['city_encoded'] = 0
            
        if 'country' in df.columns:
            df['country_encoded'] = self.label_encoders['country'].transform(df['country'])
        else:
            df['country_encoded'] = 0
        
        # Create lag features
        df = df.sort_values(['city', 'time']) if 'city' in df.columns else df.sort_values('time')
        
        lag_columns = ['temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_max']
        for col in lag_columns:
            if col in df.columns:
                if 'city' in df.columns:
                    df[f'{col}_lag1'] = df.groupby('city')[col].shift(1)
                    df[f'{col}_lag7'] = df.groupby('city')[col].shift(7)
                    df[f'{col}_ma7'] = df.groupby('city')[col].rolling(window=7).mean().values
                else:
                    df[f'{col}_lag1'] = df[col].shift(1)
                    df[f'{col}_lag7'] = df[col].shift(7)
                    df[f'{col}_ma7'] = df[col].rolling(window=7).mean()
            else:
                # Use defaults if column not present
                df[f'{col}_lag1'] = 0
                df[f'{col}_lag7'] = 0
                df[f'{col}_ma7'] = 0
        
        # Add season encoding
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Fill missing values
        df = df.fillna(method='forward').fillna(method='backward').fillna(0)
        
        return df
    
    def _get_season(self, month):
        """Get season based on month (Sri Lankan climate)"""
        if month in [12, 1, 2]:
            return 'NE_Monsoon'
        elif month in [3, 4, 5]:
            return 'Inter_Monsoon_1'
        elif month in [6, 7, 8, 9]:
            return 'SW_Monsoon'
        else:
            return 'Inter_Monsoon_2'
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and return comprehensive metrics"""
        print("Evaluating all models...")
        
        # Scale features
        X_test_scaled = self.scalers['feature_scaler'].transform(X_test)
        
        results = {}
        predictions = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Make predictions
            if model_name == 'neural_network':
                y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            else:
                y_pred_scaled = model.predict(X_test_scaled)
            
            # Inverse transform predictions
            y_pred = self.scalers['target_scaler'].inverse_transform(y_pred_scaled)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            results[model_name] = self._calculate_detailed_metrics(y_test, y_pred)
        
        # Calculate ensemble prediction
        print("Calculating ensemble prediction...")
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, weight in self.model_weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        results['ensemble'] = self._calculate_detailed_metrics(y_test, ensemble_pred)
        predictions['ensemble'] = ensemble_pred
        
        return results, predictions
    
    def _calculate_detailed_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'Explained Variance': explained_variance_score(y_true, y_pred)
        }
        
        # Per-parameter metrics
        metrics['per_parameter'] = {}
        for i, param in enumerate(self.target_columns):
            metrics['per_parameter'][param] = {
                'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'R²': r2_score(y_true[:, i], y_pred[:, i]),
                'MAPE': mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
            }
        
        return metrics
    
    def create_evaluation_report(self, results, save_path='model_evaluation_report.html'):
        """Create comprehensive evaluation report"""
        print("Creating evaluation report...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Overall MAE Comparison', 'Overall RMSE Comparison',
                'Overall R² Comparison', 'Overall MAPE Comparison',
                'Model Performance Radar', 'Parameter-wise Performance'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatterpolar"}, {"type": "heatmap"}]]
        )
        
        models = list(results.keys())
        
        # MAE comparison
        mae_values = [results[model]['overall']['MAE'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE comparison
        rmse_values = [results[model]['overall']['RMSE'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # R² comparison
        r2_values = [results[model]['overall']['R²'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # MAPE comparison
        mape_values = [results[model]['overall']['MAPE'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=mape_values, name='MAPE', marker_color='lightyellow'),
            row=2, col=2
        )
        
        # Radar chart for best model
        best_model = min(models, key=lambda x: results[x]['overall']['MAE'])
        best_metrics = results[best_model]['overall']
        
        fig.add_trace(
            go.Scatterpolar(
                r=[best_metrics['R²'], 1-best_metrics['MAPE']/100, 
                   best_metrics['Explained Variance'], 1-best_metrics['MAE']/10],
                theta=['R²', 'MAPE (inverted)', 'Explained Variance', 'MAE (inverted)'],
                fill='toself',
                name=f'Best Model: {best_model}'
            ),
            row=3, col=1
        )
        
        # Parameter-wise heatmap
        param_mae_matrix = []
        for model in models:
            param_mae_row = []
            for param in self.target_columns:
                param_mae_row.append(results[model]['per_parameter'][param]['MAE'])
            param_mae_matrix.append(param_mae_row)
        
        fig.add_trace(
            go.Heatmap(
                z=param_mae_matrix,
                x=[desc.split(' ')[0] for desc in self.target_columns],
                y=models,
                colorscale='RdYlBu_r',
                name='Parameter MAE'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            title_text="Sri Lanka Weather Model Evaluation Report",
            showlegend=False
        )
        
        # Save report
        fig.write_html(save_path)
        print(f"Evaluation report saved to {save_path}")
        
        return fig
    
    def create_prediction_comparison(self, y_true, predictions, sample_size=100):
        """Create prediction comparison visualizations"""
        print("Creating prediction comparison plots...")
        
        # Select random sample for visualization
        indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Temperature Predictions', 'Precipitation Predictions',
                'Wind Speed Predictions', 'Prediction Accuracy Distribution'
            ]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Temperature comparison
        temp_idx = self.target_columns.index('temperature_2m_mean')
        for i, (model, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=y_true[indices, temp_idx],
                    y=pred[indices, temp_idx],
                    mode='markers',
                    name=f'{model} Temperature',
                    marker=dict(color=colors[i % len(colors)], opacity=0.6)
                ),
                row=1, col=1
            )
        
        # Add perfect prediction line
        temp_range = [y_true[:, temp_idx].min(), y_true[:, temp_idx].max()]
        fig.add_trace(
            go.Scatter(
                x=temp_range, y=temp_range,
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )
        
        # Precipitation comparison
        precip_idx = self.target_columns.index('precipitation_sum')
        for i, (model, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=y_true[indices, precip_idx],
                    y=pred[indices, precip_idx],
                    mode='markers',
                    name=f'{model} Precipitation',
                    marker=dict(color=colors[i % len(colors)], opacity=0.6),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Wind speed comparison
        wind_idx = self.target_columns.index('windspeed_10m_max')
        for i, (model, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=y_true[indices, wind_idx],
                    y=pred[indices, wind_idx],
                    mode='markers',
                    name=f'{model} Wind Speed',
                    marker=dict(color=colors[i % len(colors)], opacity=0.6),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Accuracy distribution
        for i, (model, pred) in enumerate(predictions.items()):
            accuracy = 1 - np.abs(y_true - pred) / (y_true + 1e-8)
            accuracy_mean = np.mean(accuracy, axis=1)
            
            fig.add_trace(
                go.Histogram(
                    x=accuracy_mean,
                    name=f'{model} Accuracy',
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Prediction Comparison Analysis"
        )
        
        fig.write_html('prediction_comparison.html')
        print("Prediction comparison saved to prediction_comparison.html")
        
        return fig
    
    def print_detailed_results(self, results):
        """Print detailed evaluation results"""
        print("\n" + "="*80)
        print("DETAILED MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Overall comparison
        print("\nOVERALL PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        df_overall = pd.DataFrame({
            model: metrics['overall'] for model, metrics in results.items()
        }).T
        
        print(df_overall.round(4))
        
        # Best model identification
        best_model = min(results.keys(), key=lambda x: results[x]['overall']['MAE'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper()}")
        print(f"   MAE: {results[best_model]['overall']['MAE']:.4f}")
        print(f"   RMSE: {results[best_model]['overall']['RMSE']:.4f}")
        print(f"   R²: {results[best_model]['overall']['R²']:.4f}")
        
        # Parameter-wise analysis
        print("\nPARAMETER-WISE PERFORMANCE (MAE):")
        print("-" * 50)
        
        for param in self.target_columns:
            print(f"\n{self.parameter_descriptions.get(param, param)}:")
            param_results = {}
            for model in results.keys():
                param_results[model] = results[model]['per_parameter'][param]['MAE']
            
            # Sort by performance
            sorted_results = sorted(param_results.items(), key=lambda x: x[1])
            for model, mae in sorted_results:
                print(f"  {model:15}: {mae:.4f}")
        
        # Model recommendations
        print("\nMODEL RECOMMENDATIONS:")
        print("-" * 50)
        
        # Find best model for each parameter
        best_for_param = {}
        for param in self.target_columns:
            best_model_param = min(results.keys(), 
                                 key=lambda x: results[x]['per_parameter'][param]['MAE'])
            best_for_param[param] = best_model_param
        
        print("\nBest models for specific parameters:")
        for param, model in best_for_param.items():
            param_name = self.parameter_descriptions.get(param, param)
            mae = results[model]['per_parameter'][param]['MAE']
            print(f"  {param_name}: {model} (MAE: {mae:.4f})")
        
        # Ensemble analysis
        if 'ensemble' in results:
            print(f"\n🔄 ENSEMBLE MODEL PERFORMANCE:")
            ensemble_metrics = results['ensemble']['overall']
            print(f"   MAE: {ensemble_metrics['MAE']:.4f}")
            print(f"   RMSE: {ensemble_metrics['RMSE']:.4f}")
            print(f"   R²: {ensemble_metrics['R²']:.4f}")
            
            # Compare ensemble vs best individual
            if ensemble_metrics['MAE'] < results[best_model]['overall']['MAE']:
                print("   ✅ Ensemble outperforms best individual model!")
            else:
                print("   ⚠️  Best individual model outperforms ensemble")
    
    def run_complete_evaluation(self, test_data_path):
        """Run complete evaluation pipeline"""
        print("🚀 Starting Complete Model Evaluation Pipeline")
        print("=" * 80)
        
        # Load models
        self.load_models()
        
        # Load test data
        X_test, y_test, df_test = self.load_test_data(test_data_path)
        
        # Evaluate all models
        results, predictions = self.evaluate_all_models(X_test, y_test.values)
        
        # Print detailed results
        self.print_detailed_results(results)
        
        # Create visualizations
        self.create_evaluation_report(results)
        self.create_prediction_comparison(y_test.values, predictions)
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("Files created:")
        print("  - model_evaluation_report.html")
        print("  - prediction_comparison.html")
        print("="*80)
        
        return results, predictions
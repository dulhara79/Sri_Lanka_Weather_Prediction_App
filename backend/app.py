from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import os
from werkzeug.exceptions import BadRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class WeatherPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        self.model_weights = {}
        self.loaded = False

    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            logger.info("Loading models...")

            # Load sklearn models
            self.models['xgboost'] = joblib.load('models/sri_lanka_weather_model_xgboost.pkl')
            self.models['lightgbm'] = joblib.load('models/sri_lanka_weather_model_lightgbm.pkl')
            self.models['random_forest'] = joblib.load('models/sri_lanka_weather_model_random_forest.pkl')

            # Load neural network with custom metric
            @tf.keras.saving.register_keras_serializable()
            def mse(y_true, y_pred):
                return tf.keras.metrics.mean_squared_error(y_true, y_pred)

            custom_objects = {'mse': mse}
            self.models['neural_network'] = tf.keras.models.load_model(
                'models/sri_lanka_weather_model_neural_network.h5',
                custom_objects=custom_objects
            )

            # Load scalers and encoders
            self.scalers = joblib.load('models/sri_lanka_weather_model_scalers.pkl')
            self.label_encoders = joblib.load('models/sri_lanka_weather_model_label_encoders.pkl')
            self.feature_columns = joblib.load('models/sri_lanka_weather_model_feature_columns.pkl')
            self.target_columns = joblib.load('models/sri_lanka_weather_model_target_columns.pkl')
            self.model_weights = joblib.load('models/sri_lanka_weather_model_weights.pkl')

            self.loaded = True
            logger.info("Models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise

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

    def predict_weather(self, date, city, latitude=None, longitude=None, elevation=None):
        """Predict weather for a specific date and city"""
        if not self.loaded:
            raise Exception("Models not loaded")

        try:
            date_obj = pd.to_datetime(date)

            # Default coordinates for major Sri Lankan cities
            city_coordinates = {
                'Colombo': {'lat': 6.9271, 'lon': 79.8612, 'elev': 8},
                'Kandy': {'lat': 7.2906, 'lon': 80.6337, 'elev': 488},
                'Galle': {'lat': 6.0535, 'lon': 80.2210, 'elev': 13},
                'Jaffna': {'lat': 9.6615, 'lon': 80.0255, 'elev': 9},
                'Trincomalee': {'lat': 8.5874, 'lon': 81.2152, 'elev': 6},
                'Anuradhapura': {'lat': 8.3114, 'lon': 80.4037, 'elev': 81},
                'Batticaloa': {'lat': 7.7170, 'lon': 81.7000, 'elev': 2},
                'Negombo': {'lat': 7.2084, 'lon': 79.8358, 'elev': 6},
                'Matara': {'lat': 5.9549, 'lon': 80.5550, 'elev': 4},
                'Ratnapura': {'lat': 6.6828, 'lon': 80.4014, 'elev': 34}
            }

            # Use provided coordinates or defaults
            if city in city_coordinates:
                coords = city_coordinates[city]
                lat = latitude or coords['lat']
                lon = longitude or coords['lon']
                elev = elevation or coords['elev']
            else:
                lat = latitude or 7.8731  # Default to center of Sri Lanka
                lon = longitude or 80.7718
                elev = elevation or 50

            # Get city encoding
            if city in self.label_encoders['city'].classes_:
                city_encoded = self.label_encoders['city'].transform([city])[0]
            else:
                city_encoded = 0  # Default encoding for unknown cities

            # Create feature vector
            features = {
                'weathercode': 1,
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'day_of_year': date_obj.dayofyear,
                'week_of_year': date_obj.isocalendar().week,
                'month_sin': np.sin(2 * np.pi * date_obj.month / 12),
                'month_cos': np.cos(2 * np.pi * date_obj.month / 12),
                'day_sin': np.sin(2 * np.pi * date_obj.dayofyear / 365),
                'day_cos': np.cos(2 * np.pi * date_obj.dayofyear / 365),
                'city_encoded': city_encoded,
                'country_encoded': 0,
                'daylight_hours': 12,
                'winddirection_10m_dominant': 180
            }

            # Add seasonal features
            season = self._get_season(date_obj.month)
            for s in ['NE_Monsoon', 'Inter_Monsoon_1', 'SW_Monsoon', 'Inter_Monsoon_2']:
                features[f'season_{s}'] = 1 if season == s else 0

            # Add lag features with seasonal defaults
            seasonal_temp = {
                'NE_Monsoon': 25.0, 'Inter_Monsoon_1': 28.0,
                'SW_Monsoon': 26.0, 'Inter_Monsoon_2': 27.0
            }
            seasonal_rain = {
                'NE_Monsoon': 5.0, 'Inter_Monsoon_1': 8.0,
                'SW_Monsoon': 12.0, 'Inter_Monsoon_2': 3.0
            }

            lag_defaults = {
                'temperature_2m_mean_lag1': seasonal_temp[season],
                'temperature_2m_mean_lag7': seasonal_temp[season],
                'temperature_2m_mean_ma7': seasonal_temp[season],
                'precipitation_sum_lag1': seasonal_rain[season],
                'precipitation_sum_lag7': seasonal_rain[season],
                'precipitation_sum_ma7': seasonal_rain[season],
                'windspeed_10m_max_lag1': 12.0,
                'windspeed_10m_max_lag7': 12.0,
                'windspeed_10m_max_ma7': 12.0
            }
            features.update(lag_defaults)

            # Create feature vector in correct order
            feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)

            # Scale features
            feature_vector_scaled = self.scalers['feature_scaler'].transform(feature_vector)

            # Make ensemble prediction
            predictions = {}
            for model_name, model in self.models.items():
                if model_name == 'neural_network':
                    pred_scaled = model.predict(feature_vector_scaled, verbose=0)
                else:
                    pred_scaled = model.predict(feature_vector_scaled)

                pred = self.scalers['target_scaler'].inverse_transform(pred_scaled)
                predictions[model_name] = pred[0]

            # Calculate ensemble prediction
            ensemble_pred = np.zeros(len(self.target_columns))
            for model_name, weight in self.model_weights.items():
                ensemble_pred += weight * predictions[model_name]

            # Create result dictionary
            result = {}
            for i, col in enumerate(self.target_columns):
                result[col] = float(ensemble_pred[i])

            # Add weather interpretation
            result['weather_condition'] = self._interpret_weather(result)
            result['season'] = season
            result['city'] = city
            result['date'] = date
            result['coordinates'] = {'latitude': lat, 'longitude': lon, 'elevation': elev}

            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _interpret_weather(self, weather_data):
        """Interpret weather data into human-readable conditions"""
        temp_max = weather_data['temperature_2m_max']
        temp_min = weather_data['temperature_2m_min']
        precipitation = weather_data['precipitation_sum']
        wind_speed = weather_data['windspeed_10m_max']

        # Temperature interpretation
        if temp_max > 35:
            temp_desc = "Very Hot"
        elif temp_max > 30:
            temp_desc = "Hot"
        elif temp_max > 25:
            temp_desc = "Warm"
        elif temp_max > 20:
            temp_desc = "Mild"
        else:
            temp_desc = "Cool"

        # Precipitation interpretation
        if precipitation > 50:
            rain_desc = "Heavy Rain"
        elif precipitation > 20:
            rain_desc = "Moderate Rain"
        elif precipitation > 5:
            rain_desc = "Light Rain"
        elif precipitation > 0.1:
            rain_desc = "Drizzle"
        else:
            rain_desc = "No Rain"

        # Wind interpretation
        if wind_speed > 40:
            wind_desc = "Very Windy"
        elif wind_speed > 25:
            wind_desc = "Windy"
        elif wind_speed > 15:
            wind_desc = "Breezy"
        else:
            wind_desc = "Calm"

        return {
            'temperature': temp_desc,
            'precipitation': rain_desc,
            'wind': wind_desc,
            'summary': f"{temp_desc} with {rain_desc.lower()}, {wind_desc.lower()} conditions"
        }

# Initialize predictor
predictor = WeatherPredictor()

@app.before_request
def load_models_if_needed():
    """Load models when the app starts"""
    if not hasattr(app, 'models_loaded'):
        try:
            predictor.load_models()
            app.models_loaded = True
            logger.info("Models loaded successfully on first request")
        except Exception as e:
            app.models_loaded = False
            logger.error(f"Error loading models on first request: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_weather_route():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        print(data)

        if not data:
            raise BadRequest("No JSON data provided")

        # Validate required fields
        required_fields = ['date', 'city']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")

        # Extract parameters
        date = data['date']
        city = data['city']
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        elevation = data.get('elevation')

        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise BadRequest("Invalid date format. Use YYYY-MM-DD")

        # Make prediction
        prediction = predictor.predict_weather(date, city, latitude, longitude, elevation)

        return jsonify({
            'success': True,
            'data': prediction,
            'timestamp': datetime.now().isoformat()
        })

    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple dates/cities"""
    try:
        data = request.get_json()

        if not data or 'predictions' not in data:
            raise BadRequest("No prediction data provided")

        predictions = []
        errors = []

        for i, pred_data in enumerate(data['predictions']):
            try:
                # Validate each prediction request
                if 'date' not in pred_data or 'city' not in pred_data:
                    errors.append(f"Request {i}: Missing required fields")
                    continue

                prediction = predictor.predict_weather(
                    pred_data['date'],
                    pred_data['city'],
                    pred_data.get('latitude'),
                    pred_data.get('longitude'),
                    pred_data.get('elevation')
                )
                predictions.append(prediction)

            except Exception as e:
                errors.append(f"Request {i}: {str(e)}")

        return jsonify({
            'success': True,
            'data': predictions,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/cities', methods=['GET'])
def get_cities():
    """Get list of supported cities"""
    cities = [
        {'name': 'Colombo', 'latitude': 6.9271, 'longitude': 79.8612, 'elevation': 8},
        {'name': 'Kandy', 'latitude': 7.2906, 'longitude': 80.6337, 'elevation': 488},
        {'name': 'Galle', 'latitude': 6.0535, 'longitude': 80.2210, 'elevation': 13},
        {'name': 'Jaffna', 'latitude': 9.6615, 'longitude': 80.0255, 'elevation': 9},
        {'name': 'Trincomalee', 'latitude': 8.5874, 'longitude': 81.2152, 'elevation': 6},
        {'name': 'Anuradhapura', 'latitude': 8.3114, 'longitude': 80.4037, 'elevation': 81},
        {'name': 'Batticaloa', 'latitude': 7.7170, 'longitude': 81.7000, 'elevation': 2},
        {'name': 'Negombo', 'latitude': 7.2084, 'longitude': 79.8358, 'elevation': 6},
        {'name': 'Matara', 'latitude': 5.9549, 'longitude': 80.5550, 'elevation': 4},
        {'name': 'Ratnapura', 'latitude': 6.6828, 'longitude': 80.4014, 'elevation': 34}
    ]

    return jsonify({
        'success': True,
        'data': cities,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/forecast', methods=['POST'])
def get_forecast():
    """Get weather forecast for multiple days"""
    try:
        data = request.get_json()

        if not data:
            raise BadRequest("No JSON data provided")

        # Validate required fields
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        city = data.get('city')

        if not all([start_date, end_date, city]):
            raise BadRequest("Missing required fields: start_date, end_date, city")

        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_dt > end_dt:
            raise BadRequest("start_date must be before end_date")

        # Limit forecast to 30 days
        if (end_dt - start_dt).days > 30:
            raise BadRequest("Forecast period cannot exceed 30 days")

        # Generate forecast
        forecasts = []
        current_date = start_dt

        while current_date <= end_dt:
            try:
                prediction = predictor.predict_weather(
                    current_date.strftime('%Y-%m-%d'),
                    city,
                    data.get('latitude'),
                    data.get('longitude'),
                    data.get('elevation')
                )
                forecasts.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting for {current_date}: {str(e)}")

            current_date += timedelta(days=1)

        return jsonify({
            'success': True,
            'data': {
                'city': city,
                'start_date': start_date,
                'end_date': end_date,
                'forecasts': forecasts
            },
            'timestamp': datetime.now().isoformat()
        })

    except BadRequest as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load models on startup
    try:
        predictor.load_models()
        logger.info("Models loaded successfully on startup")
        app.models_loaded = True
    except Exception as e:
        logger.warning(f"Models not loaded on startup: {str(e)}")
        logger.info("Models will be loaded on first request")
        app.models_loaded = False

    app.run(debug=True, host='0.0.0.0', port=5000)

# -----------------------------------------------------------------------------------

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from datetime import datetime, timedelta
# import os
# from werkzeug.exceptions import BadRequest
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend


# class WeatherPredictor:
#     def __init__(self):
#         self.models = {}
#         self.scalers = {}
#         self.label_encoders = {}
#         self.feature_columns = []
#         self.target_columns = []
#         self.model_weights = {}
#         self.loaded = False

#     def load_models(self):
#         """Load all trained models and preprocessors"""
#         try:
#             logger.info("Loading models...")

#             # Load sklearn models
#             self.models["xgboost"] = joblib.load(
#                 "models/sri_lanka_weather_model_xgboost.pkl"
#             )
#             self.models["lightgbm"] = joblib.load(
#                 "models/sri_lanka_weather_model_lightgbm.pkl"
#             )
#             self.models["random_forest"] = joblib.load(
#                 "models/sri_lanka_weather_model_random_forest.pkl"
#             )

#             # Load neural network
#             self.models["neural_network"] = tf.keras.models.load_model(
#                 "models/sri_lanka_weather_model_neural_network.h5"
#             )

#             # Load scalers and encoders
#             self.scalers = joblib.load("models/sri_lanka_weather_model_scalers.pkl")
#             self.label_encoders = joblib.load(
#                 "models/sri_lanka_weather_model_label_encoders.pkl"
#             )
#             self.feature_columns = joblib.load(
#                 "models/sri_lanka_weather_model_feature_columns.pkl"
#             )
#             self.target_columns = joblib.load(
#                 "models/sri_lanka_weather_model_target_columns.pkl"
#             )
#             self.model_weights = joblib.load(
#                 "models/sri_lanka_weather_model_weights.pkl"
#             )

#             self.loaded = True
#             logger.info("Models loaded successfully!")

#         except Exception as e:
#             logger.error(f"Error loading models: {str(e)}")
#             raise

#     def _get_season(self, month):
#         """Get season based on month (Sri Lankan climate)"""
#         if month in [12, 1, 2]:
#             return "NE_Monsoon"
#         elif month in [3, 4, 5]:
#             return "Inter_Monsoon_1"
#         elif month in [6, 7, 8, 9]:
#             return "SW_Monsoon"
#         else:
#             return "Inter_Monsoon_2"

#     def predict_weather(
#         self, date, city, latitude=None, longitude=None, elevation=None
#     ):
#         """Predict weather for a specific date and city"""
#         if not self.loaded:
#             raise Exception("Models not loaded")

#         try:
#             date_obj = pd.to_datetime(date)

#             # Default coordinates for major Sri Lankan cities
#             city_coordinates = {
#                 "Colombo": {"lat": 6.9271, "lon": 79.8612, "elev": 8},
#                 "Kandy": {"lat": 7.2906, "lon": 80.6337, "elev": 488},
#                 "Galle": {"lat": 6.0535, "lon": 80.2210, "elev": 13},
#                 "Jaffna": {"lat": 9.6615, "lon": 80.0255, "elev": 9},
#                 "Trincomalee": {"lat": 8.5874, "lon": 81.2152, "elev": 6},
#                 "Anuradhapura": {"lat": 8.3114, "lon": 80.4037, "elev": 81},
#                 "Batticaloa": {"lat": 7.7170, "lon": 81.7000, "elev": 2},
#                 "Negombo": {"lat": 7.2084, "lon": 79.8358, "elev": 6},
#                 "Matara": {"lat": 5.9549, "lon": 80.5550, "elev": 4},
#                 "Ratnapura": {"lat": 6.6828, "lon": 80.4014, "elev": 34},
#             }

#             # Use provided coordinates or defaults
#             if city in city_coordinates:
#                 coords = city_coordinates[city]
#                 lat = latitude or coords["lat"]
#                 lon = longitude or coords["lon"]
#                 elev = elevation or coords["elev"]
#             else:
#                 lat = latitude or 7.8731  # Default to center of Sri Lanka
#                 lon = longitude or 80.7718
#                 elev = elevation or 50

#             # Get city encoding
#             if city in self.label_encoders["city"].classes_:
#                 city_encoded = self.label_encoders["city"].transform([city])[0]
#             else:
#                 city_encoded = 0  # Default encoding for unknown cities

#             # Create feature vector
#             features = {
#                 "weathercode": 1,
#                 "latitude": lat,
#                 "longitude": lon,
#                 "elevation": elev,
#                 "year": date_obj.year,
#                 "month": date_obj.month,
#                 "day": date_obj.day,
#                 "day_of_year": date_obj.dayofyear,
#                 "week_of_year": date_obj.isocalendar().week,
#                 "month_sin": np.sin(2 * np.pi * date_obj.month / 12),
#                 "month_cos": np.cos(2 * np.pi * date_obj.month / 12),
#                 "day_sin": np.sin(2 * np.pi * date_obj.dayofyear / 365),
#                 "day_cos": np.cos(2 * np.pi * date_obj.dayofyear / 365),
#                 "city_encoded": city_encoded,
#                 "country_encoded": 0,
#                 "daylight_hours": 12,
#                 "winddirection_10m_dominant": 180,
#             }

#             # Add seasonal features
#             season = self._get_season(date_obj.month)
#             for s in ["NE_Monsoon", "Inter_Monsoon_1", "SW_Monsoon", "Inter_Monsoon_2"]:
#                 features[f"season_{s}"] = 1 if season == s else 0

#             # Add lag features with seasonal defaults
#             seasonal_temp = {
#                 "NE_Monsoon": 25.0,
#                 "Inter_Monsoon_1": 28.0,
#                 "SW_Monsoon": 26.0,
#                 "Inter_Monsoon_2": 27.0,
#             }
#             seasonal_rain = {
#                 "NE_Monsoon": 5.0,
#                 "Inter_Monsoon_1": 8.0,
#                 "SW_Monsoon": 12.0,
#                 "Inter_Monsoon_2": 3.0,
#             }

#             lag_defaults = {
#                 "temperature_2m_mean_lag1": seasonal_temp[season],
#                 "temperature_2m_mean_lag7": seasonal_temp[season],
#                 "temperature_2m_mean_ma7": seasonal_temp[season],
#                 "precipitation_sum_lag1": seasonal_rain[season],
#                 "precipitation_sum_lag7": seasonal_rain[season],
#                 "precipitation_sum_ma7": seasonal_rain[season],
#                 "windspeed_10m_max_lag1": 12.0,
#                 "windspeed_10m_max_lag7": 12.0,
#                 "windspeed_10m_max_ma7": 12.0,
#             }
#             features.update(lag_defaults)

#             # Create feature vector in correct order
#             feature_vector = np.array(
#                 [features[col] for col in self.feature_columns]
#             ).reshape(1, -1)

#             # Scale features
#             feature_vector_scaled = self.scalers["feature_scaler"].transform(
#                 feature_vector
#             )

#             # Make ensemble prediction
#             predictions = {}
#             for model_name, model in self.models.items():
#                 if model_name == "neural_network":
#                     pred_scaled = model.predict(feature_vector_scaled, verbose=0)
#                 else:
#                     pred_scaled = model.predict(feature_vector_scaled)

#                 pred = self.scalers["target_scaler"].inverse_transform(pred_scaled)
#                 predictions[model_name] = pred[0]

#             # Calculate ensemble prediction
#             ensemble_pred = np.zeros(len(self.target_columns))
#             for model_name, weight in self.model_weights.items():
#                 ensemble_pred += weight * predictions[model_name]

#             # Create result dictionary
#             result = {}
#             for i, col in enumerate(self.target_columns):
#                 result[col] = float(ensemble_pred[i])

#             # Add weather interpretation
#             result["weather_condition"] = self._interpret_weather(result)
#             result["season"] = season
#             result["city"] = city
#             result["date"] = date
#             result["coordinates"] = {
#                 "latitude": lat,
#                 "longitude": lon,
#                 "elevation": elev,
#             }

#             return result

#         except Exception as e:
#             logger.error(f"Prediction error: {str(e)}")
#             raise

#     def _interpret_weather(self, weather_data):
#         """Interpret weather data into human-readable conditions"""
#         temp_max = weather_data["temperature_2m_max"]
#         temp_min = weather_data["temperature_2m_min"]
#         precipitation = weather_data["precipitation_sum"]
#         wind_speed = weather_data["windspeed_10m_max"]

#         # Temperature interpretation
#         if temp_max > 35:
#             temp_desc = "Very Hot"
#         elif temp_max > 30:
#             temp_desc = "Hot"
#         elif temp_max > 25:
#             temp_desc = "Warm"
#         elif temp_max > 20:
#             temp_desc = "Mild"
#         else:
#             temp_desc = "Cool"

#         # Precipitation interpretation
#         if precipitation > 50:
#             rain_desc = "Heavy Rain"
#         elif precipitation > 20:
#             rain_desc = "Moderate Rain"
#         elif precipitation > 5:
#             rain_desc = "Light Rain"
#         elif precipitation > 0.1:
#             rain_desc = "Drizzle"
#         else:
#             rain_desc = "No Rain"

#         # Wind interpretation
#         if wind_speed > 40:
#             wind_desc = "Very Windy"
#         elif wind_speed > 25:
#             wind_desc = "Windy"
#         elif wind_speed > 15:
#             wind_desc = "Breezy"
#         else:
#             wind_desc = "Calm"

#         return {
#             "temperature": temp_desc,
#             "precipitation": rain_desc,
#             "wind": wind_desc,
#             "summary": f"{temp_desc} with {rain_desc.lower()}, {wind_desc.lower()} conditions",
#         }


# # Initialize predictor
# predictor = WeatherPredictor()


# @app.before_request
# def load_models():
#     """Load models when the app starts"""
#     if not hasattr(app, "models_loaded"):
#         try:
#             predictor.load_models()
#             app.models_loaded = True
#             logger.info("Models loaded successfully on first request")
#         except Exception as e:
#             app.models_loaded = False
#             logger.error(f"Error loading models on first request: {str(e)}")
#             raise


# @app.route("/health", methods=["GET"])
# def health_check():
#     """Health check endpoint"""
#     return jsonify(
#         {
#             "status": "healthy",
#             "models_loaded": predictor.loaded,
#             "timestamp": datetime.now().isoformat(),
#         }
#     )


# @app.route("/predict", methods=["POST"])
# def predict_weather():
#     """Main prediction endpoint"""
#     try:
#         data = request.get_json()

#         if not data:
#             raise BadRequest("No JSON data provided")

#         # Validate required fields
#         required_fields = ["date", "city"]
#         for field in required_fields:
#             if field not in data:
#                 raise BadRequest(f"Missing required field: {field}")

#         # Extract parameters
#         date = data["date"]
#         city = data["city"]
#         latitude = data.get("latitude")
#         longitude = data.get("longitude")
#         elevation = data.get("elevation")

#         # Validate date format
#         try:
#             datetime.strptime(date, "%Y-%m-%d")
#         except ValueError:
#             raise BadRequest("Invalid date format. Use YYYY-MM-DD")

#         # Make prediction
#         prediction = predictor.predict_weather(
#             date, city, latitude, longitude, elevation
#         )

#         return jsonify(
#             {
#                 "success": True,
#                 "data": prediction,
#                 "timestamp": datetime.now().isoformat(),
#             }
#         )

#     except BadRequest as e:
#         return jsonify(
#             {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
#         ), 400

#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return jsonify(
#             {
#                 "success": False,
#                 "error": "Internal server error",
#                 "timestamp": datetime.now().isoformat(),
#             }
#         ), 500


# @app.route("/predict/batch", methods=["POST"])
# def predict_batch():
#     """Batch prediction endpoint for multiple dates/cities"""
#     try:
#         data = request.get_json()

#         if not data or "predictions" not in data:
#             raise BadRequest("No prediction data provided")

#         predictions = []
#         errors = []

#         for i, pred_data in enumerate(data["predictions"]):
#             try:
#                 # Validate each prediction request
#                 if "date" not in pred_data or "city" not in pred_data:
#                     errors.append(f"Request {i}: Missing required fields")
#                     continue

#                 prediction = predictor.predict_weather(
#                     pred_data["date"],
#                     pred_data["city"],
#                     pred_data.get("latitude"),
#                     pred_data.get("longitude"),
#                     pred_data.get("elevation"),
#                 )
#                 predictions.append(prediction)

#             except Exception as e:
#                 errors.append(f"Request {i}: {str(e)}")

#         return jsonify(
#             {
#                 "success": True,
#                 "data": predictions,
#                 "errors": errors,
#                 "timestamp": datetime.now().isoformat(),
#             }
#         )

#     except Exception as e:
#         logger.error(f"Batch prediction error: {str(e)}")
#         return jsonify(
#             {
#                 "success": False,
#                 "error": "Internal server error",
#                 "timestamp": datetime.now().isoformat(),
#             }
#         ), 500


# @app.route("/cities", methods=["GET"])
# def get_cities():
#     """Get list of supported cities"""
#     cities = [
#         {"name": "Colombo", "latitude": 6.9271, "longitude": 79.8612, "elevation": 8},
#         {"name": "Kandy", "latitude": 7.2906, "longitude": 80.6337, "elevation": 488},
#         {"name": "Galle", "latitude": 6.0535, "longitude": 80.2210, "elevation": 13},
#         {"name": "Jaffna", "latitude": 9.6615, "longitude": 80.0255, "elevation": 9},
#         {
#             "name": "Trincomalee",
#             "latitude": 8.5874,
#             "longitude": 81.2152,
#             "elevation": 6,
#         },
#         {
#             "name": "Anuradhapura",
#             "latitude": 8.3114,
#             "longitude": 80.4037,
#             "elevation": 81,
#         },
#         {
#             "name": "Batticaloa",
#             "latitude": 7.7170,
#             "longitude": 81.7000,
#             "elevation": 2,
#         },
#         {"name": "Negombo", "latitude": 7.2084, "longitude": 79.8358, "elevation": 6},
#         {"name": "Matara", "latitude": 5.9549, "longitude": 80.5550, "elevation": 4},
#         {
#             "name": "Ratnapura",
#             "latitude": 6.6828,
#             "longitude": 80.4014,
#             "elevation": 34,
#         },
#     ]

#     return jsonify(
#         {"success": True, "data": cities, "timestamp": datetime.now().isoformat()}
#     )


# @app.route("/forecast", methods=["POST"])
# def get_forecast():
#     """Get weather forecast for multiple days"""
#     try:
#         data = request.get_json()

#         if not data:
#             raise BadRequest("No JSON data provided")

#         # Validate required fields
#         start_date = data.get("start_date")
#         end_date = data.get("end_date")
#         city = data.get("city")

#         if not all([start_date, end_date, city]):
#             raise BadRequest("Missing required fields: start_date, end_date, city")

#         # Parse dates
#         start_dt = datetime.strptime(start_date, "%Y-%m-%d")
#         end_dt = datetime.strptime(end_date, "%Y-%m-%d")

#         if start_dt > end_dt:
#             raise BadRequest("start_date must be before end_date")

#         # Limit forecast to 30 days
#         if (end_dt - start_dt).days > 30:
#             raise BadRequest("Forecast period cannot exceed 30 days")

#         # Generate forecast
#         forecasts = []
#         current_date = start_dt

#         while current_date <= end_dt:
#             try:
#                 prediction = predictor.predict_weather(
#                     current_date.strftime("%Y-%m-%d"),
#                     city,
#                     data.get("latitude"),
#                     data.get("longitude"),
#                     data.get("elevation"),
#                 )
#                 forecasts.append(prediction)
#             except Exception as e:
#                 logger.error(f"Error predicting for {current_date}: {str(e)}")

#             current_date += timedelta(days=1)

#         return jsonify(
#             {
#                 "success": True,
#                 "data": {
#                     "city": city,
#                     "start_date": start_date,
#                     "end_date": end_date,
#                     "forecasts": forecasts,
#                 },
#                 "timestamp": datetime.now().isoformat(),
#             }
#         )

#     except BadRequest as e:
#         return jsonify(
#             {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
#         ), 400

#     except Exception as e:
#         logger.error(f"Forecast error: {str(e)}")
#         return jsonify(
#             {
#                 "success": False,
#                 "error": "Internal server error",
#                 "timestamp": datetime.now().isoformat(),
#             }
#         ), 500


# if __name__ == "__main__":
#     # Create models directory if it doesn't exist
#     os.makedirs("models", exist_ok=True)

#     # Load models on startup
#     try:
#         predictor.load_models()
#         logger.info("Models loaded successfully on startup")
#         app.models_loaded = True
#     except Exception as e:
#         logger.warning(f"Models not loaded on startup: {str(e)}")
#         logger.info("Models will be loaded on first request")
#         app.models_loaded = False

#     app.run(debug=True, host="0.0.0.0", port=5000)

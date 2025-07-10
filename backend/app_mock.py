from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class MockWeatherPredictor:
    def __init__(self):
        self.loaded = True
        logger.info("Mock weather predictor initialized")

    def load_models(self):
        """Pretend to load models"""
        logger.info("Mock loading models...")
        self.loaded = True
        logger.info("Mock models loaded successfully!")

    def predict_weather(self, date, city, latitude=None, longitude=None, elevation=None):
        """Return mock weather data"""
        logger.info(f"Predicting weather for {city} on {date}")
        
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month = date_obj.month
        
        # Seasonal variations
        if month in [12, 1, 2]:  # NE Monsoon
            season = 'NE_Monsoon'
            temp_base = 25
            rain_base = 5
        elif month in [3, 4, 5]:  # Inter Monsoon 1
            season = 'Inter_Monsoon_1'
            temp_base = 28
            rain_base = 8
        elif month in [6, 7, 8, 9]:  # SW Monsoon
            season = 'SW_Monsoon'
            temp_base = 26
            rain_base = 12
        else:  # Inter Monsoon 2
            season = 'Inter_Monsoon_2'
            temp_base = 27
            rain_base = 3
        
        # City-based variations
        city_temp_adjustments = {
            'Colombo': 1.5,
            'Kandy': -0.5,
            'Galle': 0.5,
            'Jaffna': 1.0,
            'Trincomalee': 1.2,
            'Anuradhapura': 2.0,
            'Batticaloa': 0.8,
            'Negombo': 0.7,
            'Matara': 0.3,
            'Ratnapura': -0.2
        }
        
        city_rain_adjustments = {
            'Colombo': 1.2,
            'Kandy': 1.5,
            'Galle': 1.8,
            'Jaffna': 0.6,
            'Trincomalee': 0.9,
            'Anuradhapura': 0.7,
            'Batticaloa': 1.3,
            'Negombo': 1.1,
            'Matara': 1.6,
            'Ratnapura': 2.0
        }
        
        # Apply city adjustments with defaults
        temp_adj = city_temp_adjustments.get(city, 0)
        rain_adj = city_rain_adjustments.get(city, 1.0)
        
        # Random variations
        temp_random = random.uniform(-1.5, 1.5)
        rain_random = random.uniform(0.8, 1.2)
        wind_random = random.uniform(5, 15)
        
        # Calculate values
        temp_mean = temp_base + temp_adj + temp_random
        temp_min = temp_mean - random.uniform(2, 4)
        temp_max = temp_mean + random.uniform(2, 4)
        precipitation = rain_base * rain_adj * rain_random
        wind_speed = wind_random + (precipitation / 5)  # Wind correlates somewhat with rain
        
        # Create result
        result = {
            'temperature_2m_mean': round(temp_mean, 1),
            'temperature_2m_min': round(temp_min, 1),
            'temperature_2m_max': round(temp_max, 1),
            'precipitation_sum': round(precipitation, 1),
            'windspeed_10m_max': round(wind_speed, 1),
            'weather_condition': self._interpret_weather({
                'temperature_2m_max': temp_max,
                'temperature_2m_min': temp_min,
                'precipitation_sum': precipitation,
                'windspeed_10m_max': wind_speed
            }),
            'season': season,
            'city': city,
            'date': date,
            'coordinates': {
                'latitude': latitude, 
                'longitude': longitude, 
                'elevation': elevation
            }
        }
        
        return result

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
predictor = MockWeatherPredictor()

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
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Validate required fields
        required_fields = ['date', 'city']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}',
                    'timestamp': datetime.now().isoformat()
                }), 400

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
            return jsonify({
                'success': False,
                'error': 'Invalid date format. Use YYYY-MM-DD',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Make prediction
        prediction = predictor.predict_weather(date, city, latitude, longitude, elevation)

        return jsonify({
            'success': True,
            'data': prediction,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
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
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Validate required fields
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        city = data.get('city')

        if not all([start_date, end_date, city]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: start_date, end_date, city',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_dt > end_dt:
            return jsonify({
                'success': False,
                'error': 'start_date must be before end_date',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Limit forecast to 30 days
        if (end_dt - start_dt).days > 30:
            return jsonify({
                'success': False,
                'error': 'Forecast period cannot exceed 30 days',
                'timestamp': datetime.now().isoformat()
            }), 400

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

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

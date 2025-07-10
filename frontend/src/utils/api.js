// src/services/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class WeatherAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'API request failed');
      }

      return data;
    } catch (error) {
      console.error('API Request Error:', error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }

  // Get supported cities
  async getCities() {
    return this.request('/cities');
  }

  // Single weather prediction
  async predictWeather(predictionData) {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify(predictionData),
    });
  }

  // Multi-day forecast
  async getForecast(forecastData) {
    return this.request('/forecast', {
      method: 'POST',
      body: JSON.stringify(forecastData),
    });
  }

  // Batch predictions
  async predictBatch(predictions) {
    return this.request('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ predictions }),
    });
  }
}

// Create singleton instance
const weatherAPI = new WeatherAPI();

// Export individual methods for easier use
export const {
  healthCheck,
  getCities,
  predictWeather,
  getForecast,
  predictBatch,
} = weatherAPI;

export default weatherAPI;
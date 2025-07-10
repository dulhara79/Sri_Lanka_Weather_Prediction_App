// src/utils/weatherHelpers.js
import { Sun, Cloud, CloudRain, CloudSnow, Wind, Zap, Eye, AlertTriangle } from 'lucide-react';

/**
 * Format date for display
 */
export const formatDate = (dateString, options = {}) => {
  const defaultOptions = {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  };
  
  return new Date(dateString).toLocaleDateString('en-US', {
    ...defaultOptions,
    ...options,
  });
};

/**
 * Format time for display
 */
export const formatTime = (dateString) => {
  return new Date(dateString).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Get weather icon based on conditions
 */
export const getWeatherIcon = (weather, size = 8) => {
  const iconClass = `w-${size} h-${size}`;
  
  if (!weather) return <Cloud className={`${iconClass} text-gray-400`} />;
  
  const condition = weather.weather_condition || {};
  const temp = condition.temperature?.toLowerCase() || '';
  const rain = condition.precipitation?.toLowerCase() || '';
  const wind = condition.wind?.toLowerCase() || '';
  
  // Determine primary condition
  if (rain.includes('heavy')) {
    return <CloudRain className={`${iconClass} text-blue-600`} />;
  }
  
  if (rain.includes('rain') || rain.includes('drizzle')) {
    return <CloudRain className={`${iconClass} text-blue-500`} />;
  }
  
  if (wind.includes('very windy') || wind.includes('windy')) {
    return <Wind className={`${iconClass} text-gray-600`} />;
  }
  
  if (temp.includes('very hot') || temp.includes('hot')) {
    return <Sun className={`${iconClass} text-yellow-500`} />;
  }
  
  if (temp.includes('cool')) {
    return <Cloud className={`${iconClass} text-blue-400`} />;
  }
  
  return <Cloud className={`${iconClass} text-gray-500`} />;
};

/**
 * Get weather condition color scheme
 */
export const getWeatherColors = (weather) => {
  if (!weather?.weather_condition) {
    return {
      primary: 'gray',
      bg: 'gray-50',
      text: 'gray-600',
      border: 'gray-200',
    };
  }
  
  const condition = weather.weather_condition;
  const rain = condition.precipitation?.toLowerCase() || '';
  const temp = condition.temperature?.toLowerCase() || '';
  
  if (rain.includes('heavy')) {
    return {
      primary: 'blue',
      bg: 'blue-50',
      text: 'blue-700',
      border: 'blue-200',
    };
  }
  
  if (rain.includes('rain')) {
    return {
      primary: 'cyan',
      bg: 'cyan-50',
      text: 'cyan-700',
      border: 'cyan-200',
    };
  }
  
  if (temp.includes('hot')) {
    return {
      primary: 'orange',
      bg: 'orange-50',
      text: 'orange-700',
      border: 'orange-200',
    };
  }
  
  if (temp.includes('cool')) {
    return {
      primary: 'slate',
      bg: 'slate-50',
      text: 'slate-700',
      border: 'slate-200',
    };
  }
  
  return {
    primary: 'green',
    bg: 'green-50',
    text: 'green-700',
    border: 'green-200',
  };
};

/**
 * Get temperature description and color
 */
export const getTemperatureInfo = (temp) => {
  if (temp > 35) {
    return { desc: 'Very Hot', color: 'red-600', bg: 'red-50' };
  } else if (temp > 30) {
    return { desc: 'Hot', color: 'orange-600', bg: 'orange-50' };
  } else if (temp > 25) {
    return { desc: 'Warm', color: 'yellow-600', bg: 'yellow-50' };
  } else if (temp > 20) {
    return { desc: 'Mild', color: 'green-600', bg: 'green-50' };
  } else {
    return { desc: 'Cool', color: 'blue-600', bg: 'blue-50' };
  }
};

/**
 * Get precipitation description and color
 */
export const getPrecipitationInfo = (precipitation) => {
  if (precipitation > 50) {
    return { desc: 'Heavy Rain', color: 'blue-700', bg: 'blue-100' };
  } else if (precipitation > 20) {
    return { desc: 'Moderate Rain', color: 'blue-600', bg: 'blue-50' };
  } else if (precipitation > 5) {
    return { desc: 'Light Rain', color: 'cyan-600', bg: 'cyan-50' };
  } else if (precipitation > 0.1) {
    return { desc: 'Drizzle', color: 'slate-600', bg: 'slate-50' };
  } else {
    return { desc: 'No Rain', color: 'gray-500', bg: 'gray-50' };
  }
};

/**
 * Get wind speed description and color
 */
export const getWindInfo = (windSpeed) => {
  if (windSpeed > 40) {
    return { desc: 'Very Windy', color: 'purple-700', bg: 'purple-100' };
  } else if (windSpeed > 25) {
    return { desc: 'Windy', color: 'purple-600', bg: 'purple-50' };
  } else if (windSpeed > 15) {
    return { desc: 'Breezy', color: 'indigo-600', bg: 'indigo-50' };
  } else {
    return { desc: 'Calm', color: 'green-600', bg: 'green-50' };
  }
};

/**
 * Calculate feels like temperature
 */
export const calculateFeelsLike = (temp, humidity, windSpeed) => {
  // Simplified heat index calculation
  if (temp < 27) return temp;
  
  const T = temp;
  const RH = humidity;
  
  let HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094));
  
  if (HI >= 80) {
    HI = -42.379 + 2.04901523 * T + 10.14333127 * RH 
        - 0.22475541 * T * RH - 0.00683783 * T * T 
        - 0.05481717 * RH * RH + 0.00122874 * T * T * RH 
        + 0.00085282 * T * RH * RH - 0.00000199 * T * T * RH * RH;
  }
  
  return Math.round(HI * 10) / 10;
};

/**
 * Get season information
 */
export const getSeasonInfo = (season) => {
  const seasons = {
    'NE_Monsoon': {
      name: 'Northeast Monsoon',
      months: 'Dec - Feb',
      description: 'Dry season with occasional rains',
      color: 'blue-600',
      bg: 'blue-50',
    },
    'Inter_Monsoon_1': {
      name: 'First Inter-Monsoon',
      months: 'Mar - May',
      description: 'Transitional period with thunderstorms',
      color: 'orange-600',
      bg: 'orange-50',
    },
    'SW_Monsoon': {
      name: 'Southwest Monsoon',
      months: 'Jun - Sep',
      description: 'Rainy season with heavy precipitation',
      color: 'green-600',
      bg: 'green-50',
    },
    'Inter_Monsoon_2': {
      name: 'Second Inter-Monsoon',
      months: 'Oct - Nov',
      description: 'Transitional period with moderate rains',
      color: 'purple-600',
      bg: 'purple-50',
    },
  };
  
  return seasons[season] || {
    name: 'Unknown Season',
    months: '',
    description: '',
    color: 'gray-600',
    bg: 'gray-50',
  };
};

/**
 * Validate date input
 */
export const validateDate = (dateString) => {
  const date = new Date(dateString);
  const today = new Date();
  const maxFuture = new Date();
  maxFuture.setDate(today.getDate() + 365); // 1 year in future
  
  return {
    isValid: !isNaN(date.getTime()),
    isPast: date < today,
    isTooFar: date > maxFuture,
    date,
  };
};

/**
 * Generate date range for forecasts
 */
export const generateDateRange = (startDate, endDate) => {
  const dates = [];
  const current = new Date(startDate);
  const end = new Date(endDate);
  
  while (current <= end) {
    dates.push(new Date(current));
    current.setDate(current.getDate() + 1);
  }
  
  return dates;
};

/**
 * Calculate weather trend
 */
export const calculateWeatherTrend = (forecasts) => {
  if (!forecasts || forecasts.length < 2) return null;
  
  const temps = forecasts.map(f => f.temperature_2m_max);
  const rains = forecasts.map(f => f.precipitation_sum);
  
  const tempTrend = temps[temps.length - 1] - temps[0];
  const rainTrend = rains.reduce((a, b) => a + b, 0) / rains.length;
  
  return {
    temperature: {
      change: tempTrend,
      trend: tempTrend > 2 ? 'rising' : tempTrend < -2 ? 'falling' : 'stable',
    },
    precipitation: {
      average: rainTrend,
      level: rainTrend > 20 ? 'high' : rainTrend > 5 ? 'moderate' : 'low',
    },
  };
};

/**
 * Get weather alerts
 */
export const getWeatherAlerts = (weather) => {
  const alerts = [];
  
  if (weather.temperature_2m_max > 35) {
    alerts.push({
      type: 'warning',
      message: 'Very hot weather expected. Stay hydrated and avoid prolonged sun exposure.',
      icon: <AlertTriangle className="w-4 h-4" />,
    });
  }
  
  if (weather.precipitation_sum > 50) {
    alerts.push({
      type: 'info',
      message: 'Heavy rain expected. Plan for potential flooding and transportation delays.',
      icon: <CloudRain className="w-4 h-4" />,
    });
  }
  
  if (weather.windspeed_10m_max > 40) {
    alerts.push({
      type: 'warning',
      message: 'Very windy conditions. Secure loose objects and be cautious outdoors.',
      icon: <Wind className="w-4 h-4" />,
    });
  }
  
  return alerts;
};

/**
 * Export all utility functions
 */
export default {
  formatDate,
  formatTime,
  getWeatherIcon,
  getWeatherColors,
  getTemperatureInfo,
  getPrecipitationInfo,
  getWindInfo,
  calculateFeelsLike,
  getSeasonInfo,
  validateDate,
  generateDateRange,
  calculateWeatherTrend,
  getWeatherAlerts,
};
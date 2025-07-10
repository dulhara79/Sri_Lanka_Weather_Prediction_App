import React, { useState, useEffect } from "react";
import {
  Calendar,
  MapPin,
  Cloud,
  Thermometer,
  Droplets,
  Wind,
  Sun,
  CloudRain,
  AlertCircle,
  TrendingUp,
  BarChart3,
  Clock,
} from "lucide-react";

const WeatherApp = () => {
  const [currentView, setCurrentView] = useState("single");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [cities, setCities] = useState([]);

  // Single prediction state
  const [singlePrediction, setSinglePrediction] = useState({
    date: "",
    city: "",
    latitude: "",
    longitude: "",
    elevation: "",
  });
  const [singleResult, setSingleResult] = useState(null);

  // Forecast state
  const [forecastData, setForecastData] = useState({
    startDate: "",
    endDate: "",
    city: "",
  });
  const [forecastResults, setForecastResults] = useState([]);

  // API base URL - adjust this to match your backend
  const API_BASE_URL = "http://localhost:5000";

  useEffect(() => {
    fetchCities();
    // Set default date to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const dateString = tomorrow.toISOString().split("T")[0];
    setSinglePrediction((prev) => ({ ...prev, date: dateString }));
    setForecastData((prev) => ({
      ...prev,
      startDate: dateString,
      endDate: new Date(tomorrow.getTime() + 7 * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0],
    }));
  }, []);

  const fetchCities = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/cities`);
      const data = await response.json();
      if (data.success) {
        setCities(data.data);
      }
    } catch (err) {
      console.error("Failed to fetch cities:", err);
    }
  };

  const handleCityChange = (cityName, type = "single") => {
    const selectedCity = cities.find((city) => city.name === cityName);
    if (selectedCity) {
      if (type === "single") {
        setSinglePrediction((prev) => ({
          ...prev,
          city: cityName,
          latitude: selectedCity.latitude,
          longitude: selectedCity.longitude,
          elevation: selectedCity.elevation,
        }));
      } else {
        setForecastData((prev) => ({ ...prev, city: cityName }));
      }
    }
  };

  // const predictSingleWeather = async () => {
  //   if (!singlePrediction.date || !singlePrediction.city) {
  //     setError('Please select both date and city');
  //     return;
  //   }

  //   setLoading(true);
  //   setError('');

  //   try {
  //     const response = await fetch(`${API_BASE_URL}/predict`, {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({
  //         date: singlePrediction.date,
  //         city: singlePrediction.city,
  //         latitude: singlePrediction.latitude || undefined,
  //         longitude: singlePrediction.longitude || undefined,
  //         elevation: singlePrediction.elevation || undefined
  //       }),
  //     });

  //     const data = await response.json();

  //     if (data.success) {
  //       setSingleResult(data.data);
  //     } else {
  //       setError(data.error || 'Prediction failed');
  //     }
  //   } catch (err) {
  //     setError('Failed to connect to the server');
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const predictSingleWeather = async () => {
    if (!singlePrediction.date || !singlePrediction.city) {
      setError("Please select both date and city");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          date: singlePrediction.date,
          city: singlePrediction.city,
          latitude: singlePrediction.latitude || undefined,
          longitude: singlePrediction.longitude || undefined,
          elevation: singlePrediction.elevation || undefined,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        // Check for HTTP errors (4xx/5xx)
        throw new Error(data.error || `Server error: ${response.status}`);
      }

      if (data.success) {
        setSingleResult(data.data);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch (err) {
      setError(err.message || "Failed to connect to the server");
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  const predictForecast = async () => {
    if (
      !forecastData.startDate ||
      !forecastData.endDate ||
      !forecastData.city
    ) {
      setError("Please fill in all forecast fields");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/forecast`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          start_date: forecastData.startDate,
          end_date: forecastData.endDate,
          city: forecastData.city,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setForecastResults(data.data.forecasts);
      } else {
        setError(data.error || "Forecast failed");
      }
    } catch (err) {
      setError("Failed to connect to the server");
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      weekday: "short",
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const getWeatherIcon = (condition) => {
    const temp = condition?.temperature?.toLowerCase() || "";
    const rain = condition?.precipitation?.toLowerCase() || "";

    if (rain.includes("heavy"))
      return <CloudRain className="w-8 h-8 text-blue-600" />;
    if (rain.includes("rain") || rain.includes("drizzle"))
      return <Cloud className="w-8 h-8 text-gray-600" />;
    if (temp.includes("hot"))
      return <Sun className="w-8 h-8 text-yellow-500" />;
    return <Cloud className="w-8 h-8 text-blue-400" />;
  };

  const WeatherCard = ({ weather, isDetailed = false }) => (
    <div
      className={`bg-white rounded-xl shadow-lg p-6 ${
        isDetailed ? "col-span-full" : ""
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">
            {weather.city} - {formatDate(weather.date)}
          </h3>
          <p className="text-sm text-gray-600">
            {weather.season?.replace("_", " ")}
          </p>
        </div>
        {getWeatherIcon(weather.weather_condition)}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-red-50 p-3 rounded-lg">
          <div className="flex items-center gap-2">
            <Thermometer className="w-4 h-4 text-red-500" />
            <span className="text-sm font-medium">Max Temp</span>
          </div>
          <p className="text-xl font-bold text-red-600">
            {weather.temperature_2m_max?.toFixed(1)}°C
          </p>
        </div>

        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="flex items-center gap-2">
            <Thermometer className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium">Min Temp</span>
          </div>
          <p className="text-xl font-bold text-blue-600">
            {weather.temperature_2m_min?.toFixed(1)}°C
          </p>
        </div>

        <div className="bg-green-50 p-3 rounded-lg">
          <div className="flex items-center gap-2">
            <Droplets className="w-4 h-4 text-green-500" />
            <span className="text-sm font-medium">Rainfall</span>
          </div>
          <p className="text-xl font-bold text-green-600">
            {weather.precipitation_sum?.toFixed(1)}mm
          </p>
        </div>

        <div className="bg-purple-50 p-3 rounded-lg">
          <div className="flex items-center gap-2">
            <Wind className="w-4 h-4 text-purple-500" />
            <span className="text-sm font-medium">Wind</span>
          </div>
          <p className="text-xl font-bold text-purple-600">
            {weather.windspeed_10m_max?.toFixed(1)} km/h
          </p>
        </div>
      </div>

      {isDetailed && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-orange-50 p-3 rounded-lg">
              <h4 className="font-medium text-orange-800 mb-2">
                Apparent Temperature
              </h4>
              <div className="space-y-1 text-sm">
                <p>Max: {weather.apparent_temperature_max?.toFixed(1)}°C</p>
                <p>Min: {weather.apparent_temperature_min?.toFixed(1)}°C</p>
                <p>Mean: {weather.apparent_temperature_mean?.toFixed(1)}°C</p>
              </div>
            </div>

            <div className="bg-cyan-50 p-3 rounded-lg">
              <h4 className="font-medium text-cyan-800 mb-2">
                Precipitation Details
              </h4>
              <div className="space-y-1 text-sm">
                <p>Total: {weather.precipitation_sum?.toFixed(1)}mm</p>
                <p>Rain: {weather.rain_sum?.toFixed(1)}mm</p>
                <p>Hours: {weather.precipitation_hours?.toFixed(1)}h</p>
              </div>
            </div>

            <div className="bg-yellow-50 p-3 rounded-lg">
              <h4 className="font-medium text-yellow-800 mb-2">
                Other Conditions
              </h4>
              <div className="space-y-1 text-sm">
                <p>Wind Gusts: {weather.windgusts_10m_max?.toFixed(1)} km/h</p>
                <p>
                  Solar Radiation: {weather.shortwave_radiation_sum?.toFixed(1)}{" "}
                  MJ/m²
                </p>
                <p>
                  Evapotranspiration:{" "}
                  {weather.et0_fao_evapotranspiration?.toFixed(1)}mm
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium text-gray-800 mb-2">Weather Summary</h4>
            <p className="text-gray-700">
              {weather.weather_condition?.summary}
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                {weather.weather_condition?.temperature}
              </span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">
                {weather.weather_condition?.precipitation}
              </span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">
                {weather.weather_condition?.wind}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Cloud className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Sri Lanka Weather Predictor
                </h1>
                <p className="text-gray-600">AI-Powered Weather Forecasting</p>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setCurrentView("single")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === "single"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                Single Day
              </button>
              <button
                onClick={() => setCurrentView("forecast")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === "forecast"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                Forecast
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-red-700">{error}</p>
            <button
              onClick={() => setError("")}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        )}

        {/* Single Day Prediction */}
        {currentView === "single" && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <Calendar className="w-6 h-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-800">
                  Single Day Weather Prediction
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Date
                  </label>
                  <input
                    type="date"
                    value={singlePrediction.date}
                    onChange={(e) =>
                      setSinglePrediction((prev) => ({
                        ...prev,
                        date: e.target.value,
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    City
                  </label>
                  <select
                    value={singlePrediction.city}
                    onChange={(e) => handleCityChange(e.target.value, "single")}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select a city</option>
                    {cities.map((city) => (
                      <option key={city.name} value={city.name}>
                        {city.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Latitude
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={singlePrediction.latitude}
                    onChange={(e) =>
                      setSinglePrediction((prev) => ({
                        ...prev,
                        latitude: e.target.value,
                      }))
                    }
                    placeholder="Auto-filled"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Longitude
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={singlePrediction.longitude}
                    onChange={(e) =>
                      setSinglePrediction((prev) => ({
                        ...prev,
                        longitude: e.target.value,
                      }))
                    }
                    placeholder="Auto-filled"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Elevation (m)
                  </label>
                  <input
                    type="number"
                    value={singlePrediction.elevation}
                    onChange={(e) =>
                      setSinglePrediction((prev) => ({
                        ...prev,
                        elevation: e.target.value,
                      }))
                    }
                    placeholder="Auto-filled"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <button
                onClick={predictSingleWeather}
                disabled={loading}
                className="w-full md:w-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
              >
                {loading ? "Predicting..." : "Predict Weather"}
              </button>
            </div>

            {/* Single Prediction Result */}
            {singleResult && (
              <WeatherCard weather={singleResult} isDetailed={true} />
            )}
          </div>
        )}

        {/* Multi-Day Forecast */}
        {currentView === "forecast" && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <BarChart3 className="w-6 h-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-800">
                  Multi-Day Weather Forecast
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={forecastData.startDate}
                    onChange={(e) =>
                      setForecastData((prev) => ({
                        ...prev,
                        startDate: e.target.value,
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={forecastData.endDate}
                    onChange={(e) =>
                      setForecastData((prev) => ({
                        ...prev,
                        endDate: e.target.value,
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    City
                  </label>
                  <select
                    value={forecastData.city}
                    onChange={(e) =>
                      handleCityChange(e.target.value, "forecast")
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select a city</option>
                    {cities.map((city) => (
                      <option key={city.name} value={city.name}>
                        {city.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={predictForecast}
                disabled={loading}
                className="w-full md:w-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
              >
                {loading ? "Generating Forecast..." : "Generate Forecast"}
              </button>
            </div>

            {/* Forecast Results */}
            {forecastResults.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Forecast Results ({forecastResults.length} days)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {forecastResults.map((forecast, index) => (
                    <WeatherCard key={index} weather={forecast} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="text-gray-600">Processing your request...</span>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p>
              © 2024 Sri Lanka Weather Predictor. Powered by AI and Machine
              Learning.
            </p>
            <p className="text-sm mt-1">
              Accurate weather predictions for 30+ cities across Sri Lanka
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default WeatherApp;

import axios from 'axios';
import authService from './authService';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = authService.getToken();
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      authService.logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

/**
 * API Service for communicating with Flask backend
 */
export const apiService = {
  /**
   * Health check endpoint
   */
  async checkHealth() {
    try {
      const response = await api.get('/api/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  },

  /**
   * Initialize models
   */
  async initializeModels() {
    try {
      const response = await api.post('/api/initialize');
      return response.data;
    } catch (error) {
      throw new Error(`Model initialization failed: ${error.message}`);
    }
  },

  /**
   * Get available years from dataset
   */
  async getAvailableYears() {
    try {
      const response = await api.get('/api/available-years');
      return response.data.years;
    } catch (error) {
      throw new Error(`Failed to fetch available years: ${error.message}`);
    }
  },

  /**
   * Get stock recommendation
   * @param {Object} params - { year, amount, days, months, years }
   */
  async getRecommendation(params) {
    try {
      const response = await api.post('/api/recommend', params);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.error || `Failed to get recommendation: ${error.message}`
      );
    }
  },

  /**
   * Get backtesting results
   * @param {Object} params - { ticker, start_date, end_date, num_stocks, predicted_price, purchase_price }
   */
  async getBacktestResults(params) {
    try {
      const response = await api.post('/api/backtest', params);
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.error || `Failed to get backtest results: ${error.message}`
      );
    }
  },
};

export default apiService;


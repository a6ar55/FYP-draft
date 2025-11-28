import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import authService from '../services/authService';
import apiService from '../services/api';

const loadingQuotes = [
  "Analyzing market trends...",
  "Crunching numbers with AI...",
  "Consulting our neural networks...",
  "Reading the market signals...",
  "Predicting future movements...",
  "Computing optimal strategy...",
  "Scanning 10 stocks...",
  "Deep learning in progress...",
  "Making intelligent decisions..."
];

const Dashboard = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [years, setYears] = useState([]);
  const [formData, setFormData] = useState({
    year: '',
    amount: '',
    days: 0,
    months: 0,
    years: 0
  });
  const [recommendation, setRecommendation] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingQuote, setLoadingQuote] = useState('');
  const [error, setError] = useState('');
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => {
    const currentUser = authService.getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
    } else {
      setUser({ name: 'Guest', email: 'demo@stockai.com' });
    }
    loadYears();
    setTimeout(() => setAnimateIn(true), 100);
  }, []);

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setLoadingQuote(loadingQuotes[Math.floor(Math.random() * loadingQuotes.length)]);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [loading]);

  const loadYears = async () => {
    try {
      const availableYears = await apiService.getAvailableYears();
      setYears(availableYears);
    } catch (err) {
      setError('Failed to load available years');
    }
  };

  const handleLogout = () => {
    authService.logout();
    navigate('/login');
  };

  const handleNextStep = () => {
    if (currentStep === 1 && !formData.year) {
      setError('Please select a year');
      return;
    }
    if (currentStep === 2 && !formData.amount) {
      setError('Please enter investment amount');
      return;
    }
    if (currentStep === 3 && (formData.days === 0 && formData.months === 0 && formData.years === 0)) {
      setError('Please enter holding period');
      return;
    }
    setError('');
    setAnimateIn(false);
    setTimeout(() => {
      setCurrentStep(currentStep + 1);
      setAnimateIn(true);
    }, 300);
  };

  const handlePrevStep = () => {
    setError('');
    setAnimateIn(false);
    setTimeout(() => {
      setCurrentStep(currentStep - 1);
      setAnimateIn(true);
    }, 300);
  };

  const handleGetRecommendation = async () => {
    setLoading(true);
    setError('');
    setLoadingQuote(loadingQuotes[0]);
    try {
      const result = await apiService.getRecommendation(formData);
      setRecommendation(result);
      
      try {
        const backtestData = await apiService.getBacktestResults({
          ticker: result.recommendation.ticker,
          start_date: result.dates.start_date,
          end_date: result.dates.end_date,
          num_stocks: result.investment.num_stocks,
          predicted_price: result.recommendation.predicted_price,
          purchase_price: result.recommendation.current_price
        });
        setBacktestResults(backtestData);
      } catch (backtestErr) {
        console.error('Backtest error:', backtestErr);
      }
      
      setAnimateIn(false);
      setTimeout(() => {
        setCurrentStep(4);
        setAnimateIn(true);
      }, 300);
    } catch (err) {
      setError(err.message || 'Failed to get recommendation');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setAnimateIn(false);
    setTimeout(() => {
      setCurrentStep(1);
      setFormData({ year: '', amount: '', days: 0, months: 0, years: 0 });
      setRecommendation(null);
      setBacktestResults(null);
      setError('');
      setAnimateIn(true);
    }, 300);
  };

  // Format date as YYYY-MM-DD
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <nav className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center shadow-lg">
                <span className="text-white font-bold text-xl">AI</span>
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                StockAI
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-gray-700 text-sm">
                Welcome, <strong className="text-blue-600">{user?.name}</strong>
              </span>
              {authService.isAuthenticated() && (
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all duration-200"
                >
                  Logout
                </button>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex justify-between items-center">
            {[1, 2, 3, 4].map((step) => (
              <div key={step} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold transition-all duration-500 ${
                  currentStep >= step 
                    ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white scale-110 shadow-lg' 
                    : 'bg-gray-200 text-gray-400'
                }`}>
                  {step}
                </div>
                {step < 4 && (
                  <div className={`w-full h-1 transition-all duration-500 ${
                    currentStep > step ? 'bg-gradient-to-r from-blue-500 to-indigo-600' : 'bg-gray-200'
                  }`} style={{ width: '80px' }} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between mt-2 text-gray-600 text-xs font-medium">
            <span>Year</span>
            <span>Amount</span>
            <span>Period</span>
            <span>Results</span>
          </div>
        </div>

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-white/95 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-6">
                <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-transparent border-t-blue-600 rounded-full animate-spin"></div>
                <div className="absolute inset-2 border-4 border-transparent border-t-indigo-600 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1s' }}></div>
              </div>
              <p className="text-gray-800 text-xl font-semibold animate-pulse">{loadingQuote}</p>
            </div>
          </div>
        )}

        {/* Content Card */}
        <div className={`bg-white rounded-3xl shadow-2xl p-8 border border-gray-100 transition-all duration-500 ${
          animateIn ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
        }`}>
          
          {error && (
            <div className="mb-6 bg-red-50 border-2 border-red-300 text-red-700 px-4 py-3 rounded-xl animate-shake">
              {error}
            </div>
          )}

          {/* Step 1: Year Selection */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-4xl font-bold text-gray-900 mb-2">Select Investment Year</h2>
                <p className="text-gray-600">Choose the year you want to invest in</p>
              </div>
              
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                {years.map(year => (
                  <button
                    key={year}
                    onClick={() => setFormData({ ...formData, year: year.toString() })}
                    className={`p-6 rounded-2xl font-bold text-2xl transition-all duration-300 transform hover:scale-105 border-2 ${
                      formData.year === year.toString()
                        ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-xl scale-105 border-transparent'
                        : 'bg-white text-gray-700 hover:bg-blue-50 border-gray-200 hover:border-blue-300'
                    }`}
                  >
                    {year}
                  </button>
                ))}
              </div>

              <button
                onClick={handleNextStep}
                disabled={!formData.year}
                className="w-full mt-8 bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-4 rounded-2xl font-bold text-lg shadow-lg hover:shadow-2xl hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-300"
              >
                Next Step →
              </button>
            </div>
          )}

          {/* Step 2: Investment Amount */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-4xl font-bold text-gray-900 mb-2">Investment Amount</h2>
                <p className="text-gray-600">How much do you want to invest?</p>
              </div>
              
              <div>
                <div className="relative">
                  <span className="absolute left-6 top-1/2 transform -translate-y-1/2 text-gray-600 text-2xl font-bold">₹</span>
                  <input
                    type="number"
                    value={formData.amount}
                    onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                    className="w-full pl-14 pr-6 py-6 bg-gray-50 border-2 border-gray-300 rounded-2xl text-gray-900 text-3xl font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all duration-300 placeholder-gray-400"
                    placeholder="100000"
                  />
                </div>
                <div className="mt-4 flex gap-2 flex-wrap">
                  {[50000, 100000, 500000, 1000000].map(amount => (
                    <button
                      key={amount}
                      onClick={() => setFormData({ ...formData, amount: amount.toString() })}
                      className="px-6 py-3 bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold rounded-xl transition-all duration-200 border border-blue-200"
                    >
                      ₹{(amount / 1000).toFixed(0)}K
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handlePrevStep}
                  className="flex-1 bg-gray-100 text-gray-700 py-4 rounded-2xl font-bold text-lg hover:bg-gray-200 transition-all duration-300"
                >
                  ← Back
                </button>
                <button
                  onClick={handleNextStep}
                  disabled={!formData.amount}
                  className="flex-1 bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-4 rounded-2xl font-bold text-lg shadow-lg hover:shadow-2xl hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-300"
                >
                  Next Step →
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Holding Period */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-4xl font-bold text-gray-900 mb-2">Holding Period</h2>
                <p className="text-gray-600">How long do you want to hold?</p>
              </div>
              
              <div className="grid grid-cols-3 gap-6">
                <div>
                  <label className="block text-gray-600 text-sm mb-2 font-medium">Days</label>
                  <input
                    type="number"
                    value={formData.days}
                    onChange={(e) => setFormData({ ...formData, days: parseInt(e.target.value) || 0 })}
                    className="w-full px-6 py-4 bg-gray-50 border-2 border-gray-300 rounded-xl text-gray-900 text-2xl font-bold text-center focus:outline-none focus:border-blue-500 focus:bg-white transition-all duration-300"
                    placeholder="0"
                  />
                </div>
                <div>
                  <label className="block text-gray-600 text-sm mb-2 font-medium">Months</label>
                  <input
                    type="number"
                    value={formData.months}
                    onChange={(e) => setFormData({ ...formData, months: parseInt(e.target.value) || 0 })}
                    className="w-full px-6 py-4 bg-gray-50 border-2 border-gray-300 rounded-xl text-gray-900 text-2xl font-bold text-center focus:outline-none focus:border-blue-500 focus:bg-white transition-all duration-300"
                    placeholder="0"
                  />
                </div>
                <div>
                  <label className="block text-gray-600 text-sm mb-2 font-medium">Years</label>
                  <input
                    type="number"
                    value={formData.years}
                    onChange={(e) => setFormData({ ...formData, years: parseInt(e.target.value) || 0 })}
                    className="w-full px-6 py-4 bg-gray-50 border-2 border-gray-300 rounded-xl text-gray-900 text-2xl font-bold text-center focus:outline-none focus:border-blue-500 focus:bg-white transition-all duration-300"
                    placeholder="0"
                  />
                </div>
              </div>

              <div className="mt-4 flex gap-2 flex-wrap justify-center">
                <button onClick={() => setFormData({ ...formData, months: 6, days: 0, years: 0 })} className="px-6 py-3 bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold rounded-xl transition-all duration-200 border border-blue-200">6 Months</button>
                <button onClick={() => setFormData({ ...formData, years: 1, months: 0, days: 0 })} className="px-6 py-3 bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold rounded-xl transition-all duration-200 border border-blue-200">1 Year</button>
                <button onClick={() => setFormData({ ...formData, years: 3, months: 0, days: 0 })} className="px-6 py-3 bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold rounded-xl transition-all duration-200 border border-blue-200">3 Years</button>
                <button onClick={() => setFormData({ ...formData, years: 5, months: 0, days: 0 })} className="px-6 py-3 bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold rounded-xl transition-all duration-200 border border-blue-200">5 Years</button>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handlePrevStep}
                  className="flex-1 bg-gray-100 text-gray-700 py-4 rounded-2xl font-bold text-lg hover:bg-gray-200 transition-all duration-300"
                >
                  ← Back
                </button>
                <button
                  onClick={handleGetRecommendation}
                  disabled={formData.days === 0 && formData.months === 0 && formData.years === 0}
                  className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 rounded-2xl font-bold text-lg shadow-lg hover:shadow-2xl hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-300"
                >
                  Get Prediction 🚀
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Results */}
          {currentStep === 4 && recommendation && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h2 className="text-4xl font-bold text-gray-900 mb-2">AI Prediction Results</h2>
                <p className="text-gray-600">Here's what our AI recommends</p>
              </div>

              {/* Recommendation Card */}
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300 rounded-2xl p-6 shadow-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-gray-900">📈 Best Stock</h3>
                  <span className="px-4 py-2 bg-green-500 text-white rounded-full font-bold text-sm shadow-md">RECOMMENDED</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm mb-1">Stock</p>
                    <p className="text-gray-900 font-bold text-xl">{recommendation.recommendation.ticker}</p>
                    <p className="text-gray-500 text-xs">{recommendation.recommendation.company}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm mb-1">Buy Price</p>
                    <p className="text-gray-900 font-bold text-xl">₹{recommendation.recommendation.current_price.toLocaleString()}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm mb-1">Target Price</p>
                    <p className="text-blue-600 font-bold text-xl">₹{recommendation.recommendation.predicted_price.toFixed(2)}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm mb-1">Expected</p>
                    <p className="text-green-600 font-bold text-xl">+{(recommendation.recommendation.expected_return * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>

              {/* Investment Details */}
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-6 shadow-lg">
                <h3 className="text-xl font-bold text-gray-900 mb-4">💰 Your Investment</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm">Stocks Bought</p>
                    <p className="font-bold text-2xl text-gray-900">{recommendation.investment.num_stocks}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm">Per Stock</p>
                    <p className="font-bold text-2xl text-gray-900">₹{recommendation.investment.price_per_stock.toLocaleString()}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm">Invested</p>
                    <p className="font-bold text-2xl text-gray-900">₹{recommendation.investment.actual_investment.toLocaleString()}</p>
                  </div>
                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-gray-600 text-sm">Cash Left</p>
                    <p className="font-bold text-2xl text-gray-900">₹{recommendation.investment.remaining_cash.toLocaleString()}</p>
                  </div>
                </div>
              </div>

              {/* Backtest Results */}
              {backtestResults && !backtestResults.error && (
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-300 rounded-2xl p-6 shadow-lg">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">🎯 What Actually Happened</h3>
                  
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-white rounded-xl p-4 text-center shadow-sm">
                      <p className="text-gray-600 text-sm">Bought At</p>
                      <p className="text-gray-900 font-bold text-xl">₹{backtestResults.prices.purchase_price.toLocaleString()}</p>
                    </div>
                    <div className="bg-white rounded-xl p-4 text-center shadow-sm">
                      <p className="text-gray-600 text-sm">AI Predicted</p>
                      <p className="text-blue-600 font-bold text-xl">₹{backtestResults.prices.predicted_price.toFixed(2)}</p>
                    </div>
                    <div className="bg-white rounded-xl p-4 text-center shadow-sm">
                      <p className="text-gray-600 text-sm">Actual Price</p>
                      <p className="text-green-600 font-bold text-xl">₹{backtestResults.prices.actual_price.toFixed(2)}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-white rounded-xl p-4 shadow-sm">
                      <p className="text-gray-600 text-sm mb-2 font-semibold">AI Prediction</p>
                      <p className="text-gray-700">Return: <span className="font-bold text-blue-600">{backtestResults.returns.predicted_return_pct > 0 ? '+' : ''}{backtestResults.returns.predicted_return_pct}%</span></p>
                      <p className="text-gray-700">Profit: <span className="font-bold">₹{backtestResults.returns.predicted_profit.toLocaleString()}</span></p>
                    </div>
                    <div className="bg-white rounded-xl p-4 shadow-sm">
                      <p className="text-gray-600 text-sm mb-2 font-semibold">Actual Result</p>
                      <p className="text-gray-700">Return: <span className="font-bold text-green-600">{backtestResults.returns.actual_return_pct > 0 ? '+' : ''}{backtestResults.returns.actual_return_pct}%</span></p>
                      <p className="text-gray-700">Profit: <span className="font-bold">₹{backtestResults.returns.actual_profit.toLocaleString()}</span></p>
                    </div>
                  </div>

                  <div className="bg-white rounded-xl p-4 shadow-sm">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <p className="text-gray-600 text-sm">Model Accuracy</p>
                        <p className="text-gray-900 font-bold text-2xl">{(100 - backtestResults.accuracy.price_error_pct).toFixed(1)}%</p>
                        <p className="text-gray-500 text-xs">Error: ₹{backtestResults.accuracy.price_error.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-gray-600 text-sm">Direction</p>
                        <p className={`font-bold text-xl ${backtestResults.accuracy.direction_correct ? 'text-green-600' : 'text-red-600'}`}>
                          {backtestResults.accuracy.direction_correct ? '✓ Correct' : '✗ Wrong'}
                        </p>
                        <p className="text-gray-500 text-xs">
                          {backtestResults.accuracy.predicted_direction} → {backtestResults.accuracy.actual_direction}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-600 text-sm">Period</p>
                        <p className="text-gray-900 font-bold">{recommendation.dates.period_days} days</p>
                        <p className="text-gray-500 text-xs">
                          {formatDate(recommendation.dates.start_date)} → {formatDate(recommendation.dates.end_date)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={handleReset}
                className="w-full bg-gradient-to-r from-gray-600 to-gray-700 text-white py-4 rounded-2xl font-bold text-lg shadow-lg hover:shadow-2xl hover:scale-105 transition-all duration-300"
              >
                Make Another Prediction
              </button>
            </div>
          )}
        </div>
      </main>

      <style jsx>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-10px); }
          75% { transform: translateX(10px); }
        }
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }
      `}</style>
    </div>
  );
};

export default Dashboard;

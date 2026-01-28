const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const authRoutes = require('./routes/auth');
const { protect } = require('./middleware/auth');

const execAsync = promisify(exec);
const app = express();
const PORT = process.env.PORT || 5001;
const PROJECT_ROOT = path.join(__dirname, '..');

// Enable CORS for all routes - most permissive configuration
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
  credentials: true
}));

// Parse JSON bodies
app.use(express.json());

// Global cache
let modelsLoaded = false;
let combinedData = null;

/**
 * Check if models exist
 */
function checkModelsExist() {
  const modelsDir = path.join(PROJECT_ROOT, 'models');
  if (!fs.existsSync(modelsDir)) {
    return false;
  }
  
  const modelFiles = fs.readdirSync(modelsDir).filter(f => f.endsWith('.keras'));
  return modelFiles.length > 0;
}

/**
 * Initialize models by calling Python script
 */
async function initializeModels() {
  try {
    // First check if models exist
    if (!checkModelsExist()) {
      return {
        status: 'error',
        message: 'Models not found. Please train models first by running: python3 model.py',
        models_loaded: 0
      };
    }

    const initScript = `
import sys
import os
import json
sys.path.insert(0, '${PROJECT_ROOT}')
from backend.api import initialize_models

try:
    portfolio_manager, combined_data = initialize_models()
    if portfolio_manager and portfolio_manager.models:
        result = {
            "status": "success",
            "models_loaded": len(portfolio_manager.models),
            "tickers": list(portfolio_manager.tickers)
        }
    else:
        result = {
            "status": "error",
            "message": "Failed to load models",
            "models_loaded": 0
        }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e), "models_loaded": 0}))
`;

    const tempScript = path.join(__dirname, `temp_init_${Date.now()}.py`);
    fs.writeFileSync(tempScript, initScript);

    const { stdout, stderr } = await execAsync(`cd "${PROJECT_ROOT}" && source venv/bin/activate && python3 "${tempScript}"`, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000
    });

    // Clean up temp file
    if (fs.existsSync(tempScript)) {
      fs.unlinkSync(tempScript);
    }

    // Parse last line of output (in case there are warnings)
    const lines = stdout.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    const result = JSON.parse(lastLine);
    
    if (result.status === 'success') {
      modelsLoaded = true;
    }
    
    return result;
  } catch (error) {
    console.error('Error initializing models:', error.message);
    return {
      status: 'error',
      message: error.message,
      models_loaded: 0
    };
  }
}

/**
 * Load combined data to get available years
 */
function loadCombinedData() {
  return new Promise((resolve, reject) => {
    const dataPath = path.join(PROJECT_ROOT, 'combined_data.csv');
    
    if (!fs.existsSync(dataPath)) {
      reject(new Error('Data file not found. Please run data_processor.py first.'));
      return;
    }

    if (combinedData) {
      const years = new Set();
      combinedData.forEach(row => {
        if (row.Date) {
          const date = new Date(row.Date);
          if (!isNaN(date.getTime())) {
            years.add(date.getFullYear());
          }
        }
      });
      resolve(Array.from(years).sort((a, b) => a - b));
      return;
    }

    const years = new Set();
    const data = [];

    fs.createReadStream(dataPath)
      .pipe(csv())
      .on('data', (row) => {
        data.push(row);
        if (row.Date) {
          const date = new Date(row.Date);
          if (!isNaN(date.getTime())) {
            years.add(date.getFullYear());
          }
        }
      })
      .on('end', () => {
        combinedData = data;
        // Remove 2014 from available years
        const filteredYears = Array.from(years).filter(y => y >= 2015).sort((a, b) => a - b);
        resolve(filteredYears);
      })
      .on('error', (err) => {
        reject(err);
      });
  });
}

/**
 * Get stock recommendation by calling Python
 */
async function getStockRecommendation(year, amount, days, months, years) {
  try {
    const script = `
import sys
import os
import json
from datetime import datetime, timedelta
sys.path.insert(0, '${PROJECT_ROOT}')
from backend.api import initialize_models, get_stock_recommendation

try:
    # Initialize models first (loads pre-trained weights)
    initialize_models()
    
    target_year = ${year}
    total_days = ${days} + (${months} * 30) + (${years} * 365)
    target_date = datetime(target_year, 1, 1)
    
    recommendation, error = get_stock_recommendation(target_date, total_days)
    
    if error:
        print(json.dumps({"error": error}))
    else:
        investment_amount = ${amount}
        current_price = recommendation['current_price']
        num_stocks = int(investment_amount / current_price)
        actual_investment = num_stocks * current_price
        remaining_cash = investment_amount - actual_investment
        end_date = target_date + timedelta(days=total_days)
        
        result = {
            "recommendation": recommendation,
            "investment": {
                "amount": investment_amount,
                "num_stocks": num_stocks,
                "price_per_stock": current_price,
                "actual_investment": actual_investment,
                "remaining_cash": remaining_cash
            },
            "dates": {
                "start_date": target_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "period_days": total_days
            }
        }
        print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    const tempScript = path.join(__dirname, `temp_recommend_${Date.now()}.py`);
    fs.writeFileSync(tempScript, script);

    const { stdout, stderr } = await execAsync(`cd "${PROJECT_ROOT}" && source venv/bin/activate && python3 "${tempScript}"`, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000
    });

    // Clean up temp file
    if (fs.existsSync(tempScript)) {
      fs.unlinkSync(tempScript);
    }

    // Parse last line (ignore TensorFlow warnings)
    const lines = stdout.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    const result = JSON.parse(lastLine);
    
    if (result.error) {
      throw new Error(result.error);
    }
    
    return result;
  } catch (error) {
    throw new Error(error.message || 'Failed to get recommendation');
  }
}

/**
 * Get backtest results by calling Python
 */
async function getBacktestResults(ticker, startDate, endDate, numStocks, predictedPrice, purchasePrice) {
  try {
    const script = `
import sys
import os
import json
sys.path.insert(0, '${PROJECT_ROOT}')
from backend.api import initialize_models, get_actual_price

try:
    # Initialize models first (loads pre-trained weights)
    initialize_models()
    
    ticker = "${ticker}"
    start_date = "${startDate}"
    end_date = "${endDate}"
    num_stocks = ${numStocks}
    predicted_price = ${predictedPrice}
    purchase_price = ${purchasePrice}
    
    start_actual = get_actual_price(ticker, start_date)
    end_actual = get_actual_price(ticker, end_date)
    
    if start_actual is None or end_actual is None:
        print(json.dumps({"error": "Could not retrieve actual prices"}))
    else:
        predicted_return_pct = ((predicted_price - purchase_price) / purchase_price) * 100
        predicted_value = num_stocks * predicted_price
        predicted_profit = predicted_value - (num_stocks * purchase_price)
        
        actual_return_pct = ((end_actual - purchase_price) / purchase_price) * 100
        actual_value = num_stocks * end_actual
        actual_profit = actual_value - (num_stocks * purchase_price)
        
        price_error = abs(predicted_price - end_actual)
        price_error_pct = (price_error / end_actual) * 100
        
        predicted_direction = 'up' if predicted_price > purchase_price else 'down'
        actual_direction = 'up' if end_actual > purchase_price else 'down'
        direction_correct = predicted_direction == actual_direction
        
        result = {
            "prices": {
                "purchase_price": purchase_price,
                "predicted_price": predicted_price,
                "actual_price": end_actual
            },
            "returns": {
                "predicted_return_pct": round(predicted_return_pct, 2),
                "actual_return_pct": round(actual_return_pct, 2),
                "predicted_profit": round(predicted_profit, 2),
                "actual_profit": round(actual_profit, 2)
            },
            "portfolio": {
                "num_stocks": num_stocks,
                "predicted_value": round(predicted_value, 2),
                "actual_value": round(actual_value, 2),
                "initial_investment": round(num_stocks * purchase_price, 2)
            },
            "accuracy": {
                "price_error": round(price_error, 2),
                "price_error_pct": round(price_error_pct, 2),
                "direction_correct": direction_correct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction
            },
            "dates": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    const tempScript = path.join(__dirname, `temp_backtest_${Date.now()}.py`);
    fs.writeFileSync(tempScript, script);

    const { stdout, stderr } = await execAsync(`cd "${PROJECT_ROOT}" && source venv/bin/activate && python3 "${tempScript}"`, {
      maxBuffer: 10 * 1024 * 1024,
      timeout: 60000
    });

    // Clean up temp file
    if (fs.existsSync(tempScript)) {
      fs.unlinkSync(tempScript);
    }

    // Parse last line (ignore warnings)
    const lines = stdout.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    const result = JSON.parse(lastLine);
    
    if (result.error) {
      throw new Error(result.error);
    }
    
    return result;
  } catch (error) {
    throw new Error(error.message || 'Failed to get backtest results');
  }
}

// MongoDB Connection with better error handling
let mongodbConnected = false;

if (process.env.MONGODB_URI && process.env.MONGODB_URI.includes('mongodb')) {
  mongoose.connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    serverSelectionTimeoutMS: 5000,
    socketTimeoutMS: 45000,
  })
  .then(() => {
    console.log('✓ MongoDB connected successfully');
    mongodbConnected = true;
  })
  .catch(err => {
    console.log('⚠ MongoDB connection failed:', err.message);
    console.log('⚠ Authentication will be disabled');
    mongodbConnected = false;
  });

  // Handle connection events
  mongoose.connection.on('error', (err) => {
    console.error('MongoDB connection error:', err);
    mongodbConnected = false;
  });

  mongoose.connection.on('disconnected', () => {
    console.log('MongoDB disconnected');
    mongodbConnected = false;
  });

  mongoose.connection.on('reconnected', () => {
    console.log('MongoDB reconnected');
    mongodbConnected = true;
  });
} else {
  console.log('⚠ MongoDB URI not provided - authentication disabled');
  console.log('💡 To enable authentication, add MONGODB_URI to your .env file');
}

// Auth routes (public)
app.use('/api/auth', authRoutes);

// Health check endpoint (public)
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    models_loaded: modelsLoaded,
    models_exist: checkModelsExist(),
    mongodb_connected: mongoose.connection.readyState === 1,
    authentication_enabled: mongoose.connection.readyState === 1
  });
});

// Initialize models endpoint
app.post('/api/initialize', async (req, res) => {
  try {
    const result = await initializeModels();
    if (result.status === 'success') {
      modelsLoaded = true;
    }
    res.json(result);
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Get available years (protected)
app.get('/api/available-years', protect, async (req, res) => {
  try {
    const years = await loadCombinedData();
    res.json({ years });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get stock recommendation (protected)
app.post('/api/recommend', protect, async (req, res) => {
  try {
    // Load models on first request if not already loaded
    if (!modelsLoaded && checkModelsExist()) {
      console.log('Loading models on first prediction request...');
      const initResult = await initializeModels();
      if (initResult.status !== 'success') {
        return res.status(500).json({ error: 'Failed to load models: ' + initResult.message });
      }
      modelsLoaded = true;
      console.log('✓ Models loaded successfully');
    }
    
    const { year, amount, days = 0, months = 0, years: yearsParam = 0 } = req.body;
    
    if (!year || !amount) {
      return res.status(400).json({ error: 'Year and amount are required' });
    }

    const result = await getStockRecommendation(
      parseInt(year),
      parseFloat(amount),
      parseInt(days),
      parseInt(months),
      parseInt(yearsParam)
    );
    
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Get backtest results (protected)
app.post('/api/backtest', protect, async (req, res) => {
  try {
    const { ticker, start_date, end_date, num_stocks, predicted_price, purchase_price } = req.body;
    
    if (!ticker || !start_date || !end_date) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }

    const result = await getBacktestResults(
      ticker,
      start_date,
      end_date,
      parseInt(num_stocks),
      parseFloat(predicted_price),
      parseFloat(purchase_price)
    );
    
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✓ Node.js backend server running on http://localhost:${PORT}`);
  console.log(`✓ CORS enabled for all origins`);
  console.log('');
  
  // Check models after server starts (but don't block)
  const modelsExist = checkModelsExist();
  if (modelsExist) {
    console.log('✓ AI models found');
    console.log('✓ Models will be loaded on first prediction request');
    console.log('');
    console.log('========================================');
    console.log('✓ Backend ready!');
    console.log('========================================');
  } else {
    console.log('⚠ Models not found');
    console.log('  Run: python3 model.py to train models');
    console.log('');
  }
});

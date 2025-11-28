"""
Flask API for Stock Prediction Backend
This API wraps the existing model functionality without modifying the core model code.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

# Add parent directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model functions (without running training)
from model import load_data, create_sequences, build_model, PortfolioManager, LOOKBACK, PREDICTION_HORIZON

# Get project root directory (parent of backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Disable any security features that might block requests
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = False

# Enable CORS for all routes - most permissive configuration
CORS(app, supports_credentials=True)

# Add CORS headers manually as additional fallback (in case CORS doesn't work)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS,PATCH')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Handle OPTIONS requests for all routes (CORS preflight)
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,X-Requested-With,Accept,Origin")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS,PATCH")
        response.headers.add('Access-Control-Allow-Credentials', "true")
        response.headers.add('Access-Control-Max-Age', "3600")
        return response

# Global variables to cache models and data
portfolio_manager = None
combined_data = None

def initialize_models():
    """Initialize models if they exist, otherwise return None"""
    global portfolio_manager, combined_data
    
    try:
        # Load combined data
        data_path = os.path.join(PROJECT_ROOT, 'combined_data.csv')
        if not os.path.exists(data_path):
            return None, None
            
        combined_data = pd.read_csv(data_path)
        combined_data['Date'] = pd.to_datetime(combined_data['Date'])
        
        # Check if models exist
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        if not os.path.exists(models_dir):
            return None, combined_data
            
        # Get available tickers
        tickers = combined_data['Ticker'].unique()
        
        # Initialize portfolio manager (without training)
        pm = PortfolioManager(tickers, lookback=LOOKBACK, horizon=PREDICTION_HORIZON)
        
        # Try to load existing models
        loaded_models = {}
        loaded_scalers = {}
        data_store = {}
        
        for ticker in tickers:
            model_path = os.path.join(models_dir, f'{ticker}.keras')
            if os.path.exists(model_path):
                try:
                    # Load model with custom objects if needed
                    try:
                        from model import directional_loss
                        model = load_model(model_path, custom_objects={'directional_loss': directional_loss})
                    except:
                        # If custom loss not needed for inference, load without it
                        model = load_model(model_path, compile=False)
                        model.compile(optimizer='adam', loss='mse')
                    loaded_models[ticker] = model
                    
                    # Load data for this ticker
                    df = load_data(data_path, ticker)
                    if df is not None:
                        # Ensure index is timezone-naive
                        if hasattr(df.index, 'tz') and df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        data_store[ticker] = df
                        
                        # Create scaler from training data
                        data = df.values
                        train_len = int(len(data) * 0.8)  # Same split as training
                        train_data = data[:train_len]
                        scaler = MinMaxScaler()
                        scaler.fit(train_data)
                        loaded_scalers[ticker] = scaler
                        
                except Exception as e:
                    print(f"Error loading model for {ticker}: {e}")
                    continue
        
        if loaded_models:
            pm.models = loaded_models
            pm.scalers = loaded_scalers
            pm.data_store = data_store
            portfolio_manager = pm
            return pm, combined_data
        
        return None, combined_data
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None, None

def get_stock_recommendation(target_date, horizon_days):
    """
    Get the best stock recommendation for a given date and investment horizon.
    Returns the ticker with highest expected return.
    """
    global portfolio_manager, combined_data
    
    if portfolio_manager is None or combined_data is None:
        return None, "Models not initialized. Please train models first."
    
    try:
        # Convert target_date to datetime (ensure timezone-naive)
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        if hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)
        
        # Find the closest available date in the dataset (before or on target_date)
        available_dates = combined_data['Date'].unique()
        # Convert to pandas Timestamp and ensure timezone-naive
        available_dates = [pd.Timestamp(d).replace(tzinfo=None) if hasattr(pd.Timestamp(d), 'tzinfo') else pd.Timestamp(d) for d in available_dates]
        available_dates = sorted([d for d in available_dates if d <= target_date])
        
        if not available_dates:
            return None, f"No data available for date {target_date}"
        
        # Use the closest date
        actual_date = available_dates[-1]
        
        # Get predictions for all tickers at this date
        recommendations = []
        
        for ticker in portfolio_manager.tickers:
            if ticker not in portfolio_manager.models:
                continue
            
            try:
                df = portfolio_manager.data_store[ticker]
                scaler = portfolio_manager.scalers[ticker]
                
                # Convert actual_date to pandas Timestamp if needed
                if isinstance(actual_date, str):
                    actual_date = pd.to_datetime(actual_date)
                
                # Find the index of the target date
                if actual_date not in df.index:
                    # Find closest date
                    date_idx = df.index.get_indexer([actual_date], method='nearest')[0]
                    actual_date = df.index[date_idx]
                
                # Get data up to the target date
                date_position = df.index.get_loc(actual_date)
                
                # Need at least LOOKBACK days before this date
                if date_position < LOOKBACK:
                    continue
                
                # Get the sequence ending at target date
                data_slice = df.iloc[date_position - LOOKBACK + 1:date_position + 1]
                data_values = data_slice.values
                
                # Scale the data
                data_scaled = scaler.transform(data_values)
                
                # Reshape for model input: (1, LOOKBACK, features)
                X = data_scaled.reshape(1, LOOKBACK, data_scaled.shape[1])
                
                # Predict price at horizon
                model = portfolio_manager.models[ticker]
                pred_scaled = model.predict(X, verbose=0)
                
                # Inverse transform
                dummy = np.zeros((1, data_values.shape[1]))
                dummy[:, 3] = pred_scaled.flatten()  # Close price is column 3
                pred_price = scaler.inverse_transform(dummy)[0, 3]
                
                # Get current price
                current_price = df.loc[actual_date, 'Close']
                
                # Calculate expected return
                expected_return = (pred_price - current_price) / current_price
                
                recommendations.append({
                    'ticker': ticker,
                    'company': combined_data[combined_data['Ticker'] == ticker]['Company'].iloc[0] if len(combined_data[combined_data['Ticker'] == ticker]) > 0 else ticker,
                    'current_price': float(current_price),
                    'predicted_price': float(pred_price),
                    'expected_return': float(expected_return),
                    'date': actual_date.strftime('%Y-%m-%d')
                })
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        if not recommendations:
            return None, "No valid recommendations found"
        
        # Sort by expected return
        recommendations.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return recommendations[0], None
        
    except Exception as e:
        return None, f"Error getting recommendation: {str(e)}"

def get_actual_price(ticker, date):
    """Get actual stock price from dataset for a given date"""
    global combined_data
    
    if combined_data is None:
        # Try to load it
        data_path = os.path.join(PROJECT_ROOT, 'combined_data.csv')
        if os.path.exists(data_path):
            combined_data = pd.read_csv(data_path)
            combined_data['Date'] = pd.to_datetime(combined_data['Date'])
        else:
            return None
    
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Ensure date is timezone-naive
        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
            date = date.replace(tzinfo=None)
        
        ticker_data = combined_data[combined_data['Ticker'] == ticker].copy()
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
        
        # Ensure all dates are timezone-naive
        if len(ticker_data) > 0 and hasattr(ticker_data['Date'].iloc[0], 'tzinfo'):
            if ticker_data['Date'].iloc[0].tzinfo is not None:
                ticker_data['Date'] = ticker_data['Date'].dt.tz_localize(None)
        
        ticker_data = ticker_data.sort_values('Date')
        ticker_data.set_index('Date', inplace=True)
        
        # Find closest date
        if date not in ticker_data.index:
            date_idx = ticker_data.index.get_indexer([date], method='nearest')[0]
            date = ticker_data.index[date_idx]
        
        if date in ticker_data.index:
            return float(ticker_data.loc[date, 'Close'])
        
        return None
        
    except Exception as e:
        print(f"Error getting actual price for {ticker} on {date}: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'models_loaded': portfolio_manager is not None})

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize models endpoint"""
    global portfolio_manager, combined_data
    portfolio_manager, combined_data = initialize_models()
    
    if portfolio_manager:
        return jsonify({
            'status': 'success',
            'tickers': list(portfolio_manager.tickers),
            'models_loaded': len(portfolio_manager.models)
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Models not found. Please train models first by running model.py'
        }), 400

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get stock recommendation for a given date and investment period"""
    try:
        data = request.json
        target_year = int(data.get('year'))
        investment_amount = float(data.get('amount', 0))
        time_period_days = int(data.get('days', 0))
        time_period_months = int(data.get('months', 0))
        time_period_years = int(data.get('years', 0))
        
        # Calculate total days
        total_days = time_period_days + (time_period_months * 30) + (time_period_years * 365)
        
        if total_days <= 0:
            return jsonify({'error': 'Invalid time period'}), 400
        
        # Set target date to start of the selected year (or a specific date)
        # For simplicity, use January 1st of the selected year
        target_date = datetime(target_year, 1, 1)
        
        # Get recommendation
        recommendation, error = get_stock_recommendation(target_date, total_days)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Calculate how many stocks can be bought
        current_price = recommendation['current_price']
        num_stocks = int(investment_amount / current_price)
        actual_investment = num_stocks * current_price
        remaining_cash = investment_amount - actual_investment
        
        # Calculate end date
        end_date = target_date + timedelta(days=total_days)
        
        return jsonify({
            'recommendation': recommendation,
            'investment': {
                'amount': investment_amount,
                'num_stocks': num_stocks,
                'price_per_stock': current_price,
                'actual_investment': actual_investment,
                'remaining_cash': remaining_cash
            },
            'dates': {
                'start_date': target_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'period_days': total_days
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Get backtesting results comparing predicted vs actual performance"""
    try:
        data = request.json
        ticker = data.get('ticker')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        num_stocks = int(data.get('num_stocks', 0))
        predicted_price = float(data.get('predicted_price', 0))
        purchase_price = float(data.get('purchase_price', 0))
        
        # Get actual prices
        start_actual = get_actual_price(ticker, start_date)
        end_actual = get_actual_price(ticker, end_date)
        
        if start_actual is None or end_actual is None:
            return jsonify({'error': 'Could not retrieve actual prices for the given dates'}), 400
        
        # Calculate returns
        # Predicted return
        predicted_return_pct = ((predicted_price - purchase_price) / purchase_price) * 100
        predicted_value = num_stocks * predicted_price
        predicted_profit = predicted_value - (num_stocks * purchase_price)
        
        # Actual return
        actual_return_pct = ((end_actual - purchase_price) / purchase_price) * 100
        actual_value = num_stocks * end_actual
        actual_profit = actual_value - (num_stocks * purchase_price)
        
        # Prediction accuracy
        price_error = abs(predicted_price - end_actual)
        price_error_pct = (price_error / end_actual) * 100
        
        # Direction accuracy (did we predict up/down correctly?)
        predicted_direction = 'up' if predicted_price > purchase_price else 'down'
        actual_direction = 'up' if end_actual > purchase_price else 'down'
        direction_correct = predicted_direction == actual_direction
        
        return jsonify({
            'prices': {
                'purchase_price': purchase_price,
                'predicted_price': predicted_price,
                'actual_price': end_actual
            },
            'returns': {
                'predicted_return_pct': round(predicted_return_pct, 2),
                'actual_return_pct': round(actual_return_pct, 2),
                'predicted_profit': round(predicted_profit, 2),
                'actual_profit': round(actual_profit, 2)
            },
            'portfolio': {
                'num_stocks': num_stocks,
                'predicted_value': round(predicted_value, 2),
                'actual_value': round(actual_value, 2),
                'initial_investment': round(num_stocks * purchase_price, 2)
            },
            'accuracy': {
                'price_error': round(price_error, 2),
                'price_error_pct': round(price_error_pct, 2),
                'direction_correct': direction_correct,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction
            },
            'dates': {
                'start_date': start_date,
                'end_date': end_date
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/available-years', methods=['GET', 'OPTIONS'])
def available_years():
    """Get list of available years in the dataset"""
    if request.method == 'OPTIONS':
        return '', 200
    
    global combined_data
    
    try:
        if combined_data is None:
            data_path = os.path.join(PROJECT_ROOT, 'combined_data.csv')
            if not os.path.exists(data_path):
                return jsonify({'error': 'Data file not found. Please run data_processor.py first.'}), 404
            combined_data = pd.read_csv(data_path)
            combined_data['Date'] = pd.to_datetime(combined_data['Date'])
        
        years = sorted(combined_data['Date'].dt.year.unique().tolist())
        # Remove 2014 from available years
        years = [y for y in years if y >= 2015]
        return jsonify({'years': years})
    except Exception as e:
        return jsonify({'error': f'Error loading years: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize models on startup
    print("Initializing models...")
    portfolio_manager, combined_data = initialize_models()
    
    if portfolio_manager:
        print(f"✓ Loaded {len(portfolio_manager.models)} models")
    else:
        print("⚠ Models not found. Please train models first by running: python3 model.py")
    
    # Run Flask app
    # Disable Flask's built-in security that might block requests
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


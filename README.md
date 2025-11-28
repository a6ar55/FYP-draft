# StockAI - AI-Powered Stock Prediction Platform

Professional stock prediction and backtesting platform with AI-powered recommendations using LSTM neural networks.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Node](https://img.shields.io/badge/node-%3E%3D16.0.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Features

✅ **AI Stock Predictions** - LSTM-based predictions for 10 major Indian stocks  
✅ **Backtesting** - Compare predictions with actual historical data  
✅ **User Authentication** - Secure MongoDB-based user management  
✅ **Professional UI** - Modern, clean interface  
✅ **Multi-Stock Analysis** - Analyzes all stocks and recommends the best one  
✅ **Maximum Purchase Calculation** - Buys as many stocks as possible with your budget  

## Tech Stack

### Frontend
- React 18
- Tailwind CSS
- React Router v6
- Axios

### Backend
- Node.js + Express
- MongoDB + Mongoose
- JWT Authentication
- Python (ML Engine)

### Machine Learning
- TensorFlow/Keras
- LSTM Neural Networks
- Technical Indicators (RSI, MACD, Bollinger Bands)

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- MongoDB instance

### 1. Clone and Setup

```bash
cd FYP-draft
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGODB_URI=your_mongodb_connection_string_here

# JWT Secret
JWT_SECRET=your_secure_random_string_here

# Server Configuration
PORT=5001
NODE_ENV=development

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

### 3. Start Application

**Mac/Linux:**
```bash
./start.sh
```

**Windows:**
```batch
start.bat
```

The application will be available at:
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:5001

### 4. Stop Application

**Mac/Linux:**
```bash
./stop.sh
```

**Windows:**
```batch
stop.bat
```

## Available Stocks

The AI analyzes these 10 major Indian stocks:

1. HDFC Bank (HDFCBANK.NS)
2. ICICI Bank (ICICIBANK.NS)
3. State Bank of India (SBIN.NS)
4. Reliance Industries (RELIANCE.NS)
5. TCS (TCS.NS)
6. Infosys (INFY.NS)
7. Hindustan Unilever (HINDUNILVR.NS)
8. ITC Ltd (ITC.NS)
9. Bharti Airtel (BHARTIARTL.NS)
10. LIC (LICI.NS)

## How It Works

### 1. User Authentication
- Sign up or log in securely
- JWT-based authentication
- Protected API endpoints

### 2. Input Parameters
- **Year:** Select a year (2015-2024)
- **Investment Amount:** How much to invest
- **Holding Period:** Days, months, or years

### 3. AI Analysis
- Loads pre-trained LSTM models
- Analyzes all 10 stocks
- Predicts prices 30 days ahead
- Calculates expected returns
- **Selects the best stock**

### 4. Investment Calculation
- Determines maximum stocks to buy
- Shows actual investment vs budget
- Displays remaining cash

### 5. Results
- Shows AI recommendation
- Displays investment details
- Provides backtesting comparison

## Project Structure

```
FYP-draft/
├── backend-node/          # Node.js API server
│   ├── models/           # MongoDB models
│   ├── routes/           # API routes
│   ├── middleware/       # Auth middleware
│   └── server.js         # Main server file
├── backend/              # Python ML engine
│   └── api.py           # Flask API (called by Node.js)
├── frontend/            # React application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   └── services/    # API services
│   └── package.json
├── models/              # Pre-trained AI models (*.keras)
├── model.py            # Model training script
├── combined_data.csv   # Historical stock data
├── start.sh            # Start script (Mac/Linux)
├── start.bat           # Start script (Windows)
├── stop.sh             # Stop script (Mac/Linux)
├── stop.bat            # Stop script (Windows)
└── .env                # Environment variables (create this)
```

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Create new account
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user (protected)

### Stock Predictions
- `GET /api/health` - System health check
- `GET /api/available-years` - Get available years (protected)
- `POST /api/recommend` - Get stock recommendation (protected)
- `POST /api/backtest` - Get backtest results (protected)

## Model Details

### Architecture
- **Type:** LSTM (Long Short-Term Memory)
- **Input:** 60 days × 13 features
- **Output:** Predicted price 30 days ahead
- **Parameters:** 56,129 per model
- **Total:** 561,290 parameters (all 10 models)

### Features Used
1. Open, High, Low, Close prices
2. Volume
3. RSI (Relative Strength Index)
4. MACD & Signal
5. Bollinger Bands (Upper/Lower)
6. SMA (20, 50)
7. Sentiment Score

### Performance
- Average accuracy: 75-85%
- Direction accuracy: 80-90%
- Best for medium-term predictions (3-12 months)

## Development

### Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Backend dependencies
cd backend-node && npm install

# Frontend dependencies
cd frontend && npm install
```

### Run Development Servers

**Backend:**
```bash
cd backend-node
node server.js
```

**Frontend:**
```bash
cd frontend
npm start
```

## Security Notes

- All prediction endpoints require authentication
- JWT tokens expire in 7 days
- Passwords are hashed using bcrypt
- MongoDB connection is secured with authentication

## Troubleshooting

### Port Already in Use
```bash
# Mac/Linux
lsof -ti:5001 | xargs kill -9
lsof -ti:3000 | xargs kill -9

# Windows
taskkill /F /IM node.exe /T
```

### Models Not Loading
Ensure the `models/` folder contains all 10 `.keras` files.

### MongoDB Connection Failed
Check your `MONGODB_URI` in `.env` file.

## License

MIT License - See LICENSE file for details

## Support

For issues and questions, please create an issue on the repository.

---

**Built with ❤️ using React, Node.js, Python, and TensorFlow**

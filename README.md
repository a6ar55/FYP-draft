# StockAI - AI-Powered Stock Prediction Platform

Professional stock prediction and backtesting platform using LSTM neural networks to predict stock prices and compare with actual results.

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Node](https://img.shields.io/badge/node-%3E%3D16.0.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Quick Start

### Prerequisites
- Node.js 16+
- Python 3.8+
- MongoDB (optional - works without it)

### Installation & Run

```bash
# Start everything (auto-installs dependencies)
./start.sh          # Mac/Linux
start.bat           # Windows
```

That's it! Open http://localhost:3000

### Stop Application

```bash
./stop.sh           # Mac/Linux
stop.bat            # Windows
```

## 📋 Features

- ✅ **AI Stock Predictions** - LSTM neural networks analyze 10 major Indian stocks
- ✅ **Backtesting** - Compare AI predictions with actual historical results
- ✅ **Beautiful UI** - Step-by-step guided flow with smooth animations
- ✅ **Authentication** - Optional MongoDB user management (works without it)
- ✅ **Real-time Analysis** - Instant predictions using pre-trained models
- ✅ **Portfolio Simulation** - See how investments would have performed

## 🏗️ Tech Stack

**Frontend**: React 18, Tailwind CSS, React Router v6  
**Backend**: Node.js + Express (port 5001)  
**Database**: MongoDB (optional)  
**ML Engine**: Python + TensorFlow/Keras  
**Authentication**: JWT (optional)

## 📊 Available Stocks

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

## 🎯 How It Works

### Step 1: Select Year
Choose investment year (2015-2024)

### Step 2: Enter Amount
Specify investment amount in ₹

### Step 3: Choose Period
Select holding period (days, months, years)

### Step 4: View Results
- AI recommends best stock
- Shows predicted price & returns
- Compares with actual results (backtesting)
- Displays model accuracy

## 🔧 Configuration

### Optional: MongoDB Setup

Create `.env` file in project root:

```env
MONGODB_URI=your_mongodb_connection_string
JWT_SECRET=your_secure_random_string
PORT=5001
NODE_ENV=development
FRONTEND_URL=http://localhost:3000
```

**Note:** Works perfectly without MongoDB in demo mode!

### MongoDB Atlas (Free)
1. Sign up at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create free cluster
3. Get connection string
4. Add to `.env` file

## 📁 Project Structure

```
FYP-draft/
├── backend-node/       # Node.js API server
├── frontend/           # React application
├── models/            # Pre-trained AI models (*.keras)
├── model.py           # ML training script
├── combined_data.csv  # Historical stock data
├── start.sh/bat       # Startup scripts
├── stop.sh/bat        # Stop scripts
├── .env               # Configuration (create this)
└── .cursorrules       # AI agent guidelines
```

## 🛠️ Development

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install backend dependencies
cd backend-node && npm install

# Install frontend dependencies
cd frontend && npm install
```

### Run Separately

```bash
# Backend (port 5001)
cd backend-node && node server.js

# Frontend (port 3000)
cd frontend && npm start
```

## 📖 API Endpoints

### Public
- `GET /api/health` - System health check
- `POST /api/auth/signup` - Create account (if MongoDB)
- `POST /api/auth/login` - User login (if MongoDB)

### Protected (or open if no MongoDB)
- `GET /api/available-years` - Get available years
- `POST /api/recommend` - Get stock recommendation
- `POST /api/backtest` - Get backtest results

## 🎨 UI Features

- **Step-by-step Flow**: Guided experience
- **Smooth Animations**: Professional transitions
- **Loading States**: AI-themed loading quotes
- **Error Handling**: Clear, helpful messages
- **Responsive Design**: Works on all devices
- **Light Theme**: Clean, professional appearance

## 🧪 Testing

### Test Prediction
1. Open http://localhost:3000
2. Select year: 2020
3. Amount: ₹100,000
4. Period: 3 years
5. Click "Get Prediction"
6. View AI recommendation and backtesting results

## 🔒 Security

- JWT authentication (optional)
- bcrypt password hashing
- CORS configured
- Input validation
- MongoDB optional (works without auth)

## 📊 Model Details

- **Architecture**: LSTM Neural Networks
- **Training Data**: 2015-2024 historical data
- **Input**: 60-day lookback window
- **Output**: 30-day ahead prediction
- **Parameters**: 56,129 per model
- **Accuracy**: 75-85% average

## 🐛 Troubleshooting

### Port Already in Use
```bash
lsof -ti:5001 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Models Not Loading
Check `models/` folder contains 10 `.keras` files

### MongoDB Issues
System works without MongoDB - just skip authentication

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- TensorFlow/Keras for ML framework
- React & Tailwind for beautiful UI
- MongoDB for database (optional)
- NSE India for stock data

## 📞 Support

- Check `.cursorrules` for detailed guidelines
- Review code comments
- Test in demo mode first
- MongoDB is optional - works without it!

---

**Made with ❤️ for backtesting and educational purposes**

⚠️ **Disclaimer**: This is for educational and backtesting purposes only. Not financial advice.

---

**Quick Links:**
- Frontend: http://localhost:3000
- Backend: http://localhost:5001
- Start: `./start.sh` or `start.bat`
- Stop: `./stop.sh` or `stop.bat`

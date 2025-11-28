# StockAI - Complete Setup Guide

## Prerequisites

### Required Software
1. **Node.js 16+** - [Download](https://nodejs.org/)
2. **Python 3.8+** - [Download](https://www.python.org/)
3. **MongoDB** - [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) (free tier available)

### Verify Installation
```bash
node --version  # Should be 16+
python3 --version  # Should be 3.8+
npm --version
```

## Step-by-Step Setup

### 1. Get MongoDB Connection String

#### Option A: MongoDB Atlas (Recommended)
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free account
3. Create a new cluster (free M0 tier)
4. Click "Connect" → "Connect your application"
5. Copy the connection string
6. Replace `<password>` with your database password

Your connection string will look like:
```
mongodb+srv://username:password@cluster.mongodb.net/stockai?retryWrites=true&w=majority
```

#### Option B: Local MongoDB
```
mongodb://localhost:27017/stockai
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cd FYP-draft
nano .env  # or use any text editor
```

Add the following content:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://your_username:your_password@cluster.mongodb.net/stockai?retryWrites=true&w=majority

# JWT Secret (generate a random string)
JWT_SECRET=your_super_secret_random_string_change_this_in_production

# Server Configuration
PORT=5001
NODE_ENV=development

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000
```

**Important:** 
- Replace `MONGODB_URI` with your actual MongoDB connection string
- Change `JWT_SECRET` to a random, secure string

### 3. Install Dependencies

The start script will automatically install dependencies, but you can do it manually:

```bash
# Python dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Backend (Node.js) dependencies
cd backend-node
npm install
cd ..

# Frontend (React) dependencies
cd frontend
npm install
cd ..
```

### 4. Verify Pre-trained Models

Check that the `models/` folder contains all 10 `.keras` files:

```bash
ls models/
```

You should see:
- BHARTIARTL.NS.keras
- HDFCBANK.NS.keras
- HINDUNILVR.NS.keras
- ICICIBANK.NS.keras
- INFY.NS.keras
- ITC.NS.keras
- LICI.NS.keras
- RELIANCE.NS.keras
- SBIN.NS.keras
- TCS.NS.keras

### 5. Start the Application

**Mac/Linux:**
```bash
chmod +x start.sh  # Make executable (first time only)
./start.sh
```

**Windows:**
```batch
start.bat
```

### 6. Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5001

### 7. Create Your First Account

1. Open http://localhost:3000
2. Click "Sign up"
3. Enter your name, email, and password
4. Start predicting!

## Testing the Setup

### 1. Test Backend Health
```bash
curl http://localhost:5001/api/health
```

Expected response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "models_exist": true,
  "mongodb_connected": true
}
```

### 2. Test Frontend
Open http://localhost:3000 in your browser. You should see the login page.

### 3. Test Authentication
1. Sign up with a test account
2. You should be redirected to the dashboard
3. Try making a prediction

## Common Issues and Solutions

### Issue: MongoDB Connection Failed

**Error:** `MongoDB connection failed: MongoServerError`

**Solution:**
1. Check your `.env` file has the correct `MONGODB_URI`
2. Verify your MongoDB Atlas IP whitelist includes your IP (0.0.0.0/0 for development)
3. Ensure your database password is correct

### Issue: Port Already in Use

**Error:** `EADDRINUSE: address already in use :::5001`

**Solution:**
```bash
# Mac/Linux
lsof -ti:5001 | xargs kill -9
lsof -ti:3000 | xargs kill -9

# Windows
taskkill /F /IM node.exe /T
```

### Issue: Models Not Loading

**Error:** `Models not found. Please train models first.`

**Solution:**
1. Verify `models/` folder exists
2. Check all 10 `.keras` files are present
3. If missing, the models are in the repository

### Issue: Python Virtual Environment Error

**Error:** `venv/bin/activate: No such file or directory`

**Solution:**
```bash
# Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Frontend Build Errors

**Error:** `Module not found: Can't resolve 'react-router-dom'`

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Production Deployment

### Environment Variables for Production

```env
MONGODB_URI=your_production_mongodb_uri
JWT_SECRET=very_secure_random_string_for_production
PORT=5001
NODE_ENV=production
FRONTEND_URL=https://your-frontend-domain.com
```

### Security Checklist
- [ ] Use strong JWT_SECRET (32+ characters)
- [ ] Use MongoDB authentication
- [ ] Enable HTTPS
- [ ] Set secure CORS origins
- [ ] Use environment-specific .env files
- [ ] Never commit .env to git

### Build Frontend for Production
```bash
cd frontend
npm run build
```

## Updating the Application

### Pull Latest Changes
```bash
git pull origin main
```

### Update Dependencies
```bash
# Python
pip install -r requirements.txt

# Backend
cd backend-node && npm install && cd ..

# Frontend
cd frontend && npm install && cd ..
```

### Restart Application
```bash
./stop.sh && ./start.sh
```

## Additional Resources

- **MongoDB Atlas Tutorial:** https://docs.mongodb.com/manual/tutorial/atlas-free-tier-setup/
- **JWT Best Practices:** https://jwt.io/introduction
- **React Router Documentation:** https://reactrouter.com/
- **TensorFlow/Keras:** https://www.tensorflow.org/guide/keras

## Need Help?

1. Check the logs:
   - Backend: `/tmp/stockai-backend.log`
   - Frontend: `/tmp/stockai-frontend.log`

2. Review the troubleshooting section in README.md

3. Create an issue on the repository with:
   - Error message
   - Steps to reproduce
   - Your environment (OS, Node version, Python version)

---

**Ready to predict? Start the application and happy investing! 📈**


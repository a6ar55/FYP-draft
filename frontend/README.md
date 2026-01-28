# Stock Prediction Backtesting Frontend

React-based frontend application for stock prediction backtesting. This application allows users to simulate stock investments by selecting a historical year, investment amount, and time period, then see how the AI model's predictions compare to actual market performance.

## Features

1. **Year Selection**: Choose a historical year to simulate your investment (Step 1)
2. **Investment Amount**: Enter the amount you want to invest (Step 2)
3. **Time Period**: Select how long to hold the investment in days, months, and years (Step 3)
4. **Recommendation & Results**: View model recommendations, purchase details, and backtesting results comparing predicted vs actual performance (Step 4)

## Project Structure

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── components/         # Reusable React components
│   │   ├── YearSelection/  # Step 1: Year selection component
│   │   ├── InvestmentAmount/ # Step 2: Investment amount input
│   │   ├── TimePeriod/      # Step 3: Time period selection
│   │   └── Results/         # Step 4: Results and backtesting display
│   ├── services/
│   │   └── api.js          # API service for backend communication
│   ├── App.js              # Main application component
│   ├── App.css             # Main application styles
│   ├── index.js            # React entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies and scripts
└── README.md              # This file
```

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

1. Make sure the Flask backend is running on `http://localhost:5000`

2. Start the React development server:
```bash
npm start
```

3. The application will open in your browser at `http://localhost:3000`

## Building for Production

To create a production build:

```bash
npm run build
```

This creates an optimized build in the `build/` directory.

## API Integration

The frontend communicates with the Flask backend through the following endpoints:

- `GET /api/health` - Health check
- `POST /api/initialize` - Initialize models
- `GET /api/available-years` - Get available years in dataset
- `POST /api/recommend` - Get stock recommendation
- `POST /api/backtest` - Get backtesting results

All API calls are handled through the `apiService` in `src/services/api.js`.

## User Flow

1. **Select Year**: User chooses a year from available years in the dataset
2. **Enter Amount**: User enters investment amount
3. **Select Period**: User specifies holding period (days, months, years)
4. **View Results**: 
   - Model recommends best stock
   - Shows purchase details (number of stocks, price, etc.)
   - Displays backtesting results comparing:
     - Predicted vs actual prices
     - Predicted vs actual returns
     - Model accuracy metrics
     - Portfolio value comparison

## Key Concepts

- **Backtesting**: The application simulates investments using historical data. When a user selects a year (e.g., 2020), they are "traveling back in time" to that year.
- **Model Prediction**: The AI model predicts the stock price at the end of the selected time period.
- **Performance Comparison**: Results show how close the model's predictions were to actual market performance.

## Dependencies

- React 18.2.0
- React DOM 18.2.0
- Axios 1.6.0 (for API calls)
- React Scripts 5.0.1 (build tools)

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Notes

- The frontend assumes the backend is running on `http://localhost:5000` by default
- You can override the API URL by setting the `REACT_APP_API_URL` environment variable
- Models must be trained before using the application (run `python3 model.py` in the project root)


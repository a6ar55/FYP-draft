#!/bin/bash

# Quick Start Script for Integrated Stock Prediction System
# This script automates the entire pipeline setup and execution

set -e  # Exit on error

echo "=========================================="
echo "  Stock Prediction System - Quick Start  "
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Step 1: Check Python version
print_status "Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if (( $(echo "$python_version >= 3.8" | bc -l) )); then
    print_success "Python version $python_version is compatible"
else
    print_error "Python 3.8 or higher required (found $python_version)"
    exit 1
fi

# Step 2: Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dependencies installed"

# Step 4: Create necessary directories
print_status "Creating directory structure..."
mkdir -p results
mkdir -p saved_models
mkdir -p processed_news_data
print_success "Directories created"

# Step 5: Check if stock data exists
print_status "Checking for stock data..."
if [ ! -d "stock_data" ] || [ -z "$(ls -A stock_data/*.csv 2>/dev/null)" ]; then
    print_error "Stock data not found in stock_data/ directory"
    print_warning "Please ensure your stock CSV files are in the stock_data/ directory"
    exit 1
else
    stock_count=$(ls -1 stock_data/*.csv 2>/dev/null | wc -l)
    print_success "Found $stock_count stock data files"
fi

# Step 6: Check if GoScraper articles exist
if [ -f "GoScraper/articles.json" ]; then
    print_success "GoScraper articles found"
else
    print_warning "GoScraper articles.json not found - will skip importing existing articles"
fi

# Step 7: Ask user what to run
echo ""
echo "=========================================="
echo "  What would you like to do?             "
echo "=========================================="
echo "1. Run complete pipeline (collect news, analyze sentiment, train models)"
echo "2. Only collect and analyze news"
echo "3. Only train models (skip news collection)"
echo "4. Make predictions using existing models"
echo "5. Quick demo (train one stock)"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        print_status "Running complete pipeline..."
        python integrated_pipeline.py
        print_success "Complete pipeline finished!"
        ;;
    2)
        print_status "Collecting and analyzing news..."
        python historical_news_scraper.py
        python sentiment_analyzer.py --model finbert
        print_success "News collection and sentiment analysis finished!"
        ;;
    3)
        print_status "Training models (skipping news collection)..."
        python integrated_pipeline.py --skip-news --skip-sentiment --only-train
        print_success "Model training finished!"
        ;;
    4)
        if [ ! -d "saved_models" ] || [ -z "$(ls -A saved_models/*.keras 2>/dev/null)" ]; then
            print_error "No trained models found. Please train models first (option 1 or 3)"
            exit 1
        fi
        print_status "Making predictions..."
        python prediction_api.py --all
        print_success "Predictions generated! Check predictions_summary.csv"
        ;;
    5)
        print_status "Running quick demo with INFY.NS..."
        python integrated_pipeline.py --ticker INFY.NS --model attention --epochs 50
        python prediction_api.py --ticker INFY.NS
        print_success "Quick demo finished!"
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
print_success "All done!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - View results in results/ directory"
echo "  - Trained models are in saved_models/"
echo "  - Make predictions: python prediction_api.py --ticker INFY.NS"
echo "  - View training summary: cat results/training_summary.csv"
echo ""
echo "For more options, see README_INTEGRATED_SYSTEM.md"
echo ""

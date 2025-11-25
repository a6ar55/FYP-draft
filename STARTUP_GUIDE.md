# Complete Startup Guide

## 🚀 Quick Start (Python Startup Script)

The **new Python startup script** (`startup.py`) replaces the bash script and provides:
- ✅ **Cross-platform support** (Windows, Linux, macOS)
- ✅ **Colored output** for better readability
- ✅ **Interactive menu** with 8 options
- ✅ **Automatic setup** and verification
- ✅ **Progress indicators** and status updates
- ✅ **Error handling** and recovery

## 📋 Prerequisites

### Minimum Requirements
- **Python:** 3.8 or higher
- **RAM:** 8GB (16GB recommended)
- **Storage:** 10GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)

### For GPU Acceleration (Recommended)
- **NVIDIA GPU:** L4, RTX 3060+, A100, etc.
- **CUDA:** 11.8 or 12.x
- **cuDNN:** Compatible with your CUDA version
- **GPU Drivers:** Latest NVIDIA drivers

## 🎯 Running the Startup Script

### Method 1: Direct Execution (Recommended)

```bash
# Make executable (first time only)
chmod +x startup.py

# Run the script
python startup.py

# Or (if executable)
./startup.py
```

### Method 2: Using Python Interpreter

```bash
python startup.py
```

### Method 3: On Windows

```cmd
python startup.py
```

## 📊 What the Script Does

### Automatic Checks (Steps 1-7)

```
[Step 1/10] Checking Python Version
  ✓ Python 3.10.12 is compatible

[Step 2/10] Setting Up Virtual Environment
  ℹ Creating virtual environment...
  ✓ Virtual environment created

[Step 3/10] Installing Dependencies
  ℹ Upgrading pip...
  ℹ Installing packages from requirements.txt...
  ⚠ This may take 5-10 minutes...
  ✓ All dependencies installed

[Step 4/10] Creating Directory Structure
  ℹ Created results/
  ℹ Created saved_models/
  ℹ Created processed_news_data/
  ℹ Created logs/
  ✓ Directory structure ready

[Step 5/10] Checking Stock Data
  ✓ Found 10 stock data files
  ℹ  • Infosys_Ltd_INFY.NS.csv
  ℹ  • HDFC_Bank_Ltd_HDFCBANK.NS.csv
  ...

[Step 6/10] Checking GoScraper Data
  ✓ GoScraper articles found (1.28 MB)

[Step 7/10] Testing GPU Configuration
  ℹ Checking TensorFlow GPU...
  ✓ TensorFlow GPU detected
  ℹ Checking PyTorch GPU...
  ✓ PyTorch GPU detected
```

### Interactive Menu (Step 8)

```
================================================================================
  What would you like to do?
================================================================================

  1. Run Complete Pipeline
     Collect news → Analyze sentiment → Train models

  2. Collect and Analyze News Only
     Skip training, just prepare news data

  3. Train Models Only
     Skip news collection (use existing data)

  4. Make Predictions
     Use existing trained models for predictions

  5. Quick Demo
     Train one stock quickly (INFY.NS, 50 epochs)

  6. Test GPU Setup
     Run comprehensive GPU verification

  7. Install Dependencies Only
     Just install packages and exit

  8. Exit
     Exit without doing anything

Enter your choice (1-8):
```

## 🎮 Menu Options Explained

### Option 1: Run Complete Pipeline ⭐ Recommended First Run

**What it does:**
1. Collects historical news from GDELT (2015-present)
2. Imports GoScraper articles
3. Analyzes sentiment using FinBERT
4. Preprocesses data (stock + technical indicators + sentiment)
5. Trains LSTM models for all 10 stocks
6. Generates evaluation metrics and plots

**Time:** 30-50 minutes (GPU) or 2-3 hours (CPU)

**When to use:** First time setup or complete refresh

**Output:**
- `news_database.db` - SQLite database with news
- `saved_models/*.keras` - Trained models (10 files)
- `results/*.png` - Prediction plots (10 files)
- `results/training_summary.csv` - Metrics summary

### Option 2: Collect and Analyze News Only

**What it does:**
1. Runs `historical_news_scraper.py`
2. Runs `sentiment_analyzer.py`

**Time:** 15-30 minutes

**When to use:**
- Update news database with latest articles
- Re-analyze sentiment with different model
- Prepare data before training

**Output:**
- `news_database.db` - Updated database
- `processed_news_data/*.csv` - Daily sentiment data

### Option 3: Train Models Only

**What it does:**
1. Skips news collection (uses existing `news_database.db`)
2. Preprocesses data
3. Trains models for all stocks

**Time:** 20-40 minutes (GPU)

**When to use:**
- Re-train models with different hyperparameters
- Train after updating news data
- Experiment with different model architectures

**Output:**
- `saved_models/*.keras` - New trained models
- `results/*.png` - Updated plots
- `results/training_summary.csv` - New metrics

### Option 4: Make Predictions

**What it does:**
1. Loads trained models from `saved_models/`
2. Generates predictions for all stocks
3. Creates summary CSV

**Time:** 1-2 minutes

**When to use:**
- After training models
- Daily predictions with latest data
- Testing model performance

**Output:**
- `predictions_summary.csv` - All predictions
- Console output with predicted prices

**Requires:** Trained models in `saved_models/`

### Option 5: Quick Demo ⚡ Fast Test

**What it does:**
1. Trains single stock (INFY.NS)
2. Uses 50 epochs (instead of 100)
3. Makes prediction for that stock

**Time:** 3-5 minutes (GPU) or 10-15 minutes (CPU)

**When to use:**
- Test system before full run
- Verify GPU is working
- Quick validation

**Output:**
- `saved_models/INFY.NS_model.keras`
- Prediction for INFY.NS

### Option 6: Test GPU Setup 🔧 GPU Verification

**What it does:**
1. Runs `test_gpu_setup.py`
2. Tests TensorFlow GPU
3. Tests PyTorch GPU
4. Runs benchmarks
5. Shows optimal batch sizes

**Time:** 30 seconds

**When to use:**
- Verify GPU configuration
- Check performance
- Troubleshoot GPU issues

**Output:** Comprehensive test results and recommendations

### Option 7: Install Dependencies Only

**What it does:**
1. Installs all packages from `requirements.txt`
2. Exits without running anything else

**Time:** 5-10 minutes

**When to use:**
- Pre-install dependencies
- Update packages
- Fix installation issues

### Option 8: Exit

Exits the script without doing anything.

## 🎨 Features

### Cross-Platform Compatibility

**Windows:**
```cmd
C:\> python startup.py
```

**Linux/macOS:**
```bash
$ python startup.py
# or
$ ./startup.py
```

### Colored Output

The script uses `colorama` for colored terminal output:
- 🟢 **Green:** Success messages
- 🔴 **Red:** Error messages
- 🟡 **Yellow:** Warnings
- 🔵 **Blue:** Information
- 🟣 **Magenta:** Prompts

If `colorama` is not installed, it falls back to plain text.

### Virtual Environment

**Automatic creation:**
- Creates `venv/` directory
- Installs all packages in isolated environment
- Uses venv Python for all operations

**Benefits:**
- Doesn't affect system Python
- Clean dependency management
- Easy to reset (delete `venv/` and re-run)

### Progress Indicators

Shows real-time progress during:
- Package installation
- News collection
- Sentiment analysis
- Model training

### Error Handling

**Graceful failures:**
- Continues even if some checks fail (with user confirmation)
- Clear error messages
- Suggestions for fixes

**Examples:**
```
✗ GPU not detected
⚠ Training will be slower on CPU
Continue anyway? (y/n):
```

## 🔧 Advanced Usage

### Environment Variables

Set these before running if needed:

```bash
# For NewsAPI (optional)
export NEWSAPI_KEY="your_api_key"

# For GPU memory limit (optional)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run script
python startup.py
```

### Custom Configuration

Edit `config_example.json` and use:

```bash
python integrated_pipeline.py --config config_example.json
```

### Manual Steps

If you prefer manual control:

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Verify GPU
python test_gpu_setup.py

# 3. Run pipeline
python integrated_pipeline.py

# 4. Make predictions
python prediction_api.py --all
```

## 🐛 Troubleshooting

### Issue: "Python version not compatible"

**Solution:**
```bash
# Check version
python --version

# Use specific version if available
python3.10 startup.py
```

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
# The script should install automatically, but if not:
pip install -r requirements.txt

# For GPU support:
pip install tensorflow[and-cuda]
```

### Issue: "GPU not detected"

**Solutions:**

1. **Check GPU exists:**
```bash
nvidia-smi
```

2. **Install CUDA:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

3. **Install GPU version of packages:**
```bash
pip install tensorflow[and-cuda]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

4. **Run GPU test:**
```bash
python test_gpu_setup.py
```

### Issue: "Stock data not found"

**Solution:**
```bash
# Create directory
mkdir -p stock_data

# Add your CSV files
cp /path/to/your/data/*.csv stock_data/

# Run again
python startup.py
```

### Issue: "Permission denied: startup.py"

**Solution:**
```bash
# Make executable
chmod +x startup.py

# Or run with python
python startup.py
```

### Issue: "colorama not found" (Colored output missing)

**Solution:**
```bash
pip install colorama

# Or the script will install it automatically
```

### Issue: Script hangs during installation

**Solution:**
```bash
# Install manually with verbose output
pip install -r requirements.txt -v

# Check for specific package errors
pip install tensorflow --verbose
```

## 📊 Expected Output

### Successful Run

```
================================================================================
================================================================================
  INTEGRATED STOCK MARKET PREDICTION SYSTEM - STARTUP MANAGER
================================================================================
================================================================================

Platform: Linux 5.15.0
Python: 3.10.12
Directory: /Users/darkarmy/Downloads/fyp_int

[Step 1/10] Checking Python Version
✓ Python 3.10.12 is compatible

[Step 2/10] Setting Up Virtual Environment
✓ Virtual environment created

[Step 3/10] Installing Dependencies
✓ All dependencies installed

[Step 4/10] Creating Directory Structure
✓ Directory structure ready

[Step 5/10] Checking Stock Data
✓ Found 10 stock data files

[Step 6/10] Checking GoScraper Data
✓ GoScraper articles found (1.28 MB)

[Step 7/10] Testing GPU Configuration
✓ TensorFlow GPU detected
✓ PyTorch GPU detected

================================================================================
  What would you like to do?
================================================================================

  1. Run Complete Pipeline
     Collect news → Analyze sentiment → Train models
  ...

Enter your choice (1-8): 5

================================================================================
  Running Quick Demo
================================================================================

ℹ Training INFY.NS with 50 epochs...
⚠ This will take 3-5 minutes with GPU (10-15 min with CPU)

[Training output...]

✓ Training completed!
ℹ Making prediction for INFY.NS...

[Prediction output...]

✓ Quick demo completed!

================================================================================
  Next Steps
================================================================================

📁 Results Location:
   • Trained models: saved_models/
   • Predictions: results/
   • Training plots: results/*.png
   • News database: news_database.db

📊 View Results:
   • Training summary: cat results/training_summary.csv
   • Predictions: cat predictions_summary.csv
   • TensorBoard: tensorboard --logdir logs/fit

🔧 Useful Commands:
   • GPU status: nvidia-smi
   • Test GPU: python test_gpu_setup.py
   • Train stock: python enhanced_model.py --ticker INFY.NS
   • Predict: python prediction_api.py --ticker INFY.NS

📖 Documentation:
   • README_INTEGRATED_SYSTEM.md - Complete guide
   • GPU_OPTIMIZATION_GUIDE.md - GPU optimization
   • SYSTEM_SUMMARY.md - Technical overview

================================================================================
                              ✓ ALL DONE!
================================================================================
```

## 🎯 Best Practices

### First Time Setup

```bash
# 1. Run startup script
python startup.py

# 2. Choose option 6 (Test GPU)
Enter your choice (1-8): 6

# 3. If GPU works, choose option 5 (Quick Demo)
python startup.py
Enter your choice (1-8): 5

# 4. If demo works, run complete pipeline
python startup.py
Enter your choice (1-8): 1
```

### Regular Usage

```bash
# Daily predictions
python startup.py
Enter your choice (1-8): 4

# Update news and retrain
python startup.py
Enter your choice (1-8): 2  # Collect news
python startup.py
Enter your choice (1-8): 3  # Train models

# Experiment with single stock
python enhanced_model.py --ticker RELIANCE.NS --model attention --epochs 150
```

### Monitoring

```bash
# Terminal 1: Run script
python startup.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: TensorBoard (after training starts)
tensorboard --logdir logs/fit
```

## 💡 Tips

1. **Always test GPU first** (Option 6) before running full pipeline
2. **Start with Quick Demo** (Option 5) to verify everything works
3. **Monitor GPU usage** with `nvidia-smi` during training
4. **Check logs** if something fails: `tail -f pipeline.log`
5. **Use virtual environment** (automatic) for clean installations
6. **Keep stock data updated** for best predictions
7. **Run news collection periodically** to update sentiment data

## 📞 Getting Help

If you encounter issues:

1. **Check logs:**
```bash
cat pipeline.log
tail -f pipeline.log  # Live monitoring
```

2. **Run GPU test:**
```bash
python test_gpu_setup.py
```

3. **Check documentation:**
- `README_INTEGRATED_SYSTEM.md` - Complete guide
- `GPU_OPTIMIZATION_GUIDE.md` - GPU issues
- `TROUBLESHOOTING.md` - Common problems

4. **Manual testing:**
```bash
# Test imports
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import torch; print(torch.__version__)"

# Test GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎉 Summary

The **Python startup script** provides a complete, user-friendly way to:
- ✅ Setup your environment automatically
- ✅ Verify all components work
- ✅ Run the entire pipeline or individual steps
- ✅ Monitor progress with colored output
- ✅ Handle errors gracefully

**Just run:** `python startup.py` and follow the menu!

---

**Pro Tip:** Bookmark this guide and refer to the menu options section when choosing what to run!

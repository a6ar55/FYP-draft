# Logging System Guide

## 📋 Overview

The system now includes **comprehensive logging** that captures everything to files:
- **`log.txt`** - Complete execution log (all operations, progress, errors)
- **`eval.txt`** - Evaluation metrics for all trained models

## 🎯 Features

### Dual Output Logging
- ✅ **Console:** Colored, user-friendly output
- ✅ **File:** Complete log saved to `log.txt`
- ✅ **Automatic:** No configuration needed
- ✅ **Thread-safe:** Safe for concurrent operations

### Metrics Logging
- ✅ **Individual:** Each model's metrics saved immediately
- ✅ **Summary:** Aggregate statistics for all models
- ✅ **Formatted:** Clean, readable format
- ✅ **Timestamped:** Know when each run completed

## 📁 Log Files

### `log.txt` - Complete Execution Log

**Contents:**
- Session start/end timestamps
- All steps and progress
- GPU configuration details
- Data preprocessing steps
- Model training progress
- Evaluation results
- Warnings and errors
- System information

**Example:**
```
================================================================================
Session started: 2024-11-25 10:30:15
================================================================================

10:30:15 - INFO - Logging initialized - Output to: /path/to/log.txt
10:30:16 - INFO - Mixed precision (FP16) enabled - expect 2-3x speedup
10:30:16 - INFO - Auto-detected optimal batch size: 256

================================================================================
  Training model for INFY.NS
================================================================================

10:30:20 - INFO - Building attention LSTM model with 52 features...
10:30:25 - INFO - Training model for up to 100 epochs...
10:30:25 - INFO - Batch size: 256 (GPU-optimized)

Epoch 1/100
20/20 [==============================] - 8s - loss: 0.0234 - val_loss: 0.0189
...

================================================================================
  Evaluation Metrics for INFY.NS
================================================================================

10:35:30 - INFO - RMSE: 18.5234
10:35:30 - INFO - MAE: 14.3421
10:35:30 - INFO - MAPE: 2.14%
10:35:30 - INFO - R² Score: 0.9456
10:35:30 - INFO - Directional Accuracy: 72.34%
10:35:30 - INFO - Metrics saved to eval.txt
```

### `eval.txt` - Evaluation Metrics

**Contents:**
- Individual model metrics
- Summary statistics
- Best/worst performers
- Timestamp for each evaluation

**Example:**
```
================================================================================
EVALUATION METRICS SUMMARY
================================================================================

Timestamp: 2024-11-25 10:35:30
--------------------------------------------------------------------------------

Stock: INFY.NS
----------------------------------------
ticker                        : INFY.NS
rmse                          :  18.5234
mae                           :  14.3421
mape                          :   2.14%
r2_score                      :   0.9456
directional_accuracy          :  72.3400

================================================================================

Timestamp: 2024-11-25 10:42:15
--------------------------------------------------------------------------------

Stock: TCS.NS
----------------------------------------
ticker                        : TCS.NS
rmse                          :  22.1567
mae                           :  17.8934
mape                          :   2.87%
r2_score                      :   0.9234
directional_accuracy          :  68.5600

================================================================================

...

================================================================================
SUMMARY OF ALL STOCKS
================================================================================

Average Metrics Across All Stocks:
--------------------------------------------------------------------------------
rmse                          :  20.3456 ±  4.2341
mae                           :  16.1234 ±  3.5678
mape                          :   2.45% ±  0.67%
r2_score                      :   0.9345 ±  0.0234
directional_accuracy          :  70.12% ±  5.34%

Best Performers (by R² Score):
--------------------------------------------------------------------------------
  INFY.NS        : R²=0.9456, RMSE=18.5234, Dir Acc=72.34%
  HDFCBANK.NS    : R²=0.9412, RMSE=19.2341, Dir Acc=71.23%
  TCS.NS         : R²=0.9234, RMSE=22.1567, Dir Acc=68.56%

Needs Improvement (by R² Score):
--------------------------------------------------------------------------------
  LICI.NS        : R²=0.8567, RMSE=28.3421, Dir Acc=62.34%
  ITC.NS         : R²=0.8789, RMSE=25.4567, Dir Acc=64.56%
  SBIN.NS        : R²=0.8934, RMSE=24.1234, Dir Acc=66.78%

================================================================================
Evaluation completed: 2024-11-25 11:30:45
================================================================================
```

## 🚀 Usage

### Automatic Logging (Default)

Everything is logged automatically when you run:

```bash
# All output goes to log.txt
python startup.py

# Or directly
python integrated_pipeline.py

# Or train single stock
python enhanced_model.py --ticker INFY.NS
```

### View Logs in Real-Time

```bash
# In another terminal, watch log file
tail -f log.txt

# Or with color highlighting (if available)
tail -f log.txt | grep --color -E "ERROR|WARNING|SUCCESS|$"
```

### View Evaluation Metrics

```bash
# View all metrics
cat eval.txt

# View just the summary
tail -50 eval.txt

# View specific stock
grep -A 10 "INFY.NS" eval.txt
```

## 📊 Log Levels

The logging system uses standard Python logging levels:

| Level | Usage | Example |
|-------|-------|---------|
| **INFO** | Normal operations | "Training started", "Model saved" |
| **WARNING** | Non-critical issues | "GPU not detected", "Some data missing" |
| **ERROR** | Critical failures | "File not found", "Training failed" |
| **DEBUG** | Detailed debugging | (Not used by default) |

## 🔧 Advanced Usage

### Custom Log File Location

Edit the scripts to change log file location:

```python
# In enhanced_model.py or integrated_pipeline.py
from logger_config import setup_logging

# Custom location
logger = setup_logging("my_custom_log.txt", level=logging.INFO)
```

### Enable Debug Logging

For more detailed logging:

```python
from logger_config import setup_logging
import logging

logger = setup_logging("log.txt", level=logging.DEBUG)
```

### Separate Log Files for Each Run

Add timestamp to log filename:

```python
from datetime import datetime
from logger_config import setup_logging

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logging(f"log_{timestamp}.txt")
```

### Capture All stdout/stderr

To capture even print statements:

```python
from logger_config import setup_logging

logger = setup_logging("log.txt", capture_stdout=True)
```

## 📈 Log Analysis

### Check for Errors

```bash
# Find all errors
grep "ERROR" log.txt

# Find all warnings
grep "WARNING" log.txt

# Count errors
grep -c "ERROR" log.txt
```

### Extract Training Times

```bash
# Find training completion messages
grep "Training completed" log.txt

# Extract evaluation metrics
grep -A 5 "Evaluation Metrics" log.txt
```

### Find Best Model

```bash
# Find highest R² scores in eval.txt
grep "r2_score" eval.txt | sort -k 3 -n -r | head -5
```

### Monitor Training Progress

```bash
# Watch for epoch completions
tail -f log.txt | grep "Epoch"

# Watch for GPU utilization
tail -f log.txt | grep "GPU"
```

## 🎯 Log File Management

### Rotate Logs

Automatically archive old logs:

```bash
# Archive logs older than 7 days
find . -name "log.txt" -mtime +7 -exec gzip {} \;

# Or create a rotation script
#!/bin/bash
if [ -f log.txt ]; then
    mv log.txt "log_$(date +%Y%m%d_%H%M%S).txt"
    gzip "log_$(date +%Y%m%d_%H%M%S).txt"
fi
```

### Clear Logs

```bash
# Clear log file (keep file)
> log.txt

# Clear eval file
> eval.txt

# Remove old logs
rm log_*.txt.gz
```

### Backup Logs

```bash
# Backup logs before new run
cp log.txt "log_backup_$(date +%Y%m%d).txt"
cp eval.txt "eval_backup_$(date +%Y%m%d).txt"

# Run your training
python integrated_pipeline.py
```

## 📋 What Gets Logged

### Startup Script (`startup.py`)
- ✅ Python version check
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Directory structure creation
- ✅ Stock data verification
- ✅ GPU detection
- ✅ User menu choices
- ✅ All operations and results

### Pipeline (`integrated_pipeline.py`)
- ✅ Pipeline start/end times
- ✅ Configuration settings
- ✅ Each step execution
- ✅ Success/failure status
- ✅ Error messages
- ✅ Warnings

### News Scraper (`historical_news_scraper.py`)
- ✅ News sources accessed
- ✅ Articles found per company
- ✅ Date ranges processed
- ✅ Database operations
- ✅ Statistics

### Sentiment Analyzer (`sentiment_analyzer.py`)
- ✅ Model loading
- ✅ Batch processing progress
- ✅ Articles analyzed
- ✅ Sentiment distribution
- ✅ Performance metrics

### Model Training (`enhanced_model.py`)
- ✅ GPU configuration
- ✅ Model architecture
- ✅ Training progress (epoch-by-epoch)
- ✅ Validation metrics
- ✅ Early stopping triggers
- ✅ Final evaluation metrics
- ✅ Model save locations

## 🔍 Troubleshooting with Logs

### Training Not Improving

```bash
# Check validation loss
grep "val_loss" log.txt | tail -20

# Look for early stopping
grep "Early stopping" log.txt
```

### GPU Not Being Used

```bash
# Check GPU detection
grep "GPU" log.txt | head -10

# Check batch size
grep "batch size" log.txt
```

### Out of Memory Errors

```bash
# Find OOM errors
grep -i "out of memory\|OOM" log.txt

# Check batch sizes used
grep "Batch size" log.txt
```

### Data Loading Issues

```bash
# Find file not found errors
grep "not found" log.txt

# Check data loading messages
grep "Loading\|Loaded" log.txt
```

## 💡 Best Practices

### 1. Always Check Logs After Training

```bash
# Quick health check
tail -100 log.txt

# Check for errors
grep "ERROR" log.txt

# View final metrics
tail -50 eval.txt
```

### 2. Archive Logs for Each Experiment

```bash
# Before new experiment
mkdir -p experiments/exp_$(date +%Y%m%d_%H%M%S)
cp log.txt eval.txt experiments/exp_$(date +%Y%m%d_%H%M%S)/
```

### 3. Compare Experiments

```bash
# Compare metrics from different runs
diff eval_run1.txt eval_run2.txt

# Or use a comparison script
python compare_metrics.py eval_run1.txt eval_run2.txt
```

### 4. Monitor Long Runs

```bash
# In tmux or screen session
python integrated_pipeline.py

# In another session
tail -f log.txt | grep -E "Epoch|completed|ERROR"
```

## 📝 Log File Locations

All logs are saved in the project root directory:

```
fyp_int/
├── log.txt              ← Main execution log
├── eval.txt             ← Evaluation metrics
├── pipeline.log         ← Legacy pipeline log (deprecated)
├── results/
│   └── training_summary.csv  ← CSV version of metrics
├── logs/
│   └── fit/             ← TensorBoard logs
└── predictions_summary.csv   ← Prediction results
```

## 🎉 Summary

The logging system provides:

✅ **Complete visibility** into all operations
✅ **Persistent records** of training runs
✅ **Easy debugging** with detailed logs
✅ **Performance tracking** with metrics
✅ **Comparison** between different runs
✅ **Automatic** - no manual setup needed

**Key Files:**
- **`log.txt`** - Everything that happened
- **`eval.txt`** - How well models performed

**Quick Commands:**
```bash
# View recent activity
tail -50 log.txt

# Check for problems
grep "ERROR\|WARNING" log.txt

# See final results
tail -100 eval.txt

# Monitor training
tail -f log.txt
```

---

**Pro Tip:** Always keep `log.txt` and `eval.txt` from successful training runs for future reference!

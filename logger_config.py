"""
Centralized Logging Configuration
Logs everything to both console and log.txt file
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading

# Thread-safe logging
_lock = threading.Lock()
_logger_initialized = False


class DualLogger:
    """Logger that writes to both console and file"""

    def __init__(self, log_file: str = "log.txt", level=logging.INFO):
        """
        Initialize dual logger

        Args:
            log_file: Path to log file
            level: Logging level
        """
        self.log_file = Path(log_file)
        self.level = level

        # Ensure log file exists
        self.log_file.touch(exist_ok=True)

    def get_logger(self, name: str = "stock_prediction") -> logging.Logger:
        """
        Get or create logger instance

        Args:
            name: Logger name

        Returns:
            Configured logger
        """
        global _logger_initialized

        with _lock:
            logger = logging.getLogger(name)

            # Only configure once
            if _logger_initialized:
                return logger

            logger.setLevel(self.level)
            logger.handlers.clear()  # Clear any existing handlers

            # Create formatters
            detailed_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            console_formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )

            # File handler (detailed logging)
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(self.level)
            file_handler.setFormatter(detailed_formatter)

            # Console handler (less verbose)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(console_formatter)

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            _logger_initialized = True

            # Log initialization
            logger.info("="*80)
            logger.info(f"Logging initialized - Output to: {self.log_file.absolute()}")
            logger.info("="*80)

            return logger


class TeeOutput:
    """Redirect stdout/stderr to both console and file"""

    def __init__(self, file_path: str, mode: str = 'a'):
        """
        Initialize Tee output

        Args:
            file_path: Path to output file
            mode: File open mode
        """
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        """Write to both console and file"""
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        """Flush both outputs"""
        self.stdout.flush()
        self.file.flush()

    def close(self):
        """Close file"""
        self.file.close()


def setup_logging(log_file: str = "log.txt",
                  level=logging.INFO,
                  capture_stdout: bool = True) -> logging.Logger:
    """
    Setup centralized logging

    Args:
        log_file: Path to log file
        level: Logging level
        capture_stdout: Whether to capture stdout/stderr

    Returns:
        Configured logger
    """
    # Create log file with header
    log_path = Path(log_file)

    # Write header to log file
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

    # Create logger
    dual_logger = DualLogger(log_file, level)
    logger = dual_logger.get_logger()

    # Optionally capture all stdout/stderr
    if capture_stdout:
        sys.stdout = TeeOutput(log_file)
        sys.stderr = TeeOutput(log_file)

    return logger


def log_section(logger: logging.Logger, title: str):
    """Log a section header"""
    logger.info("")
    logger.info("="*80)
    logger.info(f"  {title}")
    logger.info("="*80)
    logger.info("")


def log_subsection(logger: logging.Logger, title: str):
    """Log a subsection header"""
    logger.info("")
    logger.info("-"*80)
    logger.info(f"  {title}")
    logger.info("-"*80)


def log_metrics(metrics: dict, output_file: str = "eval.txt", append: bool = True):
    """
    Log evaluation metrics to file

    Args:
        metrics: Dictionary of metrics
        output_file: Path to output file
        append: Whether to append or overwrite
    """
    mode = 'a' if append else 'w'
    output_path = Path(output_file)

    with open(output_path, mode, encoding='utf-8') as f:
        # Write header if this is first entry or overwrite mode
        if not append or not output_path.exists() or output_path.stat().st_size == 0:
            f.write("="*80 + "\n")
            f.write("EVALUATION METRICS SUMMARY\n")
            f.write("="*80 + "\n\n")

        # Write timestamp
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-"*80 + "\n")

        # Write metrics
        if 'ticker' in metrics:
            f.write(f"\nStock: {metrics['ticker']}\n")
            f.write("-"*40 + "\n")

        for key, value in metrics.items():
            if key == 'ticker':
                continue

            # Format based on type
            if isinstance(value, float):
                if 'accuracy' in key.lower() or 'r2' in key.lower():
                    f.write(f"{key:30s}: {value:8.4f}\n")
                elif 'mape' in key.lower():
                    f.write(f"{key:30s}: {value:8.2f}%\n")
                else:
                    f.write(f"{key:30s}: {value:8.4f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")

        f.write("\n" + "="*80 + "\n\n")


def log_summary_metrics(all_metrics: list, output_file: str = "eval.txt"):
    """
    Log summary of all metrics

    Args:
        all_metrics: List of metric dictionaries
        output_file: Path to output file
    """
    import pandas as pd
    import numpy as np

    output_path = Path(output_file)

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY OF ALL STOCKS\n")
        f.write("="*80 + "\n\n")

        # Create DataFrame
        df = pd.DataFrame(all_metrics)

        # Calculate averages
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'ticker']

        f.write("Average Metrics Across All Stocks:\n")
        f.write("-"*80 + "\n")

        for col in numeric_cols:
            avg = df[col].mean()
            std = df[col].std()

            if 'accuracy' in col.lower() or 'r2' in col.lower():
                f.write(f"{col:30s}: {avg:8.4f} ± {std:6.4f}\n")
            elif 'mape' in col.lower():
                f.write(f"{col:30s}: {avg:8.2f}% ± {std:6.2f}%\n")
            else:
                f.write(f"{col:30s}: {avg:8.4f} ± {std:6.4f}\n")

        f.write("\n" + "="*80 + "\n")

        # Best and worst performers
        if 'r2_score' in df.columns:
            f.write("\nBest Performers (by R² Score):\n")
            f.write("-"*80 + "\n")
            best = df.nlargest(3, 'r2_score')
            for _, row in best.iterrows():
                f.write(f"  {row['ticker']:15s}: R²={row['r2_score']:.4f}, "
                       f"RMSE={row.get('rmse', 0):.4f}, "
                       f"Dir Acc={row.get('directional_accuracy', 0):.2f}%\n")

            f.write("\nNeeds Improvement (by R² Score):\n")
            f.write("-"*80 + "\n")
            worst = df.nsmallest(3, 'r2_score')
            for _, row in worst.iterrows():
                f.write(f"  {row['ticker']:15s}: R²={row['r2_score']:.4f}, "
                       f"RMSE={row.get('rmse', 0):.4f}, "
                       f"Dir Acc={row.get('directional_accuracy', 0):.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")


def close_logging():
    """Close all logging handlers"""
    # Restore stdout/stderr
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    if isinstance(sys.stderr, TeeOutput):
        sys.stderr.close()
        sys.stderr = sys.__stderr__

    # Close all handlers
    logger = logging.getLogger("stock_prediction")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# Default logger instance
default_logger = None


def get_logger(name: str = "stock_prediction") -> logging.Logger:
    """
    Get default logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    global default_logger

    if default_logger is None:
        default_logger = setup_logging()

    return default_logger


if __name__ == "__main__":
    # Test logging
    logger = setup_logging("test_log.txt")

    logger.info("Testing logging system")
    logger.warning("This is a warning")
    logger.error("This is an error")

    log_section(logger, "Test Section")
    log_subsection(logger, "Test Subsection")

    # Test metrics logging
    test_metrics = {
        'ticker': 'TEST.NS',
        'rmse': 25.34,
        'mae': 18.56,
        'mape': 2.45,
        'r2_score': 0.9234,
        'directional_accuracy': 68.75
    }

    log_metrics(test_metrics, "test_eval.txt", append=False)
    logger.info("Metrics logged to test_eval.txt")

    close_logging()
    print("Logging test completed - check test_log.txt and test_eval.txt")

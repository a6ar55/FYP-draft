#!/usr/bin/env python3
"""
Complete Startup Script for Integrated Stock Prediction System
Handles setup, verification, and execution of the entire pipeline
Cross-platform compatible (Windows, Linux, macOS)
"""

import os
import sys
import subprocess
import platform
import time
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

# Try to import colorama for cross-platform colors
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback if colorama not installed
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = ''
    class Style:
        BRIGHT = RESET_ALL = ''
    class Back:
        RED = GREEN = YELLOW = BLUE = ''


class Colors:
    """Color codes for terminal output"""
    if COLORS_AVAILABLE:
        HEADER = Fore.CYAN + Style.BRIGHT
        SUCCESS = Fore.GREEN + Style.BRIGHT
        ERROR = Fore.RED + Style.BRIGHT
        WARNING = Fore.YELLOW + Style.BRIGHT
        INFO = Fore.BLUE
        PROMPT = Fore.MAGENTA + Style.BRIGHT
        RESET = Style.RESET_ALL
    else:
        HEADER = SUCCESS = ERROR = WARNING = INFO = PROMPT = RESET = ''


class StartupManager:
    """Manages the complete startup process"""

    def __init__(self):
        self.project_dir = Path.cwd()
        self.python_cmd = sys.executable
        self.is_windows = platform.system() == 'Windows'
        self.is_linux = platform.system() == 'Linux'
        self.is_mac = platform.system() == 'Darwin'

        # Paths
        self.venv_dir = self.project_dir / 'venv'
        self.stock_data_dir = self.project_dir / 'stock_data'
        self.results_dir = self.project_dir / 'results'
        self.models_dir = self.project_dir / 'saved_models'
        self.news_dir = self.project_dir / 'processed_news_data'
        self.logs_dir = self.project_dir / 'logs'

        # Setup logging to file
        self.log_file = self.project_dir / 'log.txt'
        self._setup_file_logging()

    def _setup_file_logging(self):
        """Setup logging to file"""
        # Write startup header to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Startup Script Executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n")
            f.write(f"Directory: {self.project_dir}\n")
            f.write("="*80 + "\n\n")

    def _log_to_file(self, message: str):
        """Log message to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # Remove color codes for file logging
                import re
                clean_msg = re.sub(r'\033\[[0-9;]+m', '', message)
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {clean_msg}\n")
        except Exception:
            pass  # Silently fail if logging fails

    def print_banner(self):
        """Print welcome banner"""
        banner = f"""
{Colors.HEADER}{'='*80}
{'='*80}
{'  INTEGRATED STOCK MARKET PREDICTION SYSTEM - STARTUP MANAGER  '.center(80)}
{'='*80}
{'='*80}{Colors.RESET}

{Colors.INFO}Platform: {platform.system()} {platform.release()}
Python: {sys.version.split()[0]}
Directory: {self.project_dir}{Colors.RESET}
"""
        print(banner)

    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{Colors.HEADER}{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}{Colors.RESET}\n")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.SUCCESS}✓ {message}{Colors.RESET}")
        self._log_to_file(f"✓ {message}")

    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.ERROR}✗ {message}{Colors.RESET}")
        self._log_to_file(f"✗ {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.WARNING}⚠ {message}{Colors.RESET}")
        self._log_to_file(f"⚠ {message}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.INFO}ℹ {message}{Colors.RESET}")
        self._log_to_file(f"ℹ {message}")

    def print_step(self, step: int, total: int, message: str):
        """Print step indicator"""
        print(f"\n{Colors.PROMPT}[Step {step}/{total}] {message}{Colors.RESET}")

    def run_command(self, cmd: List[str], description: str,
                   check: bool = True, capture: bool = False) -> Tuple[bool, str]:
        """
        Run a command and handle output

        Args:
            cmd: Command to run as list
            description: Description for user
            check: Whether to check return code
            capture: Whether to capture output

        Returns:
            Tuple of (success, output)
        """
        self.print_info(f"{description}...")

        try:
            if capture:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=check
                )
                return True, result.stdout
            else:
                subprocess.run(cmd, check=check)
                return True, ""
        except subprocess.CalledProcessError as e:
            if capture:
                return False, e.stderr
            return False, str(e)
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.print_step(1, 10, "Checking Python Version")

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major >= 3 and version.minor >= 8:
            self.print_success(f"Python {version_str} is compatible")
            return True
        else:
            self.print_error(f"Python {version_str} is not compatible (need 3.8+)")
            return False

    def check_pip(self) -> bool:
        """Check if pip is available"""
        cmd = [self.python_cmd, '-m', 'pip', '--version']
        success, output = self.run_command(cmd, "Checking pip", capture=True)

        if success:
            self.print_success(f"pip is available")
            return True
        else:
            self.print_error("pip is not available")
            return False

    def create_virtual_environment(self) -> bool:
        """Create virtual environment if it doesn't exist"""
        self.print_step(2, 10, "Setting Up Virtual Environment")

        if self.venv_dir.exists():
            self.print_warning("Virtual environment already exists")
            return True

        self.print_info("Creating virtual environment...")
        cmd = [self.python_cmd, '-m', 'venv', str(self.venv_dir)]
        success, _ = self.run_command(cmd, "Creating venv", check=False)

        if success:
            self.print_success("Virtual environment created")
            return True
        else:
            self.print_warning("Could not create venv (will use system Python)")
            return True  # Continue anyway

    def get_venv_python(self) -> str:
        """Get path to Python in virtual environment"""
        if not self.venv_dir.exists():
            return self.python_cmd

        if self.is_windows:
            venv_python = self.venv_dir / 'Scripts' / 'python.exe'
        else:
            venv_python = self.venv_dir / 'bin' / 'python'

        if venv_python.exists():
            return str(venv_python)
        return self.python_cmd

    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        self.print_step(3, 10, "Installing Dependencies")

        requirements_file = self.project_dir / 'requirements.txt'

        if not requirements_file.exists():
            self.print_error("requirements.txt not found")
            return False

        python_cmd = self.get_venv_python()

        # Upgrade pip first
        self.print_info("Upgrading pip...")
        cmd = [python_cmd, '-m', 'pip', 'install', '--upgrade', 'pip']
        subprocess.run(cmd, capture_output=True)

        # Install colorama if not available
        if not COLORS_AVAILABLE:
            self.print_info("Installing colorama for better output...")
            cmd = [python_cmd, '-m', 'pip', 'install', 'colorama']
            subprocess.run(cmd, capture_output=True)

        # Install requirements
        self.print_info("Installing packages from requirements.txt...")
        self.print_warning("This may take 5-10 minutes...")

        cmd = [python_cmd, '-m', 'pip', 'install', '-r', str(requirements_file)]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Show progress
            for line in process.stdout:
                line = line.strip()
                if line.startswith('Collecting') or line.startswith('Downloading'):
                    print(f"  {line}")

            process.wait()

            if process.returncode == 0:
                self.print_success("All dependencies installed")
                return True
            else:
                self.print_error("Some packages failed to install")
                return False

        except Exception as e:
            self.print_error(f"Installation failed: {e}")
            return False

    def create_directories(self) -> bool:
        """Create necessary directories"""
        self.print_step(4, 10, "Creating Directory Structure")

        directories = [
            self.results_dir,
            self.models_dir,
            self.news_dir,
            self.logs_dir,
        ]

        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self.print_info(f"Created {directory.name}/")
            else:
                self.print_info(f"Directory {directory.name}/ exists")

        self.print_success("Directory structure ready")
        return True

    def check_stock_data(self) -> bool:
        """Check if stock data exists"""
        self.print_step(5, 10, "Checking Stock Data")

        if not self.stock_data_dir.exists():
            self.print_error(f"Stock data directory not found: {self.stock_data_dir}")
            self.print_warning("Please create 'stock_data/' and add your CSV files")
            return False

        csv_files = list(self.stock_data_dir.glob('*.csv'))

        if not csv_files:
            self.print_error("No CSV files found in stock_data/")
            self.print_warning("Please add stock data CSV files to stock_data/")
            return False

        self.print_success(f"Found {len(csv_files)} stock data files")
        for csv_file in csv_files[:5]:  # Show first 5
            self.print_info(f"  • {csv_file.name}")
        if len(csv_files) > 5:
            self.print_info(f"  • ... and {len(csv_files) - 5} more")

        return True

    def check_goscraper_data(self) -> bool:
        """Check if GoScraper articles exist"""
        self.print_step(6, 10, "Checking GoScraper Data")

        goscraper_dir = self.project_dir / 'GoScraper'
        articles_file = goscraper_dir / 'articles.json'

        if articles_file.exists():
            size_mb = articles_file.stat().st_size / (1024 * 1024)
            self.print_success(f"GoScraper articles found ({size_mb:.2f} MB)")
            return True
        else:
            self.print_warning("GoScraper articles.json not found")
            self.print_info("Will collect news data from GDELT only")
            return True  # Not critical

    def test_gpu(self) -> bool:
        """Test GPU availability"""
        self.print_step(7, 10, "Testing GPU Configuration")

        python_cmd = self.get_venv_python()

        # Test TensorFlow GPU
        tf_cmd = [
            python_cmd, '-c',
            'import tensorflow as tf; print("GPU" if tf.config.list_physical_devices("GPU") else "CPU")'
        ]
        success, output = self.run_command(tf_cmd, "Checking TensorFlow GPU",
                                          check=False, capture=True)

        if success and 'GPU' in output:
            self.print_success("TensorFlow GPU detected")
            gpu_available = True
        else:
            self.print_warning("TensorFlow GPU not detected (will use CPU)")
            gpu_available = False

        # Test PyTorch GPU
        pt_cmd = [
            python_cmd, '-c',
            'import torch; print("GPU" if torch.cuda.is_available() else "CPU")'
        ]
        success, output = self.run_command(pt_cmd, "Checking PyTorch GPU",
                                          check=False, capture=True)

        if success and 'GPU' in output:
            self.print_success("PyTorch GPU detected")
        else:
            self.print_warning("PyTorch GPU not detected (will use CPU)")

        if not gpu_available:
            self.print_warning("Training will be slower on CPU")
            response = input(f"\n{Colors.PROMPT}Continue anyway? (y/n): {Colors.RESET}")
            return response.lower() in ['y', 'yes']

        return True

    def run_gpu_test(self) -> bool:
        """Run comprehensive GPU test"""
        self.print_info("Running comprehensive GPU verification...")

        python_cmd = self.get_venv_python()
        test_script = self.project_dir / 'test_gpu_setup.py'

        if not test_script.exists():
            self.print_warning("test_gpu_setup.py not found, skipping detailed test")
            return True

        cmd = [python_cmd, str(test_script)]

        print(f"\n{Colors.INFO}{'─'*80}{Colors.RESET}")
        success, _ = self.run_command(cmd, "Running GPU tests", check=False)
        print(f"{Colors.INFO}{'─'*80}{Colors.RESET}\n")

        return success

    def show_menu(self) -> str:
        """Show interactive menu and get user choice"""
        self.print_section("What would you like to do?")

        options = {
            '1': ('Run Complete Pipeline',
                  'Collect news → Analyze sentiment → Train models'),
            '2': ('Collect and Analyze News Only',
                  'Skip training, just prepare news data'),
            '3': ('Train Models Only',
                  'Skip news collection (use existing data)'),
            '4': ('Make Predictions',
                  'Use existing trained models for predictions'),
            '5': ('Quick Demo',
                  'Train one stock quickly (INFY.NS, 50 epochs)'),
            '6': ('Test GPU Setup',
                  'Run comprehensive GPU verification'),
            '7': ('Install Dependencies Only',
                  'Just install packages and exit'),
            '8': ('Exit',
                  'Exit without doing anything'),
        }

        for key, (title, desc) in options.items():
            print(f"  {Colors.PROMPT}{key}.{Colors.RESET} {Colors.INFO}{title}{Colors.RESET}")
            print(f"     {desc}\n")

        while True:
            choice = input(f"{Colors.PROMPT}Enter your choice (1-8): {Colors.RESET}").strip()
            if choice in options:
                return choice
            print(f"{Colors.ERROR}Invalid choice. Please enter 1-8.{Colors.RESET}")

    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline"""
        self.print_section("Running Complete Pipeline")

        python_cmd = self.get_venv_python()
        script = self.project_dir / 'integrated_pipeline.py'

        if not script.exists():
            self.print_error("integrated_pipeline.py not found")
            return False

        self.print_warning("This will take 30-50 minutes with GPU (2-3 hours with CPU)")
        response = input(f"{Colors.PROMPT}Continue? (y/n): {Colors.RESET}")

        if response.lower() not in ['y', 'yes']:
            return False

        cmd = [python_cmd, str(script)]
        self.print_info("\nStarting pipeline...\n")

        try:
            subprocess.run(cmd)
            self.print_success("\nComplete pipeline finished!")
            return True
        except KeyboardInterrupt:
            self.print_warning("\nPipeline interrupted by user")
            return False
        except Exception as e:
            self.print_error(f"\nPipeline failed: {e}")
            return False

    def run_news_collection(self) -> bool:
        """Run news collection and sentiment analysis"""
        self.print_section("Collecting and Analyzing News")

        python_cmd = self.get_venv_python()

        # Step 1: Collect news
        self.print_info("\n[1/2] Collecting historical news...")
        script = self.project_dir / 'historical_news_scraper.py'

        if script.exists():
            subprocess.run([python_cmd, str(script)])
            self.print_success("News collection completed")
        else:
            self.print_error("historical_news_scraper.py not found")
            return False

        # Step 2: Analyze sentiment
        self.print_info("\n[2/2] Analyzing sentiment...")
        script = self.project_dir / 'sentiment_analyzer.py'

        if script.exists():
            subprocess.run([python_cmd, str(script), '--model', 'finbert'])
            self.print_success("Sentiment analysis completed")
        else:
            self.print_error("sentiment_analyzer.py not found")
            return False

        self.print_success("\nNews collection and analysis finished!")
        return True

    def run_training_only(self) -> bool:
        """Run model training only"""
        self.print_section("Training Models")

        python_cmd = self.get_venv_python()
        script = self.project_dir / 'integrated_pipeline.py'

        if not script.exists():
            self.print_error("integrated_pipeline.py not found")
            return False

        cmd = [python_cmd, str(script), '--skip-news', '--skip-sentiment', '--only-train']

        self.print_info("\nStarting training...\n")

        try:
            subprocess.run(cmd)
            self.print_success("\nModel training finished!")
            return True
        except KeyboardInterrupt:
            self.print_warning("\nTraining interrupted by user")
            return False
        except Exception as e:
            self.print_error(f"\nTraining failed: {e}")
            return False

    def run_predictions(self) -> bool:
        """Run predictions using existing models"""
        self.print_section("Making Predictions")

        # Check if models exist
        if not self.models_dir.exists() or not list(self.models_dir.glob('*.keras')):
            self.print_error("No trained models found")
            self.print_warning("Please train models first (option 1 or 3)")
            return False

        model_count = len(list(self.models_dir.glob('*.keras')))
        self.print_success(f"Found {model_count} trained models")

        python_cmd = self.get_venv_python()
        script = self.project_dir / 'prediction_api.py'

        if not script.exists():
            self.print_error("prediction_api.py not found")
            return False

        cmd = [python_cmd, str(script), '--all']

        self.print_info("\nGenerating predictions...\n")

        try:
            subprocess.run(cmd)
            self.print_success("\nPredictions generated!")
            self.print_info("Check predictions_summary.csv for results")
            return True
        except Exception as e:
            self.print_error(f"\nPrediction failed: {e}")
            return False

    def run_quick_demo(self) -> bool:
        """Run quick demo with one stock"""
        self.print_section("Running Quick Demo")

        python_cmd = self.get_venv_python()

        self.print_info("Training INFY.NS with 50 epochs...")
        self.print_warning("This will take 3-5 minutes with GPU (10-15 min with CPU)")

        # Training
        script = self.project_dir / 'integrated_pipeline.py'
        if script.exists():
            cmd = [python_cmd, str(script), '--ticker', 'INFY.NS',
                   '--model', 'attention', '--epochs', '50']

            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                self.print_warning("\nDemo interrupted")
                return False

        # Prediction
        self.print_info("\nMaking prediction for INFY.NS...")
        script = self.project_dir / 'prediction_api.py'
        if script.exists():
            cmd = [python_cmd, str(script), '--ticker', 'INFY.NS']
            subprocess.run(cmd)

        self.print_success("\nQuick demo completed!")
        return True

    def print_next_steps(self):
        """Print next steps for user"""
        self.print_section("Next Steps")

        print(f"{Colors.INFO}📁 Results Location:")
        print(f"   • Trained models: {Colors.CYAN}saved_models/{Colors.RESET}")
        print(f"   • Predictions: {Colors.CYAN}results/{Colors.RESET}")
        print(f"   • Training plots: {Colors.CYAN}results/*.png{Colors.RESET}")
        print(f"   • News database: {Colors.CYAN}news_database.db{Colors.RESET}")

        print(f"\n{Colors.INFO}📊 View Results:")
        print(f"   • Training summary: {Colors.CYAN}cat results/training_summary.csv{Colors.RESET}")
        print(f"   • Predictions: {Colors.CYAN}cat predictions_summary.csv{Colors.RESET}")
        print(f"   • TensorBoard: {Colors.CYAN}tensorboard --logdir logs/fit{Colors.RESET}")

        print(f"\n{Colors.INFO}🔧 Useful Commands:")
        print(f"   • GPU status: {Colors.CYAN}nvidia-smi{Colors.RESET}")
        print(f"   • Test GPU: {Colors.CYAN}python test_gpu_setup.py{Colors.RESET}")
        print(f"   • Train stock: {Colors.CYAN}python enhanced_model.py --ticker INFY.NS{Colors.RESET}")
        print(f"   • Predict: {Colors.CYAN}python prediction_api.py --ticker INFY.NS{Colors.RESET}")

        print(f"\n{Colors.INFO}📖 Documentation:")
        print(f"   • {Colors.CYAN}README_INTEGRATED_SYSTEM.md{Colors.RESET} - Complete guide")
        print(f"   • {Colors.CYAN}GPU_OPTIMIZATION_GUIDE.md{Colors.RESET} - GPU optimization")
        print(f"   • {Colors.CYAN}SYSTEM_SUMMARY.md{Colors.RESET} - Technical overview")

    def run(self):
        """Main execution flow"""
        try:
            # Print banner
            self.print_banner()

            # Check Python version
            if not self.check_python_version():
                return 1

            # Check pip
            if not self.check_pip():
                self.print_error("pip is required. Please install pip first.")
                return 1

            # Create virtual environment
            self.create_virtual_environment()

            # Install dependencies
            if not self.install_dependencies():
                self.print_error("Failed to install dependencies")
                response = input(f"\n{Colors.PROMPT}Continue anyway? (y/n): {Colors.RESET}")
                if response.lower() not in ['y', 'yes']:
                    return 1

            # Create directories
            self.create_directories()

            # Check stock data
            if not self.check_stock_data():
                self.print_error("Stock data is required to proceed")
                return 1

            # Check GoScraper data (optional)
            self.check_goscraper_data()

            # Test GPU
            if not self.test_gpu():
                return 1

            # Show menu and execute choice
            choice = self.show_menu()

            if choice == '1':
                success = self.run_complete_pipeline()
            elif choice == '2':
                success = self.run_news_collection()
            elif choice == '3':
                success = self.run_training_only()
            elif choice == '4':
                success = self.run_predictions()
            elif choice == '5':
                success = self.run_quick_demo()
            elif choice == '6':
                success = self.run_gpu_test()
            elif choice == '7':
                self.print_success("Dependencies already installed!")
                success = True
            elif choice == '8':
                self.print_info("Exiting...")
                return 0
            else:
                success = False

            # Print next steps if successful
            if success:
                self.print_next_steps()

                print(f"\n{Colors.SUCCESS}{'='*80}")
                print(f"{'  ✓ ALL DONE!  '.center(80)}")
                print(f"{'='*80}{Colors.RESET}\n")

                return 0
            else:
                self.print_error("\nSome operations failed. Check logs above.")
                return 1

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Interrupted by user. Exiting...{Colors.RESET}")
            return 1
        except Exception as e:
            print(f"\n\n{Colors.ERROR}Unexpected error: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Entry point"""
    manager = StartupManager()
    sys.exit(manager.run())


if __name__ == "__main__":
    main()

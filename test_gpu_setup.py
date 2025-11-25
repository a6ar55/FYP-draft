#!/usr/bin/env python3
"""
GPU Setup Test Script
Verifies that GPU is properly configured for optimal performance
"""

import sys
import os

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def test_imports():
    """Test if required packages are installed"""
    print_header("Testing Package Imports")

    packages = {
        'tensorflow': 'TensorFlow',
        'torch': 'PyTorch',
        'transformers': 'Transformers (HuggingFace)',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn'
    }

    all_imported = True

    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} not installed")
            all_imported = False

    return all_imported

def test_tensorflow_gpu():
    """Test TensorFlow GPU configuration"""
    print_header("Testing TensorFlow GPU")

    try:
        import tensorflow as tf

        # Check TensorFlow version
        print_info(f"TensorFlow version: {tf.__version__}")

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            print_error("No GPU detected by TensorFlow")
            return False

        print_success(f"Found {len(gpus)} GPU(s)")

        # Get GPU details
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            gpu_name = details.get('device_name', 'Unknown')
            print_info(f"GPU {i}: {gpu_name}")

        # Test GPU computation
        print_info("Testing GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)

        print_success("GPU computation test passed")

        # Check mixed precision
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.global_policy()
            print_info(f"Mixed precision policy: {policy.name}")
        except Exception as e:
            print_warning(f"Mixed precision check failed: {e}")

        return True

    except Exception as e:
        print_error(f"TensorFlow GPU test failed: {e}")
        return False

def test_pytorch_gpu():
    """Test PyTorch GPU configuration"""
    print_header("Testing PyTorch GPU")

    try:
        import torch

        # Check PyTorch version
        print_info(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            print_error("CUDA not available for PyTorch")
            return False

        print_success("CUDA is available")

        # GPU count
        gpu_count = torch.cuda.device_count()
        print_success(f"Found {gpu_count} GPU(s)")

        # GPU details
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print_info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

        # Test GPU computation
        print_info("Testing GPU computation...")
        device = torch.device('cuda:0')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)

        print_success("GPU computation test passed")

        # Check cuDNN
        print_info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print_info(f"cuDNN version: {torch.backends.cudnn.version()}")

        return True

    except Exception as e:
        print_error(f"PyTorch GPU test failed: {e}")
        return False

def test_gpu_config():
    """Test custom GPU configuration module"""
    print_header("Testing GPU Configuration Module")

    try:
        from gpu_config import setup_gpu, GPUConfig

        print_info("Initializing GPU configuration...")
        config = setup_gpu(mixed_precision=True, memory_growth=True, verbose=False)

        if config.gpu_available:
            print_success("GPU configuration successful")
            print_info(f"Number of GPUs: {config.num_gpus}")

            # Test optimal batch size calculation
            lstm_batch = config.get_optimal_batch_size('lstm', feature_count=50, sequence_length=60)
            sentiment_batch = config.get_optimal_batch_size('sentiment')

            print_info(f"Optimal LSTM batch size: {lstm_batch}")
            print_info(f"Optimal sentiment batch size: {sentiment_batch}")

            return True
        else:
            print_error("GPU not available in configuration")
            return False

    except ImportError as e:
        print_error(f"gpu_config module not found: {e}")
        return False
    except Exception as e:
        print_error(f"GPU configuration test failed: {e}")
        return False

def test_memory_benchmark():
    """Run a quick memory benchmark"""
    print_header("GPU Memory Benchmark")

    try:
        import torch
        import time

        if not torch.cuda.is_available():
            print_warning("GPU not available, skipping memory benchmark")
            return True

        device = torch.device('cuda:0')
        print_info("Running memory allocation test...")

        # Test different sizes
        sizes = [
            (1000, 1000, "1M elements"),
            (5000, 5000, "25M elements"),
            (10000, 10000, "100M elements")
        ]

        for rows, cols, desc in sizes:
            try:
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated(0) / (1024**2)

                tensor = torch.randn(rows, cols, device=device)

                end_mem = torch.cuda.memory_allocated(0) / (1024**2)
                mem_used = end_mem - start_mem

                print_success(f"{desc}: Allocated {mem_used:.2f} MB")

                del tensor

            except RuntimeError as e:
                print_error(f"{desc}: Failed - {e}")
                break

        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print_error(f"Memory benchmark failed: {e}")
        return False

def test_speed_benchmark():
    """Run a quick speed benchmark"""
    print_header("GPU Speed Benchmark")

    try:
        import tensorflow as tf
        import time
        import numpy as np

        print_info("Running matrix multiplication benchmark...")

        # CPU benchmark
        with tf.device('/CPU:0'):
            a = tf.random.normal([2000, 2000])
            b = tf.random.normal([2000, 2000])

            start = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()  # Force execution
            cpu_time = time.time() - start

        print_info(f"CPU time: {cpu_time*1000:.2f} ms")

        # GPU benchmark
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                a = tf.random.normal([2000, 2000])
                b = tf.random.normal([2000, 2000])

                # Warm-up
                _ = tf.matmul(a, b)

                start = time.time()
                c = tf.matmul(a, b)
                _ = c.numpy()  # Force execution
                gpu_time = time.time() - start

            print_info(f"GPU time: {gpu_time*1000:.2f} ms")

            speedup = cpu_time / gpu_time
            print_success(f"GPU speedup: {speedup:.2f}x")

            if speedup > 5:
                print_success("Excellent GPU performance!")
            elif speedup > 2:
                print_warning("Moderate GPU performance")
            else:
                print_warning("Low GPU speedup - check configuration")

        return True

    except Exception as e:
        print_error(f"Speed benchmark failed: {e}")
        return False

def print_recommendations():
    """Print optimization recommendations"""
    print_header("Optimization Recommendations")

    print(f"{Colors.BOLD}For your L4 GPU (24GB VRAM):{Colors.END}\n")

    recommendations = [
        ("Optimal LSTM batch size", "128-256", "Balance between speed and memory"),
        ("Optimal sentiment batch size", "64-128", "Efficient for FinBERT"),
        ("Mixed precision", "Enabled", "2-3x speedup with minimal accuracy loss"),
        ("Memory growth", "Enabled", "Prevents memory allocation issues"),
        ("XLA compilation", "Enabled", "10-20% speedup"),
        ("TF32 (Ampere)", "Enabled", "Faster matrix operations"),
    ]

    for setting, value, desc in recommendations:
        print(f"  {Colors.CYAN}•{Colors.END} {Colors.BOLD}{setting}:{Colors.END} {Colors.GREEN}{value}{Colors.END}")
        print(f"    {Colors.YELLOW}→{Colors.END} {desc}\n")

    print(f"\n{Colors.BOLD}Expected Performance:{Colors.END}")
    print(f"  • LSTM training: {Colors.GREEN}3-5 minutes per stock{Colors.END}")
    print(f"  • Sentiment analysis: {Colors.GREEN}2-3 minutes for 10K articles{Colors.END}")
    print(f"  • Total pipeline: {Colors.GREEN}30-50 minutes{Colors.END} (vs 7-10 hours on CPU)")

    print(f"\n{Colors.BOLD}Monitoring Commands:{Colors.END}")
    print(f"  • Watch GPU: {Colors.CYAN}watch -n 1 nvidia-smi{Colors.END}")
    print(f"  • TensorBoard: {Colors.CYAN}tensorboard --logdir logs/fit{Colors.END}")
    print(f"  • GPU info: {Colors.CYAN}python gpu_config.py{Colors.END}")

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║              GPU SETUP VERIFICATION & BENCHMARK                    ║")
    print("║                  Stock Prediction System                           ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(Colors.END)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['tensorflow'] = test_tensorflow_gpu()
    results['pytorch'] = test_pytorch_gpu()
    results['config'] = test_gpu_config()
    results['memory'] = test_memory_benchmark()
    results['speed'] = test_speed_benchmark()

    # Print summary
    print_header("Test Summary")

    all_passed = all(results.values())

    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        color = Colors.GREEN if passed else Colors.RED
        print(f"  {test.capitalize()}: {color}{status}{Colors.END}")

    print()

    if all_passed:
        print_success("All tests passed! GPU is properly configured.")
        print_recommendations()
        sys.exit(0)
    else:
        print_error("Some tests failed. Please check the errors above.")
        print_info("Refer to GPU_OPTIMIZATION_GUIDE.md for troubleshooting")
        sys.exit(1)

if __name__ == "__main__":
    main()

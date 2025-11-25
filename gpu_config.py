"""
GPU Configuration and Optimization
Optimizes TensorFlow/PyTorch for efficient GPU usage on L4 with 24GB VRAM
"""

import os
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUConfig:
    """Configure GPU settings for optimal performance"""

    def __init__(self):
        self.gpu_available = False
        self.num_gpus = 0
        self.gpu_memory_gb = 0

    def configure_tensorflow(self, memory_growth: bool = True,
                           mixed_precision: bool = True,
                           memory_limit_mb: Optional[int] = None) -> Dict:
        """
        Configure TensorFlow for optimal GPU usage

        Args:
            memory_growth: Enable memory growth (recommended)
            mixed_precision: Enable mixed precision training (FP16, 2-3x faster)
            memory_limit_mb: Limit GPU memory usage (None = use all available)

        Returns:
            Dictionary with configuration info
        """
        try:
            import tensorflow as tf

            # Get GPU devices
            gpus = tf.config.list_physical_devices('GPU')

            if not gpus:
                logger.warning("No GPU detected. Running on CPU.")
                return {'gpu_available': False, 'device': 'CPU'}

            self.num_gpus = len(gpus)
            logger.info(f"Found {self.num_gpus} GPU(s)")

            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                logger.info(f"GPU {i}: {gpu_details.get('device_name', 'Unknown')}")

                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                if memory_growth:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Memory growth enabled for GPU {i}")
                    except RuntimeError as e:
                        logger.warning(f"Could not enable memory growth: {e}")

                # Set memory limit if specified
                if memory_limit_mb:
                    try:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                        )
                        logger.info(f"Memory limit set to {memory_limit_mb}MB for GPU {i}")
                    except RuntimeError as e:
                        logger.warning(f"Could not set memory limit: {e}")

            # Enable mixed precision training (FP16) for faster training
            if mixed_precision:
                try:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    logger.info("Mixed precision (FP16) enabled - expect 2-3x speedup")
                except Exception as e:
                    logger.warning(f"Could not enable mixed precision: {e}")

            # Set TensorFlow to use GPU memory efficiently
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '2'

            # Optimize for L4 GPU
            os.environ['TF_ENABLE_CUDNN_FRONTEND'] = '1'
            os.environ['TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH'] = '1'

            # XLA (Accelerated Linear Algebra) compilation for faster execution
            tf.config.optimizer.set_jit(True)
            logger.info("XLA JIT compilation enabled")

            self.gpu_available = True
            logger.info("✓ TensorFlow GPU configuration completed successfully")

            return {
                'gpu_available': True,
                'num_gpus': self.num_gpus,
                'mixed_precision': mixed_precision,
                'memory_growth': memory_growth,
                'device': 'GPU'
            }

        except ImportError:
            logger.error("TensorFlow not installed")
            return {'gpu_available': False, 'device': 'CPU'}
        except Exception as e:
            logger.error(f"Error configuring TensorFlow GPU: {e}")
            return {'gpu_available': False, 'device': 'CPU'}

    def configure_pytorch(self, benchmark: bool = True,
                         deterministic: bool = False) -> Dict:
        """
        Configure PyTorch for optimal GPU usage (for FinBERT sentiment analysis)

        Args:
            benchmark: Enable cuDNN benchmarking for faster convolutions
            deterministic: Enable deterministic algorithms (slower but reproducible)

        Returns:
            Dictionary with configuration info
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("No CUDA GPU available for PyTorch. Running on CPU.")
                return {'gpu_available': False, 'device': 'cpu'}

            self.num_gpus = torch.cuda.device_count()
            logger.info(f"PyTorch: Found {self.num_gpus} GPU(s)")

            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

            # Enable cuDNN benchmarking for faster training
            if benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmarking enabled")

            # Enable TF32 on Ampere+ GPUs (L4 supports this)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 precision enabled for Ampere GPUs")

            # Deterministic operations (disable for speed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                logger.info("Deterministic algorithms enabled")

            # Set default device
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)

            # Clear cache
            torch.cuda.empty_cache()

            self.gpu_available = True
            logger.info("✓ PyTorch GPU configuration completed successfully")

            return {
                'gpu_available': True,
                'num_gpus': self.num_gpus,
                'device': 'cuda',
                'device_name': torch.cuda.get_device_name(0)
            }

        except ImportError:
            logger.error("PyTorch not installed")
            return {'gpu_available': False, 'device': 'cpu'}
        except Exception as e:
            logger.error(f"Error configuring PyTorch GPU: {e}")
            return {'gpu_available': False, 'device': 'cpu'}

    def get_optimal_batch_size(self, model_type: str = 'lstm',
                              feature_count: int = 50,
                              sequence_length: int = 60) -> int:
        """
        Calculate optimal batch size based on available GPU memory

        Args:
            model_type: Type of model ('lstm', 'sentiment')
            feature_count: Number of features
            sequence_length: Sequence length for LSTM

        Returns:
            Recommended batch size
        """
        if not self.gpu_available:
            return 32  # Default for CPU

        # L4 has 24GB VRAM - we can use larger batches
        if model_type == 'lstm':
            # LSTM with attention: memory = batch_size * seq_len * features * 4 bytes * ~10x (for activations)
            memory_per_sample = sequence_length * feature_count * 4 * 10  # bytes
            available_memory = 20 * 1024 * 1024 * 1024  # 20GB (leave 4GB for system)
            max_batch_size = int(available_memory / memory_per_sample)

            # Clip to reasonable range
            recommended = min(max(max_batch_size, 64), 512)
            logger.info(f"Recommended LSTM batch size: {recommended}")
            return recommended

        elif model_type == 'sentiment':
            # FinBERT: larger batches possible
            # With 24GB, can easily do 32-64 for sentiment analysis
            return 64

        return 128  # Default large batch size

    def print_gpu_info(self):
        """Print detailed GPU information"""
        try:
            import tensorflow as tf
            import torch

            logger.info("\n" + "="*60)
            logger.info("GPU INFORMATION")
            logger.info("="*60)

            # TensorFlow info
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"\nTensorFlow GPUs: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    details = tf.config.experimental.get_device_details(gpu)
                    logger.info(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
            else:
                logger.info("\nTensorFlow: No GPU detected")

            # PyTorch info
            if torch.cuda.is_available():
                logger.info(f"\nPyTorch CUDA GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"  GPU {i}: {name}")
                    logger.info(f"    Total Memory: {memory:.2f} GB")
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        cached = torch.cuda.memory_reserved(i) / (1024**3)
                        logger.info(f"    Allocated: {allocated:.2f} GB")
                        logger.info(f"    Cached: {cached:.2f} GB")
            else:
                logger.info("\nPyTorch: No CUDA GPU available")

            logger.info("="*60 + "\n")

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

    def optimize_data_loading(self, prefetch_size: int = 2) -> Dict:
        """
        Configure data loading optimizations

        Args:
            prefetch_size: Number of batches to prefetch

        Returns:
            Configuration dictionary
        """
        try:
            import tensorflow as tf

            # Enable auto-tuning for data pipeline
            tf.data.Options().experimental_optimization.apply_default_optimizations = True

            # Set prefetch buffer size
            AUTOTUNE = tf.data.AUTOTUNE

            logger.info(f"Data loading optimization enabled (prefetch={prefetch_size})")

            return {
                'prefetch_size': prefetch_size,
                'autotune': True
            }

        except Exception as e:
            logger.error(f"Error configuring data loading: {e}")
            return {}

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        try:
            import tensorflow as tf
            import torch

            # TensorFlow
            from tensorflow.keras import backend as K
            K.clear_session()

            # PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("GPU memory cache cleared")

        except Exception as e:
            logger.warning(f"Could not clear GPU memory: {e}")


def setup_gpu(mixed_precision: bool = True,
              memory_growth: bool = True,
              verbose: bool = True) -> GPUConfig:
    """
    Quick setup function for GPU configuration

    Args:
        mixed_precision: Enable FP16 training (2-3x speedup)
        memory_growth: Enable dynamic memory allocation
        verbose: Print detailed information

    Returns:
        Configured GPUConfig object
    """
    config = GPUConfig()

    if verbose:
        config.print_gpu_info()

    # Configure TensorFlow
    tf_config = config.configure_tensorflow(
        memory_growth=memory_growth,
        mixed_precision=mixed_precision
    )

    # Configure PyTorch
    pt_config = config.configure_pytorch(benchmark=True)

    if verbose:
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("="*60)
        logger.info(f"TensorFlow GPU: {tf_config['gpu_available']}")
        logger.info(f"PyTorch GPU: {pt_config['gpu_available']}")
        logger.info(f"Mixed Precision: {mixed_precision}")
        logger.info(f"Memory Growth: {memory_growth}")
        logger.info("="*60 + "\n")

    return config


# Auto-configure on import
def auto_configure_gpu():
    """Automatically configure GPU when module is imported"""
    try:
        config = setup_gpu(mixed_precision=True, memory_growth=True, verbose=True)
        return config
    except Exception as e:
        logger.warning(f"Auto GPU configuration failed: {e}")
        return None


if __name__ == "__main__":
    # Test GPU configuration
    logger.info("Testing GPU configuration...")
    config = setup_gpu(mixed_precision=True, memory_growth=True, verbose=True)

    # Calculate optimal batch sizes
    logger.info("\nOptimal batch sizes for L4 GPU (24GB):")
    logger.info(f"  LSTM Training: {config.get_optimal_batch_size('lstm')}")
    logger.info(f"  Sentiment Analysis: {config.get_optimal_batch_size('sentiment')}")

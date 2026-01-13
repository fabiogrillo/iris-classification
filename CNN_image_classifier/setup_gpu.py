"""
GPU Configuration for RTX 5080

IMPORTANT: The RTX 5080 (compute capability 12.0) is too new for current TensorFlow/PyTorch.
This script forces CPU mode until library support improves.

For GPU training, consider:
- Google Colab (free T4/A100 GPUs with full support)
- Wait for TensorFlow 2.21+ or PyTorch 2.7+
- Use Docker containers with nightly builds

Usage in notebook:
    import setup_gpu
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce log verbosity

import tensorflow as tf

def configure():
    """Configure TensorFlow for CPU mode"""
    print("=" * 70)
    print("TensorFlow Configuration")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"\n⚠️  RTX 5080 (Compute Capability 12.0) Status:")
    print("   Current TensorFlow doesn't fully support this GPU yet")
    print("   Running in CPU mode for stability")
    print(f"\n✅ Available devices:")

    for device in tf.config.list_physical_devices():
        print(f"   - {device.device_type}: {device.name}")

    print("\n💡 Alternatives for GPU training:")
    print("   - Use Google Colab (free GPU with full support)")
    print("   - Wait for TensorFlow 2.21+ with Blackwell support")
    print("   - CPU training for CIFAR-10 is actually quite fast!")
    print("=" * 70)

configure()

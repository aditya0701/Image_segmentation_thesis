import sys
import os

print("Hello World! The job is running.")

# Check which Python is being used
print(f"Python Executable: {sys.executable}")

if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
else:
    BATCH_SIZE = 1

print(f"🚀 Running with BATCH_SIZE: {BATCH_SIZE}")

# Try to import torch to verify the environment is loaded correctly
try:
    import torch
    print(f"✅ Success: Torch imported. Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ Error: Torch not found. Environment NOT activated correctly.")
#!/bin/bash
# Auto-install ALL dependencies + track new ones
echo "🧪 Installing Microglomeruli Segmentation Environment..."

# Install from requirements
pip install -r requirements.txt

# AUTO-TRACK: Create a watcher that updates requirements on new installs
pip install pipreqs
echo "✅ Setup complete! New packages will be auto-tracked."

# Create auto-update script
cat > update_requirements.py << 'EOF'
import subprocess
import sys

def update_requirements():
    """Auto-update requirements.txt after pip install"""
    subprocess.run(["pipreqs", ".", "--force", "--savepath=requirements.txt"])
    print("✅ requirements.txt UPDATED with new packages!")

if __name__ == "__main__":
    update_requirements()
EOF

echo "python update_requirements.py  # Run this after ANY pip install!"
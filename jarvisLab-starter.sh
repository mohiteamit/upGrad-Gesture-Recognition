#!/bin/bash

# Create a working directory under /home/<your_user>
WORK_DIR="$HOME/my_vm_setup"
mkdir -p "$WORK_DIR"

# Update and install Python (installing in user space)
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK_DIR/bin"
cd "$WORK_DIR"

# Install Python and dependencies (local installation)
if ! command -v python3.10 &> /dev/null; then
  echo "Installing Python 3.10..."
  wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
  tar -xvf Python-3.10.9.tgz
  cd Python-3.10.9
  ./configure --prefix="$WORK_DIR/python3.10"
  make && make install
  export PATH="$WORK_DIR/python3.10/bin:$PATH"
  cd ..
fi

# Set up a Python virtual environment
echo "Setting up Python virtual environment..."
$WORK_DIR/python3.10/bin/python3.10 -m venv "$WORK_DIR/env"
source "$WORK_DIR/env/bin/activate"

# Upgrade pip and install required libraries
pip install --upgrade pip
pip install tensorflow==2.18.0 opencv-python mediapipe numpy pandas matplotlib scikit-learn seaborn tensorboard scipy Pillow gdown jupyter notebook

# Confirm installation and GPU access
echo "Confirming TensorFlow GPU setup..."
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

echo "Setup complete. Activate your environment with:"
echo "source $WORK_DIR/env/bin/activate"

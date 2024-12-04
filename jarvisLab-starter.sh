#!/bin/bash

# Create a working directory under /home/<your_user>
WORK_DIR="$HOME/my_vm_setup"
mkdir -p "$WORK_DIR"

# Update and install Python (installing in user space)
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK_DIR/bin"
cd "$WORK_DIR"

git clone https://github.com/mohiteamit/upGrad-Gesture-Recognition.git

pip install --upgrade pip

pip install mediapipe gdown

gdown "https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL" -O "data.zip"

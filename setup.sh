#!/bin/bash

# mkdir /home/working
# cd /home/working
# git clone https://github.com/mohiteamit/upGrad-Gesture-Recognition.git
# python -m venv .venv/tensorFlow
# source /home/working.venv/tensorFlow/bin/activate
# ln -s /home/datasets/Project_data /home/working/upGrad-Gesture-Recognition/data


# # cuDNN update - do this before tensorflow install or reompile
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg-deb -x cuda-keyring_1.1-1_all.deb ~/local/

# cd ~/local
# ls

# export PATH=~/local/usr/bin:$PATH
# export LD_LIBRARY_PATH=~/local/usr/lib:$LD_LIBRARY_PATH

# echo 'export PATH=~/local/usr/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=~/local/usr/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc


# # Define directories
# HOME_DIR="$HOME"
# VENV_DIR="$HOME_DIR/tensorFlow"
# REPO_DIR="$HOME_DIR/upGrad-Gesture-Recognition"
# DATA_DIR="$REPO_DIR/data"

# # Step 1: Install virtualenv locally if not already installed
# echo "Installing virtualenv locally (if not installed)..."
# pip install --user virtualenv || { echo "Failed to install virtualenv"; exit 1; }

# # Step 2: Create a virtual environment named 'tensorFlow' under the home directory
# echo "Creating virtual environment 'tensorFlow' under home directory..."
# ~/.local/bin/virtualenv "$VENV_DIR" || { echo "Failed to create virtual environment"; exit 1; }

# # Step 3: Activate the virtual environment
# echo "Activating the virtual environment..."
# source "$VENV_DIR/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }

# # Step 4: Clone the repository into the home directory
# echo "Cloning the repository into the home directory..."
# git clone https://github.com/mohiteamit/upGrad-Gesture-Recognition.git "$REPO_DIR" || { echo "Failed to clone repository"; exit 1; }

# # Step 5: Navigate to the cloned repository folder
# cd "$REPO_DIR" || { echo "Repository folder not found"; exit 1; }

# # Step 6: Install requirements from requirements.txt
# echo "Installing Python requirements..."
# pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }
# pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }

# # Step 7: Create and navigate to the data folder within the repository
# echo "Setting up data directory under the repository..."
# mkdir -p "$DATA_DIR"
# cd "$DATA_DIR" || { echo "Data folder not found"; exit 1; }

# # Step 8: Download the data.zip file into the data directory
# echo "Downloading data.zip..."
# gdown "https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL" -O "$DATA_DIR/data.zip" || { echo "Failed to download data.zip"; exit 1; }

# # Step 9: Unzip data.zip within the data directory
# echo "Unzipping data.zip..."
# unzip -o "$DATA_DIR/data.zip" -d "$DATA_DIR" || { echo "Failed to unzip data.zip"; exit 1; }

# # Step 10: Move contents from Project_data (if exists) to the data directory
# if [ -d "$DATA_DIR/Project_data" ]; then
#     echo "Moving files from Project_data to data folder..."
#     mv "$DATA_DIR/Project_data/"* "$DATA_DIR" || { echo "Failed to move files from Project_data"; exit 1; }
#     rm -rf "$DATA_DIR/Project_data"  # Remove the empty Project_data folder
# else
#     echo "Project_data folder not found. Skipping move step."
# fi

# # Step 11: Delete the data.zip file
# echo "Deleting data.zip..."
# rm -f "$DATA_DIR/data.zip" || { echo "Failed to delete data.zip"; exit 1; }

# echo "Setup completed successfully!"

# # Reminder for the user to activate the virtual environment in future sessions
# echo "To activate the virtual environment in future sessions, run: source $VENV_DIR/bin/activate"

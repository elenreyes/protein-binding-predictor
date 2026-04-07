#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y dssp

# Create environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python deps
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."

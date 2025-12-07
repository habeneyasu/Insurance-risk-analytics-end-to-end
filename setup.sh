#!/bin/bash
# Setup script for Insurance Risk Analytics project

echo "Setting up Insurance Risk Analytics project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p reports/figures
mkdir -p data
mkdir -p notebooks

echo "Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run EDA, execute: python src/run_eda.py"


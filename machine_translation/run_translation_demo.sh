#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Run the simple example by default
if [ "$1" == "" ]; then
    echo "🚀 Running simple translation example..."
    python simple_translation_example.py
else
    # Run with provided arguments
    echo "🚀 Running machine translation demo with custom arguments..."
    python machine_translation_demo.py "$@"
fi

# Deactivate virtual environment
echo "✅ Done!" 
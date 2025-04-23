#!/bin/bash

# Check if jupytext is installed
if ! pip show jupytext &>/dev/null; then
    echo "📦 Installing jupytext..."
    pip install jupytext
fi

# Convert Python scripts to notebooks
echo "🔄 Checking for notebooks and converting if needed..."

# Convert translation tutorial if needed
if [ ! -f "machine_translation_tutorial.ipynb" ]; then
    echo "Converting machine_translation_tutorial.py to notebook..."
    jupytext --to notebook machine_translation_tutorial.py
    echo "✅ Created machine_translation_tutorial.ipynb"
else
    echo "✅ machine_translation_tutorial.ipynb already exists"
fi

# Convert model architecture exploration if needed
if [ ! -f "translation_model_architecture.ipynb" ]; then
    echo "Converting translation_model_architecture.py to notebook..."
    jupytext --to notebook translation_model_architecture.py
    echo "✅ Created translation_model_architecture.ipynb"
else
    echo "✅ translation_model_architecture.ipynb already exists"
fi

echo "🎉 All notebooks are ready! You can now open them in Jupyter." 
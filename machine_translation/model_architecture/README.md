# 🏗️ Translation Model Architecture Activity

This activity explores the internal workings of transformer-based translation models.

## 🎯 Learning Objectives

- Understand the architecture of transformer models for translation
- Explore encoder-decoder structures in translation models
- Visualize attention mechanisms in transformers
- Learn about tokenization and embedding processes

## 🚀 Quick Start

```bash
# Run the model architecture exploration script
python translation_model_architecture.py

# Or explore the interactive notebook
jupyter notebook translation_model_architecture.ipynb
```

## 📋 Contents

- `translation_model_architecture.py` - Script exploring translation model architectures
- `translation_model_architecture.ipynb` - Interactive notebook with visualizations

## 🧩 Model Components

The activity explores:
- **Encoder-Decoder Architecture** - How information flows through the model
- **Attention Mechanisms** - How attention helps align words between languages
- **Tokenization** - How text is converted to tokens for the model
- **Embeddings** - How words are represented in high-dimensional space

## 💻 Example Code

```python
from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Print model architecture
print(model)

# Analyze number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

## 🔍 Assignment Ideas

1. Visualize attention patterns for specific words across languages
2. Compare architectures of different translation models
3. Analyze how model size affects translation quality
4. Explore how positional encoding works in translation models 
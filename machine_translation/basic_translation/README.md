# 🌎 Basic Translation Activity

This activity introduces you to using pre-trained transformer models for basic machine translation tasks.

## 🎯 Learning Objectives

- Use pre-trained Hugging Face models for translation
- Understand how to load and use translation pipelines
- Translate text between different language pairs
- Learn basic model handling and inference

## 🚀 Quick Start

```bash
# Run the simple translation example
python simple_translation_example.py
```

## 📋 Contents

- `simple_translation_example.py` - Basic script demonstrating translation with pre-trained models
- `huggingface_hub_example.py` - Example showing how to use the Hugging Face Hub

## 💻 Example Code

```python
from transformers import pipeline

# Load a pre-trained translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

# Translate text
text = "Hello, how are you doing today?"
translated = translator(text)
print(translated[0]['translation_text'])
```

## 🔍 Assignment Ideas

1. Try translating with different pre-trained models
2. Compare translations between different models for the same language pair
3. Create a simple command-line translation tool
4. Try translating text from different domains (news, technical, conversational) 
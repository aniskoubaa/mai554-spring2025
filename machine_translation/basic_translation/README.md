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
   - Experiment with models like `Helsinki-NLP/opus-mt-fr-en`, `Helsinki-NLP/opus-mt-ar-en`, or `facebook/nllb-200-distilled-600M`
   - Compare output quality across different language families

2. Compare translations between different models for the same language pair
   - Try both specialized models (Helsinki-NLP) and multilingual models (NLLB, mBART)
   - Analyze differences in handling of idioms, technical terms, or cultural references

3. Create a simple command-line translation tool
   - Build a tool that accepts language codes as parameters
   - Add features like batch translation from text files
   - Implement caching to avoid re-translating the same content

4. Try translating text from different domains (news, technical, conversational)
   - Collect sample texts from various domains
   - Analyze where models perform well or struggle
   - Document domain-specific translation challenges 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Translation Tutorial with Pretrained Transformers
MAI554 Deep Learning Course

This script demonstrates how to use Hugging Face's pretrained transformer models for
machine translation and how to evaluate the results using BLEU score.

This can be run as a Python script or converted to a Jupyter notebook.
To convert to a notebook, run:
    jupytext --to notebook machine_translation_tutorial.py
"""

# %% [markdown]
# # 🌍 Machine Translation with Pretrained Transformers 🤖
# 
# This notebook demonstrates how to use Hugging Face's pretrained transformer models for machine translation 
# and how to evaluate the results using BLEU score.
# 
# ## 📚 Overview
# 
# In this tutorial, we will:
# 1. Load a pretrained transformer model for translation
# 2. Translate text from one language to another
# 3. Evaluate the translation quality using BLEU score
# 4. Visualize the results
# 5. Learn how to push models to HuggingFace Hub

# %% [markdown]
# ## 📦 Install Required Packages
# 
# Let's start by installing the necessary packages:

# %%
# Uncomment and run this cell if you need to install the packages
# !pip install transformers datasets nltk matplotlib tqdm huggingface-hub sentencepiece sacremoses torch

# %% [markdown]
# ## 🚀 Import Libraries

# %%
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Ensure the required NLTK data is downloaded
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
if not os.path.exists(punkt_dir):
    print("Downloading NLTK punkt tokenizer data...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
else:
    print("NLTK punkt tokenizer data already exists.")

# Also make sure punkt_tab is downloaded
print("Downloading additional NLTK resources needed for tokenization...")
nltk.download('all', download_dir=nltk_data_dir, quiet=False)

# %% [markdown]
# ## 🔄 Loading a Translation Model
# 
# We'll use Hugging Face's `pipeline` API to load a pretrained English-to-French translation model:

# %%
# Define language parameters
source_lang = 'en'
target_lang = 'fr'
model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Load the translation model
translator = pipeline(f"translation_{source_lang}_to_{target_lang}", model=model_name)
print(f"✅ Loaded model {model_name} for translation from {source_lang} to {target_lang}")

# %% [markdown]
# ## 🌐 Basic Translation Example
# 
# Let's try translating a simple sentence:

# %%
# Define a sample sentence
source_sentence = "Machine translation is transforming natural language processing."

# Translate the sentence
result = translator(source_sentence, max_length=60)
translated_text = result[0]['translation_text']

# Display results
print(f"English: {source_sentence}")
print(f"French: {translated_text}")

# %% [markdown]
# ## 📊 Evaluating Translation Quality with BLEU
# 
# BLEU (Bilingual Evaluation Understudy) is a metric for evaluating machine translations. It compares 
# a machine translation output against one or more reference translations and computes a score based 
# on n-gram precision.

# %%
def compute_bleu_score(reference, hypothesis):
    """Compute BLEU score between reference and hypothesis"""
    try:
        # Tokenize the sentences into words
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate BLEU score
        return sentence_bleu([reference_tokens], hypothesis_tokens)
    except LookupError:
        # If NLTK data is not found, try downloading it again
        print("⚠️ NLTK data not found. Attempting to download again...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
        # Try again after downloading
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        return sentence_bleu([reference_tokens], hypothesis_tokens)

# Define a source sentence and its reference translation
source = "Machine translation is transforming natural language processing."
reference = "La traduction automatique transforme le traitement du langage naturel."

# Translate the source text
result = translator(source, max_length=100)
translation = result[0]['translation_text']

# Compute BLEU score
bleu = compute_bleu_score(reference, translation)

# Display results
print(f"Source:      {source}")
print(f"Reference:   {reference}")
print(f"Translation: {translation}")
print(f"BLEU Score:  {bleu:.4f}")

# %% [markdown]
# ## 🧪 Multiple Examples and Evaluation

# %%
# Define source sentences and their reference translations
examples = [
    (
        "Machine translation is transforming natural language processing.",
        "La traduction automatique transforme le traitement du langage naturel."
    ),
    (
        "The quick brown fox jumps over the lazy dog.",
        "Le rapide renard brun saute par-dessus le chien paresseux."
    ),
    (
        "Artificial intelligence is the future of technology.",
        "L'intelligence artificielle est l'avenir de la technologie."
    ),
    (
        "I love learning about deep learning and neural networks.",
        "J'adore apprendre sur l'apprentissage profond et les réseaux de neurones."
    ),
    (
        "The weather is beautiful today.",
        "Le temps est magnifique aujourd'hui."
    )
]

results = []
bleu_scores = []

print("🔄 Translating examples...")
for source_text, reference_text in tqdm(examples):
    # Translate the source text
    translation_output = translator(source_text, max_length=100)
    translated_text = translation_output[0]['translation_text']
    
    # Compute BLEU score
    bleu = compute_bleu_score(reference_text, translated_text)
    bleu_scores.append(bleu)
    
    # Store results
    results.append({
        'source': source_text,
        'reference': reference_text,
        'translation': translated_text,
        'bleu': bleu
    })

# Display results
print("\n🔍 Translation Results:")
print("=" * 80)
for i, result in enumerate(results):
    print(f"Example {i+1}:")
    print(f"Source:     {result['source']}")
    print(f"Reference:  {result['reference']}")
    print(f"Translation: {result['translation']}")
    print(f"BLEU Score: {result['bleu']:.4f}")
    print("-" * 80)

# Calculate and display average BLEU score
avg_bleu = np.mean(bleu_scores)
print(f"📈 Average BLEU Score: {avg_bleu:.4f}")

# %% [markdown]
# ## 📊 Visualizing BLEU Scores

# %%
# Visualize BLEU scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(bleu_scores)), bleu_scores, alpha=0.7, color='skyblue')
plt.axhline(y=avg_bleu, color='red', linestyle='--', label=f'Mean BLEU: {avg_bleu:.4f}')
plt.xlabel('Example')
plt.ylabel('BLEU Score')
plt.title('BLEU Scores by Example')
plt.xticks(range(len(bleu_scores)), [f'Example {i+1}' for i in range(len(bleu_scores))])
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bleu_scores_by_example.png')
plt.show()

# %% [markdown]
# ## 🔍 Using a Different Language Pair
# 
# Let's try a different language pair. Here we'll translate from English to German:

# %%
# Define new language parameters
new_source_lang = 'en'
new_target_lang = 'de'
new_model_name = 'Helsinki-NLP/opus-mt-en-de'

# Load the new translation model
translator_de = pipeline(f"translation_{new_source_lang}_to_{new_target_lang}", model=new_model_name)
print(f"✅ Loaded model {new_model_name} for translation from {new_source_lang} to {new_target_lang}")

# Define an example sentence
source_sentence = "Artificial intelligence is revolutionizing the field of natural language processing."

# Translate the sentence
result = translator_de(source_sentence, max_length=100)
translated_text = result[0]['translation_text']

# Display results
print(f"English: {source_sentence}")
print(f"German: {translated_text}")

# %% [markdown]
# ## 🚀 Pushing Models to Hugging Face Hub
# 
# If you want to share your model with others or use it in production, you can push it to the Hugging Face Hub.

# %%
# Uncomment and run these cells to push a model to the Hugging Face Hub

# from huggingface_hub import notebook_login
# notebook_login()

# model_name = 'Helsinki-NLP/opus-mt-en-fr'
# hub_model_id = 'YOUR_USERNAME/YOUR_MODEL_NAME'  # Change this to your desired model ID

# # Load model and tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Push to Hub
# model.push_to_hub(hub_model_id)
# tokenizer.push_to_hub(hub_model_id)

# %% [markdown]
# ## 🎓 Conclusion
# 
# In this tutorial, we've learned how to:
# - Use pretrained transformer models for machine translation
# - Evaluate translation quality using BLEU score
# - Work with different language pairs
# - Push models to the Hugging Face Hub
# 
# This demonstrates the power of modern Neural Machine Translation (NMT) systems that leverage transformer architectures. Try experimenting with different language pairs and models to see how the quality varies!
# 
# ## 📌 Next Steps
# 
# 1. Try using different models for the same language pair
# 2. Compare different evaluation metrics
# 3. Fine-tune a model on domain-specific data
# 4. Create a simple web interface for your translator 
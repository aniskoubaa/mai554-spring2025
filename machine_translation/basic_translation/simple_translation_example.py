#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Translation Example - Using Pretrained Transformers
MAI554 Deep Learning Course
"""

from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
import nltk
import os

# Make sure NLTK data is properly downloaded
# First, check if the directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Check if punkt is already downloaded
punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
if not os.path.exists(punkt_dir):
    print("Downloading NLTK punkt tokenizer data...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
else:
    print("NLTK punkt tokenizer data already exists.")

# Also make sure punkt_tab is downloaded
print("Downloading additional NLTK resources needed for tokenization...")
nltk.download('all', download_dir=nltk_data_dir, quiet=False)

def main():
    # Load an English-to-French translation model
    print("🔄 Loading translation model...")
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

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
        )
    ]

    print("\n🔍 Translation Results:")
    print("=" * 80)
    
    total_bleu = 0
    
    for i, (source, reference) in enumerate(examples):
        # Translate the source text
        result = translator(source, max_length=100)
        translation = result[0]['translation_text']
        
        # Compute BLEU score
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            translation_tokens = nltk.word_tokenize(translation.lower())
            bleu = sentence_bleu([reference_tokens], translation_tokens)
        except LookupError:
            # If NLTK data is not found, try downloading it again
            print("⚠️ NLTK data not found. Attempting to download again...")
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
            # Try again after downloading
            reference_tokens = nltk.word_tokenize(reference.lower())
            translation_tokens = nltk.word_tokenize(translation.lower())
            bleu = sentence_bleu([reference_tokens], translation_tokens)
            
        total_bleu += bleu
        
        # Display results
        print(f"Example {i+1}:")
        print(f"Source:      {source}")
        print(f"Reference:   {reference}")
        print(f"Translation: {translation}")
        print(f"BLEU Score:  {bleu:.4f}")
        print("-" * 80)
    
    # Calculate and display average BLEU score
    avg_bleu = total_bleu / len(examples)
    print(f"📈 Average BLEU Score: {avg_bleu:.4f}")

if __name__ == "__main__":
    main() 
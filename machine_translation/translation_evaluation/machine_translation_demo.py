#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Translation Activity - Using Pretrained Transformers and BLEU Evaluation
MAI554 Deep Learning Course
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Make sure NLTK data is properly downloaded
# First, check if the directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Check if punkt and punkt_tab are already downloaded
punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab')

if not os.path.exists(punkt_dir):
    print("Downloading NLTK punkt tokenizer data...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
else:
    print("NLTK punkt tokenizer data already exists.")

# Also make sure punkt_tab is downloaded
nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
# For punkt_tab, we need to download the whole punkt package
print("Downloading additional NLTK resources needed for tokenization...")
nltk.download('all', download_dir=nltk_data_dir, quiet=False)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Machine Translation with Transformers')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-en-fr',
                        help='Model ID from Hugging Face Hub (default: Helsinki-NLP/opus-mt-en-fr)')
    parser.add_argument('--source', type=str, default='en',
                        help='Source language code (default: en)')
    parser.add_argument('--target', type=str, default='fr', 
                        help='Target language code (default: fr)')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of examples to translate (default: 10)')
    parser.add_argument('--save_to_hub', action='store_true',
                        help='Push model to Hugging Face Hub')
    parser.add_argument('--hub_model_id', type=str, default=None,
                        help='Model ID for Hugging Face Hub (required if save_to_hub=True)')
    parser.add_argument('--hub_token', type=str, default=None,
                        help='Hugging Face Hub token (required if save_to_hub=True)')
    return parser.parse_args()

def load_translation_model(model_name, source, target):
    """Load translation model from Hugging Face Hub"""
    try:
        # Try to use the generic pipeline if available
        task = f"translation_{source}_to_{target}"
        translator = pipeline(task, model=model_name)
        print(f"✅ Loaded model {model_name} for {task}")
        return translator
    except Exception as e:
        print(f"⚠️ Couldn't load generic pipeline: {e}")
        print("ℹ️ Loading model and tokenizer manually...")
        
        # Manual loading
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create a custom translation function
        def translate(text, max_length=100):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            outputs = model.generate(**inputs, max_length=max_length)
            return [{"translation_text": tokenizer.decode(output, skip_special_tokens=True)} 
                    for output in outputs]
        
        print(f"✅ Manually loaded model and tokenizer for {model_name}")
        return translate

def get_evaluation_dataset(source, target, num_examples=10):
    """Load a parallel dataset for evaluation"""
    print(f"📚 Loading evaluation dataset ({source}-{target})...")
    
    # Dictionary of example translations for common language pairs
    examples_by_lang_pair = {
        'en-fr': [
            ("Machine translation is transforming natural language processing.", 
             "La traduction automatique transforme le traitement du langage naturel."),
            ("The quick brown fox jumps over the lazy dog.",
             "Le rapide renard brun saute par-dessus le chien paresseux."),
            ("Artificial intelligence is the future of technology.",
             "L'intelligence artificielle est l'avenir de la technologie."),
            ("I love learning about deep learning and neural networks.",
             "J'adore apprendre sur l'apprentissage profond et les réseaux de neurones."),
            ("The weather is beautiful today.",
             "Le temps est magnifique aujourd'hui.")
        ],
        'en-de': [
            ("Machine translation is transforming natural language processing.", 
             "Die maschinelle Übersetzung verändert die Verarbeitung natürlicher Sprache."),
            ("The quick brown fox jumps over the lazy dog.",
             "Der schnelle braune Fuchs springt über den faulen Hund."),
            ("Artificial intelligence is the future of technology.",
             "Künstliche Intelligenz ist die Zukunft der Technologie."),
            ("I love learning about deep learning and neural networks.",
             "Ich liebe es, über Deep Learning und neuronale Netze zu lernen."),
            ("The weather is beautiful today.",
             "Das Wetter ist heute schön.")
        ],
        'en-es': [
            ("Machine translation is transforming natural language processing.", 
             "La traducción automática está transformando el procesamiento del lenguaje natural."),
            ("The quick brown fox jumps over the lazy dog.",
             "El rápido zorro marrón salta sobre el perro perezoso."),
            ("Artificial intelligence is the future of technology.",
             "La inteligencia artificial es el futuro de la tecnología."),
            ("I love learning about deep learning and neural networks.",
             "Me encanta aprender sobre el aprendizaje profundo y las redes neuronales."),
            ("The weather is beautiful today.",
             "El clima está hermoso hoy.")
        ]
    }
    
    # Create a key for the language pair
    lang_pair = f"{source}-{target}"
    
    # Try to find pre-defined examples for this language pair
    if lang_pair in examples_by_lang_pair:
        print(f"Using pre-defined examples for {lang_pair}")
        return examples_by_lang_pair[lang_pair][:num_examples]
    
    # Try to load from WMT datasets if supported
    try:
        # WMT datasets use different naming convention
        if source == 'en' and target in ['fr', 'de', 'cs', 'ru', 'hi']:
            # For these targets, WMT14 has TARGET-en configurations
            dataset = load_dataset("wmt14", f"{target}-{source}", split="test[:100]")
            # Swap the order since the dataset is in reverse
            return [(example['translation'][source], example['translation'][target]) for example in dataset[:num_examples]]
        else:
            # Try to find a suitable dataset for other language pairs
            try:
                dataset = load_dataset("opus_books", f"{source}-{target}", split="test[:100]")
                return [(example[source], example[target]) for example in dataset[:num_examples]]
            except Exception as e:
                print(f"⚠️ Could not load opus_books dataset: {e}")
                # Fall back to pre-defined examples
                print("⚠️ No dataset found for this language pair. Using example sentences.")
                if 'en-fr' in examples_by_lang_pair:
                    return examples_by_lang_pair['en-fr'][:num_examples]
                else:
                    return examples_by_lang_pair['en-de'][:num_examples]
    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}")
        # Fall back to pre-defined examples if available
        if lang_pair in examples_by_lang_pair:
            return examples_by_lang_pair[lang_pair][:num_examples]
        elif 'en-fr' in examples_by_lang_pair:
            return examples_by_lang_pair['en-fr'][:num_examples]
        else:
            return examples_by_lang_pair['en-de'][:num_examples]

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

def translate_and_evaluate(translator, examples, source, target):
    """Translate a set of examples and evaluate using BLEU score"""
    results = []
    bleu_scores = []
    
    print(f"🔄 Translating {len(examples)} examples from {source} to {target}...")
    
    for source_text, reference_text in tqdm(examples):
        # Translate the source text
        if callable(translator):
            # For custom translate function
            translation_output = translator(source_text)
            translated_text = translation_output[0]['translation_text']
        else:
            # For pipeline
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
    
    return results, bleu_scores

def visualize_bleu_scores(bleu_scores):
    """Visualize BLEU scores"""
    plt.figure(figsize=(10, 6))
    plt.hist(bleu_scores, bins=10, alpha=0.7, color='skyblue')
    plt.axvline(x=np.mean(bleu_scores), color='red', linestyle='--', 
                label=f'Mean BLEU: {np.mean(bleu_scores):.4f}')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of BLEU Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bleu_scores_distribution.png')
    print("📊 BLEU score distribution saved to 'bleu_scores_distribution.png'")

def push_to_hub(model_name, hub_model_id, hub_token):
    """Push model to Hugging Face Hub"""
    from huggingface_hub import notebook_login
    import torch
    
    if hub_token:
        notebook_login(hub_token)
    else:
        notebook_login()
    
    print(f"🚀 Pushing model to Hugging Face Hub as {hub_model_id}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save model and tokenizer to Hub
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    
    print(f"✅ Model successfully pushed to https://huggingface.co/{hub_model_id}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load translation model
    translator = load_translation_model(args.model, args.source, args.target)
    
    # Get evaluation dataset
    examples = get_evaluation_dataset(args.source, args.target, args.num_examples)
    
    # Translate and evaluate
    results, bleu_scores = translate_and_evaluate(translator, examples, args.source, args.target)
    
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
    
    # Visualize BLEU scores
    visualize_bleu_scores(bleu_scores)
    
    # Push to Hub if requested
    if args.save_to_hub:
        if not args.hub_model_id:
            print("❌ Error: hub_model_id is required when save_to_hub=True")
            return
        push_to_hub(args.model, args.hub_model_id, args.hub_token)

if __name__ == "__main__":
    main() 
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
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK data needed for tokenization"""
    try:
        # Make sure NLTK data is properly downloaded
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)

        # Check if punkt is already downloaded
        punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
        if not os.path.exists(punkt_dir):
            logger.info("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        else:
            logger.info("NLTK punkt tokenizer data already exists.")

        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK: {e}")
        return False

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
    parser.add_argument('--input_text', type=str, default=None,
                        help='Optional: Single sentence to translate (overrides dataset examples)')
    parser.add_argument('--save_to_hub', action='store_true',
                        help='Push model to Hugging Face Hub')
    parser.add_argument('--hub_model_id', type=str, default=None,
                        help='Model ID for Hugging Face Hub (required if save_to_hub=True)')
    parser.add_argument('--hub_token', type=str, default=None,
                        help='Hugging Face Hub token (required if save_to_hub=True)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output files (default: ./output)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def load_translation_model(model_name, source, target):
    """Load translation model from Hugging Face Hub"""
    try:
        # Try to use the generic pipeline if available
        task = f"translation_{source}_to_{target}"
        logger.info(f"Loading model {model_name} for {task}...")
        translator = pipeline(task, model=model_name)
        logger.info(f"✅ Loaded model {model_name} for {task}")
        return translator
    except Exception as e:
        logger.warning(f"⚠️ Couldn't load generic pipeline: {e}")
        logger.info("ℹ️ Loading model and tokenizer manually...")
        
        # Manual loading
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create a custom translation function
        def translate(text, max_length=100):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            outputs = model.generate(**inputs, max_length=max_length)
            return [{"translation_text": tokenizer.decode(output, skip_special_tokens=True)} 
                    for output in outputs]
        
        logger.info(f"✅ Manually loaded model and tokenizer for {model_name}")
        return translate

def get_evaluation_dataset(source, target, num_examples=10):
    """Load a parallel dataset for evaluation"""
    logger.info(f"📚 Loading evaluation dataset ({source}-{target})...")
    
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
        logger.info(f"Using pre-defined examples for {lang_pair}")
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
                logger.warning(f"⚠️ Could not load opus_books dataset: {e}")
                # Fall back to pre-defined examples
                logger.warning("⚠️ No dataset found for this language pair. Using example sentences.")
                if 'en-fr' in examples_by_lang_pair:
                    return examples_by_lang_pair['en-fr'][:num_examples]
                else:
                    return examples_by_lang_pair['en-de'][:num_examples]
    except Exception as e:
        logger.warning(f"⚠️ Error loading dataset: {e}")
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
        logger.warning("⚠️ NLTK data not found. Attempting to download again...")
        nltk.download('punkt')
        # Try again after downloading
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        return sentence_bleu([reference_tokens], hypothesis_tokens)

def translate_and_evaluate(translator, examples, source, target):
    """Translate a set of examples and evaluate using BLEU score"""
    results = []
    bleu_scores = []
    
    logger.info(f"🔄 Translating {len(examples)} examples from {source} to {target}...")
    
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

def translate_single_text(translator, text):
    """Translate a single text input"""
    # Translate the text
    if callable(translator):
        # For custom translate function
        translation_output = translator(text)
        translated_text = translation_output[0]['translation_text']
    else:
        # For pipeline
        translation_output = translator(text, max_length=100)
        translated_text = translation_output[0]['translation_text']
    
    return translated_text

def visualize_bleu_scores(bleu_scores, output_dir):
    """Visualize BLEU scores"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'bleu_scores_distribution.png')
    
    plt.figure(figsize=(10, 6))
    plt.hist(bleu_scores, bins=10, alpha=0.7, color='skyblue')
    plt.axvline(x=np.mean(bleu_scores), color='red', linestyle='--', 
                label=f'Mean BLEU: {np.mean(bleu_scores):.4f}')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of BLEU Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    logger.info(f"📊 BLEU score distribution saved to '{output_file}'")

def push_to_hub(model_name, hub_model_id, hub_token):
    """Push model to Hugging Face Hub"""
    try:
        from huggingface_hub import login
        import torch
        
        if hub_token:
            login(hub_token)
        else:
            login()
        
        logger.info(f"🚀 Pushing model to Hugging Face Hub as {hub_model_id}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save model and tokenizer to Hub
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        
        logger.info(f"✅ Model successfully pushed to https://huggingface.co/{hub_model_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error pushing model to Hub: {e}")
        return False

def save_results_to_file(results, avg_bleu, output_dir):
    """Save translation results to a file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'translation_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Translation Results:\n")
        f.write("=" * 80 + "\n")
        for i, result in enumerate(results):
            f.write(f"Example {i+1}:\n")
            f.write(f"Source:     {result['source']}\n")
            f.write(f"Reference:  {result['reference']}\n")
            f.write(f"Translation: {result['translation']}\n")
            f.write(f"BLEU Score: {result['bleu']:.4f}\n")
            f.write("-" * 80 + "\n")
        
        f.write(f"\nAverage BLEU Score: {avg_bleu:.4f}\n")
    
    logger.info(f"💾 Results saved to '{output_file}'")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Setup NLTK
    if not setup_nltk():
        logger.error("Failed to set up NLTK. Exiting.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load translation model
        translator = load_translation_model(args.model, args.source, args.target)
        
        # If input_text is provided, just translate that text
        if args.input_text:
            translated_text = translate_single_text(translator, args.input_text)
            print("\n🔠 Translation Result:")
            print(f"Source: {args.input_text}")
            print(f"Translation: {translated_text}")
            
            # Save to output file
            with open(os.path.join(args.output_dir, 'single_translation.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Source: {args.input_text}\n")
                f.write(f"Translation: {translated_text}\n")
            
            logger.info(f"💾 Single translation saved to '{os.path.join(args.output_dir, 'single_translation.txt')}'")
        else:
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
            visualize_bleu_scores(bleu_scores, args.output_dir)
            
            # Save results to file
            save_results_to_file(results, avg_bleu, args.output_dir)
        
        # Push to Hub if requested
        if args.save_to_hub:
            if not args.hub_model_id:
                logger.error("❌ Error: hub_model_id is required when save_to_hub=True")
                return 1
            if not push_to_hub(args.model, args.hub_model_id, args.hub_token):
                return 1
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
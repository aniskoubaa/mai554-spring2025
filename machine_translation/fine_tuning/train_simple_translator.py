#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Simple Machine Translation Model - Educational Script
MAI554 Deep Learning Course

This script demonstrates:
1. Loading a pre-trained transformer model for translation
2. Visualizing the model architecture
3. Downloading a dataset (English-French, French-English, English-Arabic, Arabic-English)
4. Fine-tuning the model on this dataset
5. Performing inference and evaluation with BLEU score
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torchinfo import summary
from nltk.translate.bleu_score import corpus_bleu
import nltk
from tqdm.auto import tqdm

# Make sure NLTK data is properly downloaded
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔄 Using device: {device}")

def display_model_architecture(model):
    """Display architecture details of the transformer model"""
    print("\n🏗️ Model Architecture:")
    print("=" * 80)
    
    # Get model summary
    model_summary = summary(
        model, 
        input_data=[
            torch.ones(1, 30, dtype=torch.long).to(device), 
            torch.ones(1, 30, dtype=torch.long).to(device)
        ],
        depth=2,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        verbose=0
    )
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {all_params - trainable_params:,}")
    
    # Display transformer blocks
    print("\nTransformer Components:")
    print("-" * 80)
    if hasattr(model, "encoder"):
        print(f"Encoder Layers: {len(model.encoder.block) if hasattr(model.encoder, 'block') else len(model.encoder.layers)}")
    if hasattr(model, "decoder"):
        print(f"Decoder Layers: {len(model.decoder.block) if hasattr(model.decoder, 'block') else len(model.decoder.layers)}")
    
    print("-" * 80)

def prepare_dataset(language_pair="en-fr", max_samples=5000):
    """Load and prepare a dataset for machine translation"""
    print(f"\n📚 Loading {language_pair} translation dataset...")
    
    src_lang, tgt_lang = language_pair.split('-')
    
    # Determine which dataset to use based on language pair
    if language_pair in ["en-fr", "fr-en"]:
        # Load English-French WMT dataset (smaller subset for educational purposes)
        raw_datasets = load_dataset("wmt14", "fr-en", split="train[:5000]")
        # Reorganize into train/validation/test
        train_val_test = raw_datasets.train_test_split(test_size=0.2)
        val_test = train_val_test["test"].train_test_split(test_size=0.5)
        datasets = DatasetDict({
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
        # Fix column names to match our expectation
        datasets = datasets.rename_column("translation.en", "en")
        datasets = datasets.rename_column("translation.fr", "fr")
        
        # If target is French-English, swap the source and target
        if language_pair == "fr-en":
            # We don't need to swap columns, just use them differently in training
            pass
    elif language_pair in ["en-ar", "ar-en"]:
        # Load English-Arabic dataset
        raw_datasets = load_dataset("opus100", "en-ar", split="train[:5000]")
        train_val_test = raw_datasets.train_test_split(test_size=0.2)
        val_test = train_val_test["test"].train_test_split(test_size=0.5)
        datasets = DatasetDict({
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
    else:
        raise ValueError(f"Unsupported language pair: {language_pair}")
    
    print(f"Train set size: {len(datasets['train'])}")
    print(f"Validation set size: {len(datasets['validation'])}")
    print(f"Test set size: {len(datasets['test'])}")
    
    # Display a few examples
    print("\nExample data:")
    for i, example in enumerate(datasets["train"][:3]):
        print(f"Example {i+1}:")
        print(f"  {src_lang}: {example[src_lang]}")
        print(f"  {tgt_lang}: {example[tgt_lang]}")
    
    return datasets, src_lang, tgt_lang

def preprocess_data(datasets, tokenizer, src_lang, tgt_lang, max_length=128):
    """Tokenize and prepare the dataset for training"""
    print("\n🔄 Preprocessing dataset...")
    
    def preprocess_function(examples):
        source_texts = examples[src_lang]
        target_texts = examples[tgt_lang]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(
            source_texts, 
            max_length=max_length, 
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_texts, 
                max_length=max_length, 
                truncation=True,
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing to all splits
    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    
    return tokenized_datasets

def compute_metrics(model, tokenizer, test_dataset, src_lang, tgt_lang):
    """Compute BLEU metrics on test set"""
    print("\n📊 Computing BLEU score on test set...")
    
    references = []
    hypotheses = []
    
    for example in tqdm(test_dataset, total=len(test_dataset)):
        source_text = example[src_lang]
        target_text = example[tgt_lang]
        
        # Tokenize input
        inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add to lists for BLEU computation
        references.append([nltk.word_tokenize(target_text)])
        hypotheses.append(nltk.word_tokenize(prediction))
    
    # Calculate corpus BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU Score: {bleu_score:.4f}")
    
    return bleu_score

def get_test_sentences(src_lang):
    """Get test sentences in the source language"""
    if src_lang == "en":
        return [
            "Hello, how are you doing today?",
            "Machine learning is transforming the world.",
            "I would like to learn a new language.",
            "Artificial intelligence can help solve complex problems."
        ]
    elif src_lang == "fr":
        return [
            "Bonjour, comment allez-vous aujourd'hui?",
            "L'apprentissage automatique transforme le monde.",
            "Je voudrais apprendre une nouvelle langue.",
            "L'intelligence artificielle peut aider à résoudre des problèmes complexes."
        ]
    elif src_lang == "ar":
        return [
            "مرحبا، كيف حالك اليوم؟",
            "التعلم الآلي يغير العالم.",
            "أود أن أتعلم لغة جديدة.",
            "يمكن للذكاء الاصطناعي المساعدة في حل المشكلات المعقدة."
        ]
    else:
        return ["Test sentence"] * 3

def main():
    # Choose language pair
    print("🌍 Welcome to the Simple Machine Translation Trainer!")
    print("This script demonstrates training a transformer-based translation model.")
    print("\nChoose a language pair:")
    print("1. English to French (en-fr)")
    print("2. French to English (fr-en)")
    print("3. English to Arabic (en-ar)")
    print("4. Arabic to English (ar-en)")
    
    choice = input("Enter choice (1/2/3/4) [default=1]: ").strip() or "1"
    
    if choice == "1":
        language_pair = "en-fr"
        model_name = "Helsinki-NLP/opus-mt-en-fr"
    elif choice == "2":
        language_pair = "fr-en" 
        model_name = "Helsinki-NLP/opus-mt-fr-en"
    elif choice == "3":
        language_pair = "en-ar"
        model_name = "Helsinki-NLP/opus-mt-en-ar"
    elif choice == "4":
        language_pair = "ar-en"
        model_name = "Helsinki-NLP/opus-mt-ar-en"
    else:
        print("Invalid choice. Using default: English to French")
        language_pair = "en-fr"
        model_name = "Helsinki-NLP/opus-mt-en-fr"
    
    print(f"\n🔄 Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Display model architecture
    display_model_architecture(model)
    
    # Prepare dataset
    datasets, src_lang, tgt_lang = prepare_dataset(language_pair, max_samples=1000)
    
    # Preprocess data
    tokenized_datasets = preprocess_data(datasets, tokenizer, src_lang, tgt_lang)
    
    # Define training arguments
    batch_size = 16
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results-{language_pair}",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        logging_dir=f"./logs-{language_pair}",
        logging_steps=50,
        save_steps=500,
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("\n🚀 Training the model...")
    trainer.train()
    
    # Save model
    model_save_path = f"./fine-tuned-translator-{language_pair}"
    print(f"\n💾 Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Test translations
    print(f"\n🔍 Testing translations from {src_lang} to {tgt_lang}:")
    
    test_sentences = get_test_sentences(src_lang)
    
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Source ({src_lang}): {sentence}")
        print(f"Translation ({tgt_lang}): {translation}")
        print("-" * 80)
    
    # Evaluate on test set
    bleu = compute_metrics(model, tokenizer, datasets["test"], src_lang, tgt_lang)
    
    print("\n✅ Training and evaluation complete!")
    print(f"Model saved to: {model_save_path}")
    print(f"Final BLEU score: {bleu:.4f}")
    
    # Plot training history if available
    if hasattr(trainer, "state.log_history") and trainer.state.log_history:
        history = trainer.state.log_history
        
        # Extract loss values
        train_loss = [x['loss'] for x in history if 'loss' in x]
        eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(np.arange(0, len(train_loss), len(train_loss)//len(eval_loss))[:len(eval_loss)], 
                eval_loss, 'o-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"training_loss_{language_pair}.png")
        plt.show()

if __name__ == "__main__":
    main() 
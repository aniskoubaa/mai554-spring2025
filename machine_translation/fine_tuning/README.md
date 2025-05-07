# 🔄 Translation Model Fine-Tuning Activity

This activity teaches you how to fine-tune pre-trained translation models on specific datasets.

## 🎯 Learning Objectives

- Learn how to fine-tune translation models on custom datasets
- Understand the process of adapting models to specific domains
- Evaluate fine-tuned models and compare to pre-trained versions
- Save and share fine-tuned models on Hugging Face Hub

## 🚀 Quick Start

```bash
# Run the fine-tuning script
python train_simple_translator.py
```

## 📋 Contents

- `train_simple_translator.py` - Script for fine-tuning translation models
- `machine_translation_tutorial.py` - Python script version of the tutorial
- `machine_translation_tutorial.ipynb` - Interactive notebook version

## 🌐 Supported Language Pairs

The script supports the following language pairs:
- English to French (en-fr)
- French to English (fr-en)
- English to Arabic (en-ar)
- Arabic to English (ar-en)

When you run the script, you'll be prompted to choose one of these language pairs.

## 🧪 Fine-tuning Process

Fine-tuning involves:
1. **Loading a pre-trained model** - Starting with existing knowledge
2. **Preparing a custom dataset** - Collecting parallel sentences
3. **Training the model** - Updating parameters on your data
4. **Evaluating performance** - Measuring improvement
5. **Sharing the model** - Publishing to Hugging Face Hub

## 💻 Example Code

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

# Train model
trainer.train()
```

## 🤖 Understanding the Transformer Architecture

In the code example above, the transformer architecture is used through the `AutoModelForSeq2SeqLM` class from Hugging Face. Let's connect the explanation to specific parts of the code:

### Encoder-Decoder Structure
- **Encoder-Decoder Loading**: `model = AutoModelForSeq2SeqLM.from_pretrained(model_name)` loads both the encoder (for input language) and decoder (for output language)
- **Tokenizer**: `tokenizer = AutoTokenizer.from_pretrained(model_name)` prepares text for the encoder

### How the Code Uses the Transformer
```python
# This loads the pre-trained transformer with encoder-decoder architecture
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# The tokenizer converts text to tokens the encoder can understand
tokenizer = AutoTokenizer.from_pretrained(model_name)

# During training, the model uses:
# - Self-attention in both encoder and decoder
# - Cross-attention from decoder to encoder output
# - Feed-forward networks and layer normalization
trainer.train()

# When generating translations, the decoder uses beam search
# (controlled by predict_with_generate=True in training_args)
```

### Key Components (Hidden in the Model)
- **Self-Attention**: Allows the model to focus on different parts of the input sequence
- **Cross-Attention**: Helps the decoder focus on relevant parts of the encoded input
- **Feed-Forward Networks**: Process the attention outputs
- **Layer Normalization**: Stabilizes the learning process

When you fine-tune the model using `trainer.train()`, you're adjusting the weights of this pre-trained encoder-decoder architecture to better translate your specific domain or language pair.

## 🧩 Understanding Hugging Face Parameters & Components

### Key Hugging Face Classes Explained
- **AutoModelForSeq2SeqLM**: Automatically loads the appropriate sequence-to-sequence model architecture based on the checkpoint name
- **AutoTokenizer**: Converts text to token IDs and back, handling special tokens, padding, etc.
- **Seq2SeqTrainingArguments**: Configures all training hyperparameters
- **Seq2SeqTrainer**: Handles the training loop, evaluation, and saving checkpoints
- **DataCollatorForSeq2Seq**: Efficiently batches examples with padding for sequence-to-sequence models

### Training Parameters Explained
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",                # Where to save model checkpoints
    evaluation_strategy="epoch",           # Evaluate after each epoch
    learning_rate=2e-5,                    # Small learning rate for fine-tuning
    per_device_train_batch_size=16,        # Batch size per GPU/CPU for training
    per_device_eval_batch_size=16,         # Batch size per GPU/CPU for evaluation
    weight_decay=0.01,                     # L2 regularization to prevent overfitting
    save_total_limit=3,                    # Keep only the 3 most recent checkpoints
    num_train_epochs=3,                    # Number of training epochs
    predict_with_generate=True,            # Use generation for evaluation metrics
)
```

### Dataset Processing in Hugging Face
When you see `tokenized_datasets["train"]` in the code, it refers to a dataset that has been:
1. Loaded from a source (often using `datasets.load_dataset()`)
2. Processed with the tokenizer to convert text to token IDs
3. Formatted with source and target languages properly aligned

Example of dataset preparation (not shown in the original code):
```python
# Load a dataset
from datasets import load_dataset
dataset = load_dataset("opus100", "en-fr")  # English-French parallel corpus

# Process the dataset
def preprocess_function(examples):
    inputs = [ex for ex in examples["en"]]  # Source language (English)
    targets = [ex for ex in examples["fr"]]  # Target language (French)
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing and create train/validation splits
tokenized_datasets = dataset.map(preprocess_function, batched=True)
```

## 🔍 Assignment Ideas

1. Fine-tune a model on a specialized domain (medical, legal, technical)
   - **Hint**: Find domain-specific parallel corpora like medical papers with translations or legal documents. Compare BLEU scores before and after fine-tuning on domain-specific test sets.

2. Compare performance of fine-tuned models across different data sizes
   - **Hint**: Create subsets of your training data (e.g., 100, 500, 1000, 5000 examples) and plot a learning curve showing how performance changes with more training data.

3. Create a multi-lingual fine-tuned model
   - **Hint**: Use a model like "Helsinki-NLP/opus-mt-mul-en" that already supports multiple languages, then fine-tune it with a mixture of language pairs to improve overall performance.

4. Implement techniques to prevent catastrophic forgetting during fine-tuning
   - **Hint**: Experiment with techniques like regularization, knowledge distillation, or mixture of experts. Try using a small percentage of original training data mixed with new domain data.

5. Compare the performance of different language pairs and analyze the challenges of each 
   - **Hint**: Select language pairs with varying degrees of similarity (e.g., English-French vs. English-Arabic) and analyze where the model struggles most (e.g., word order, morphology, idioms). 
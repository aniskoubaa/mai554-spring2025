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
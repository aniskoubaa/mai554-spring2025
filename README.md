# 🌍 Machine Translation with Transformers 🤖

This hands-on activity demonstrates how to use pre-trained transformer models from Hugging Face for Machine Translation and evaluate translations using the BLEU score metric.

## 🎯 Learning Objectives

- Learn how to use pre-trained Transformer models for machine translation
- Understand how to evaluate machine translation with BLEU scores
- Explore different language pairs and models
- Learn how to push models to Hugging Face Hub for sharing
- Understand the architecture of transformer-based translation models

## 📋 Prerequisites

- Python 3.8+
- Basic understanding of neural networks and NLP concepts
- Hugging Face account (for pushing models to Hub)

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd mai554-spring2025
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

For a simple demonstration of machine translation:

```bash
python simple_translation_example.py
```

This script translates a few English sentences to German and calculates BLEU scores.

## 📓 Jupyter Notebooks

This repository includes Jupyter notebooks that provide interactive tutorials:

```bash
# Convert Python files to Jupyter notebooks
./convert_to_notebook.sh
```

### Available Notebooks:

1. **machine_translation_tutorial.ipynb**: A step-by-step tutorial on machine translation with transformers.
2. **translation_model_architecture.ipynb**: A detailed exploration of translation model architectures.

## 🔍 Advanced Usage

The main script `machine_translation_demo.py` provides more features:

```bash
python machine_translation_demo.py --model Helsinki-NLP/opus-mt-en-de --source en --target de --num_examples 5
```

### Command Line Arguments

- `--model`: Hugging Face model ID (default: Helsinki-NLP/opus-mt-en-de)
- `--source`: Source language code (default: en)
- `--target`: Target language code (default: de)
- `--num_examples`: Number of examples to translate and evaluate (default: 10)
- `--save_to_hub`: Push the model to Hugging Face Hub
- `--hub_model_id`: Model ID for Hugging Face Hub (required if save_to_hub=True)
- `--hub_token`: Hugging Face Hub token (or you'll be prompted to log in)

### Examples

Translate from English to French:
```bash
python machine_translation_demo.py --model Helsinki-NLP/opus-mt-en-fr --source en --target fr
```

Translate more examples:
```bash
python machine_translation_demo.py --num_examples 20
```

Push a model to Hugging Face Hub:
```bash
python machine_translation_demo.py --save_to_hub --hub_model_id your-username/your-model-name
```

## 📊 Interpreting BLEU Scores

BLEU (Bilingual Evaluation Understudy) is a metric for evaluating machine translations:

- **0.0**: No match between the reference and translation
- **0.0-0.3**: Low quality translation
- **0.3-0.5**: Medium quality translation
- **0.5-0.7**: High quality translation
- **0.7-1.0**: Very high quality, potentially human-level translation

BLEU has limitations, so it should be used alongside human evaluation for a complete assessment.

## 🔄 Available Language Models

Some popular Hugging Face translation models:

- **Helsinki-NLP/opus-mt-en-de**: English to German
- **Helsinki-NLP/opus-mt-en-fr**: English to French
- **Helsinki-NLP/opus-mt-en-es**: English to Spanish
- **Helsinki-NLP/opus-mt-en-ru**: English to Russian
- **Helsinki-NLP/opus-mt-en-zh**: English to Chinese
- **Helsinki-NLP/opus-mt-ar-en**: Arabic to English

Explore more models at [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=translation).

## 🏗️ Translation Model Architectures

The `translation_model_architecture.py` script (and notebook) explores the internal workings of translation models:

- Detailed architecture of MarianMT and T5 models
- Visualization of the attention mechanism
- Exploration of tokenization and the translation process
- Comparison of different model types and performance

## 📝 Assignment Ideas

1. Compare translation quality between different pre-trained models for the same language pair
2. Implement another evaluation metric (like METEOR or TER) alongside BLEU
3. Fine-tune a translation model on domain-specific data
4. Create a simple web interface for your translator
5. Implement back-translation as a data augmentation technique
6. Analyze the attention patterns in transformer models for specific words or phrases

## 🔗 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [BLEU Score Paper](https://aclanthology.org/P02-1040.pdf)
- [Helsinki-NLP OPUS-MT Models](https://github.com/Helsinki-NLP/OPUS-MT-train)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details. 
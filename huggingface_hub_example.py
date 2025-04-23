#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hugging Face Hub Example - Pushing Translation Models
MAI554 Deep Learning Course

This script demonstrates how to push a pretrained translation model to
the Hugging Face Hub for sharing and reuse.
"""

import os
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import notebook_login, login


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Push translation model to Hugging Face Hub')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-en-de',
                        help='Model ID from Hugging Face Hub (default: Helsinki-NLP/opus-mt-en-de)')
    parser.add_argument('--hub_model_id', type=str, required=True,
                        help='Your Model ID for Hugging Face Hub (format: username/model-name)')
    parser.add_argument('--hub_token', type=str, default=None,
                        help='Hugging Face Hub token (or set HF_TOKEN env variable)')
    parser.add_argument('--model_card', action='store_true',
                        help='Generate a simple model card')
    return parser.parse_args()


def generate_model_card(source_model, hub_model_id):
    """Generate a simple model card README.md for the model"""
    model_card = f"""---
language:
- en
- de
tags:
- translation
- opus-mt
- helsinki-nlp
license: apache-2.0
---

# 🌍 Machine Translation Model: {hub_model_id}

This is a machine translation model originally trained by Helsinki-NLP as part of the OPUS-MT project,
specifically the `{source_model}` model.

## Model Description

This is a transformer-based machine translation model. It was originally trained by the Helsinki-NLP group
and is being shared with the community through this repo.

## Usage

```python
from transformers import pipeline

translator = pipeline("translation", model="{hub_model_id}")

# Translate from English to German
text = "Machine translation is transforming natural language processing."
translated = translator(text)[0]['translation_text']
print(translated)
```

## Training Data

This model was trained on the OPUS dataset, which includes various parallel corpora for machine translation.

## Evaluation

The model has been evaluated on various translation benchmarks and typically achieves good performance for
general domain text.

## Limitations

- The model may not perform well on domain-specific content
- Long sequences might be challenging
- Rare words or expressions might not translate accurately

## Citation

```bibtex
@inproceedings{{tiedemann-thottingal-2020-opus,
    title = "{{OPUS}}-{{MT}} {{--}} Building open translation services for the World",
    author = "Tiedemann, J{{\\\"o}}rg  and Thottingal, Santhosh",
    booktitle = "Proceedings of the 22nd Annual Conference of the European Association for Machine Translation",
    month = nov,
    year = "2020",
    address = "Lisboa, Portugal",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2020.eamt-1.61",
    pages = "479--480",
}}
```
"""
    return model_card


def push_to_hub(model_name, hub_model_id, hub_token=None, create_model_card=False):
    """Push model to Hugging Face Hub"""
    # Login to Hugging Face Hub
    if hub_token:
        login(token=hub_token)
    elif os.environ.get('HF_TOKEN'):
        login(token=os.environ.get('HF_TOKEN'))
    else:
        print("🔑 Please login to Hugging Face Hub...")
        notebook_login()
    
    print(f"🚀 Pushing model {model_name} to Hugging Face Hub as {hub_model_id}...")
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create model card if requested
    if create_model_card:
        model_card = generate_model_card(model_name, hub_model_id)
        with open("README.md", "w") as f:
            f.write(model_card)
        print("📝 Created model card README.md")
    
    # Push model and tokenizer to Hub
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    
    # Push model card if created
    if create_model_card:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=hub_model_id,
        )
        os.remove("README.md")  # Clean up
    
    print(f"✅ Model successfully pushed to https://huggingface.co/{hub_model_id}")
    print("\n📋 Usage example:")
    print(f"""
from transformers import pipeline

translator = pipeline("translation", model="{hub_model_id}")
text = "Machine translation is transforming natural language processing."
translated = translator(text)[0]['translation_text']
print(translated)
    """)


def main():
    """Main function"""
    args = parse_arguments()
    push_to_hub(args.model, args.hub_model_id, args.hub_token, args.model_card)


if __name__ == "__main__":
    main() 
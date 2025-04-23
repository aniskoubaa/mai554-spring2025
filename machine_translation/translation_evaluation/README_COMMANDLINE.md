# Command-Line Machine Translation Tool

This directory contains a command-line tool for machine translation using pre-trained transformer models from Hugging Face and evaluating translations using the BLEU score metric.

## Requirements

- Python 3.6+
- Required Python packages: transformers, datasets, nltk, numpy, matplotlib, tqdm
- Internet connection for downloading models and datasets

## Quick Start

To quickly translate a text from English to French:

```bash
./translate.sh --input "Hello, how are you today?"
```

## Usage Options

The `translate.sh` script accepts the following options:

```
Usage: ./translate.sh [options]

Options:
  -m, --model MODEL         Translation model to use (default: Helsinki-NLP/opus-mt-en-fr)
  -s, --source LANG         Source language code (default: en)
  -t, --target LANG         Target language code (default: fr)
  -n, --num NUM             Number of examples to translate (default: 10)
  -i, --input TEXT          Single text to translate (enclose in quotes)
  -o, --output DIR          Output directory (default: ./output)
  -v, --verbose             Enable verbose output
  -h, --help                Show this help message
```

## Examples

1. Translate a single sentence from English to French:
   ```bash
   ./translate.sh --input "Machine translation is transforming natural language processing."
   ```

2. Translate from English to German:
   ```bash
   ./translate.sh --source en --target de --input "Artificial intelligence is the future of technology."
   ```

3. Use a different model:
   ```bash
   ./translate.sh --model Helsinki-NLP/opus-mt-en-es --source en --target es
   ```

4. Translate and evaluate 5 examples using BLEU score:
   ```bash
   ./translate.sh --num 5
   ```

5. Save output to a specific directory:
   ```bash
   ./translate.sh --output ./my_translations
   ```

## Output

When translating a single input, the tool will:
- Display the translation in the terminal
- Save the translation to a file in the output directory

When evaluating with example translations, the tool will:
- Display each translation along with its BLEU score
- Calculate and display the average BLEU score
- Generate a histogram of BLEU scores
- Save all results to files in the output directory

## Available Models and Languages

The tool uses machine translation models from the Hugging Face Hub. Some commonly used models include:

- `Helsinki-NLP/opus-mt-en-fr`: English to French
- `Helsinki-NLP/opus-mt-en-de`: English to German
- `Helsinki-NLP/opus-mt-en-es`: English to Spanish
- `Helsinki-NLP/opus-mt-fr-en`: French to English
- `Helsinki-NLP/opus-mt-de-en`: German to English

For a complete list, visit: https://huggingface.co/models?pipeline_tag=translation

## Customizing the Python Script

For advanced users who need more control, you can directly use the Python script:

```bash
python machine_translation_demo.py --help
``` 
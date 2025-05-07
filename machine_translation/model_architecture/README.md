# 🏗️ Translation Model Architecture Activity

This activity explores the internal workings of transformer-based translation models.

## 🎯 Learning Objectives

- Understand the architecture of transformer models for translation
- Explore encoder-decoder structures in translation models
- Visualize attention mechanisms in transformers
- Learn about tokenization and embedding processes

## 🚀 Quick Start

```bash
# Run the model architecture exploration script
python translation_model_architecture.py

# Or explore the interactive notebook
jupyter notebook translation_model_architecture.ipynb
```

## 📋 Contents

- `translation_model_architecture.py` - Script exploring translation model architectures
- `translation_model_architecture.ipynb` - Interactive notebook with visualizations

## 🧩 Model Components

The activity explores:
- **Encoder-Decoder Architecture** - How information flows through the model
- **Attention Mechanisms** - How attention helps align words between languages
- **Tokenization** - How text is converted to tokens for the model
- **Embeddings** - How words are represented in high-dimensional space

## 💻 Example Code

```python
from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Print model architecture
print(model)

# Analyze number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

## 🔍 Assignment Ideas

1. Visualize attention patterns for specific words across languages
   - Hint: Use the `attention_weights` output from transformer models and matplotlib to create heatmaps
   ```python
   from transformers import MarianMTModel, MarianTokenizer
   import matplotlib.pyplot as plt
   import torch
   
   # Load model and tokenizer
   model_name = "Helsinki-NLP/opus-mt-en-de"
   model = MarianMTModel.from_pretrained(model_name)
   tokenizer = MarianTokenizer.from_pretrained(model_name)
   
   # Get input text and translate
   src_text = "The cat sat on the mat."
   inputs = tokenizer(src_text, return_tensors="pt")
   
   # Get attention weights
   with torch.no_grad():
       outputs = model(**inputs, output_attentions=True)
       attention_weights = outputs.encoder_attentions
   
   # Plot attention map for first head in first layer
   plt.figure(figsize=(10, 8))
   plt.imshow(attention_weights[0][0].numpy(), cmap='viridis')
   plt.xlabel('Target tokens')
   plt.ylabel('Source tokens')
   plt.title('Attention weights')
   plt.colorbar()
   plt.show()
   ```
   - Try comparing attention patterns for similar words across different language pairs

2. Compare architectures of different translation models
   - Hint: Analyze models like MarianMT, T5, and mBART to identify key structural differences
   ```python
   from transformers import MarianMTModel, T5Model, MBartModel
   
   # Load different models
   marian = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
   t5 = T5Model.from_pretrained("t5-small")
   mbart = MBartModel.from_pretrained("facebook/mbart-large-cc25")
   
   # Function to count parameters
   def count_parameters(model):
       return sum(p.numel() for p in model.parameters())
   
   # Compare model architectures
   models = {"MarianMT": marian, "T5": t5, "mBART": mbart}
   
   for name, model in models.items():
       print(f"Model: {name}")
       print(f"Parameters: {count_parameters(model):,}")
       print(f"Encoder layers: {len(model.get_encoder().layers)}")
       print(f"Hidden size: {model.config.hidden_size}")
       print("---")
   ```
   - Create a table comparing parameter counts, layer configurations, and performance metrics

3. Analyze how model size affects translation quality
   - Hint: Test translation quality using BLEU scores across models of varying sizes
   ```python
   from transformers import MarianMTModel, MarianTokenizer
   from sacrebleu import corpus_bleu
   import torch
   
   # Models of different sizes
   model_names = [
       "Helsinki-NLP/opus-mt-en-de",         # Base model
       "facebook/nllb-200-distilled-600M",    # Medium model
       "facebook/mbart-large-50-one-to-many-mmt"  # Large model
   ]
   
   test_sentences = ["Hello world", "The weather is nice", "I love programming"]
   reference_translations = ["Hallo Welt", "Das Wetter ist schön", "Ich liebe Programmieren"]
   
   results = {}
   
   for model_name in model_names:
       try:
           model = MarianMTModel.from_pretrained(model_name)
           tokenizer = MarianTokenizer.from_pretrained(model_name)
           
           # Get translations
           translations = []
           for text in test_sentences:
               inputs = tokenizer(text, return_tensors="pt")
               with torch.no_grad():
                   output = model.generate(**inputs)
               translation = tokenizer.decode(output[0], skip_special_tokens=True)
               translations.append(translation)
           
           # Calculate BLEU score
           bleu = corpus_bleu(translations, [reference_translations])
           
           # Count parameters
           params = sum(p.numel() for p in model.parameters())
           
           results[model_name] = {"params": params, "bleu": bleu.score}
       except Exception as e:
           print(f"Error with model {model_name}: {e}")
   
   # Print results
   for model, stats in results.items():
       print(f"Model: {model}")
       print(f"Parameters: {stats['params']:,}")
       print(f"BLEU score: {stats['bleu']:.2f}")
       print("---")
   ```
   - Plot the relationship between parameter count and translation accuracy

4. Explore how positional encoding works in translation models
   - Hint: Visualize the positional encodings using a heatmap
   ```python
   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   
   def get_positional_encoding(max_seq_len, d_model):
       """Create positional encodings"""
       pe = torch.zeros(max_seq_len, d_model)
       position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
       
       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       
       return pe
   
   # Create positional encodings
   max_seq_len = 50
   d_model = 512
   pe = get_positional_encoding(max_seq_len, d_model)
   
   # Visualize
   plt.figure(figsize=(15, 8))
   plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
   plt.xlabel('Embedding dimension')
   plt.ylabel('Position in sequence')
   plt.title('Positional Encoding')
   plt.colorbar()
   plt.show()
   
   # Extract a few positions to see the sinusoidal pattern
   positions_to_show = [0, 10, 20, 30, 40]
   plt.figure(figsize=(15, 5))
   for pos in positions_to_show:
       plt.plot(pe[pos, :100].numpy(), label=f'Position {pos}')
   plt.legend()
   plt.title('Positional Encoding Values')
   plt.xlabel('Embedding dimension')
   plt.ylabel('Value')
   plt.grid(True)
   plt.show()
   ```
   - Experiment with modifying the positional encoding and observe the effects on translation

5. Implement a mini transformer for translation
   - Hint: Create a simplified version with just 2 encoder and decoder layers
   ```python
   import torch
   import torch.nn as nn
   
   class SimpleTransformer(nn.Module):
       def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, 
                    num_encoder_layers=2, num_decoder_layers=2):
           super().__init__()
           
           # Embeddings
           self.src_embedding = nn.Embedding(src_vocab_size, d_model)
           self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
           self.positional_encoding = nn.Parameter(torch.zeros(1000, d_model))
           
           # Transformer components
           self.transformer = nn.Transformer(
               d_model=d_model,
               nhead=nhead,
               num_encoder_layers=num_encoder_layers,
               num_decoder_layers=num_decoder_layers,
               dim_feedforward=d_model*4
           )
           
           # Output layer
           self.output_layer = nn.Linear(d_model, tgt_vocab_size)
       
       def forward(self, src, tgt):
           # Create src and tgt masks
           src_mask = self.transformer.generate_square_subsequent_mask(src.size(0))
           tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0))
           
           # Embedding with positional encoding
           src_emb = self.src_embedding(src) + self.positional_encoding[:src.size(0), :]
           tgt_emb = self.tgt_embedding(tgt) + self.positional_encoding[:tgt.size(0), :]
           
           # Transformer forward pass
           output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
           
           # Project to vocabulary
           output = self.output_layer(output)
           
           return output
   
   # Example usage
   src_vocab_size = 5000  # Source language vocabulary size
   tgt_vocab_size = 6000  # Target language vocabulary size
   
   # Create the model
   model = SimpleTransformer(src_vocab_size, tgt_vocab_size)
   
   # Example inputs (batch_size=1, sequence_length)
   src = torch.randint(0, src_vocab_size, (10, 1))  # Source sentence
   tgt = torch.randint(0, tgt_vocab_size, (12, 1))  # Target sentence
   
   # Forward pass
   output = model(src, tgt)
   print(f"Output shape: {output.shape}")  # Expected: [tgt_len, batch_size, tgt_vocab_size]
   ```
   - Focus on understanding the core mechanisms rather than achieving state-of-the-art performance

6. Study the impact of context window size on translation quality
   - Hint: Experiment with truncating input sequences at different lengths
   ```python
   from transformers import MarianMTModel, MarianTokenizer
   from sacrebleu import corpus_bleu
   
   # Load model and tokenizer
   model_name = "Helsinki-NLP/opus-mt-en-de"
   model = MarianMTModel.from_pretrained(model_name)
   tokenizer = MarianTokenizer.from_pretrained(model_name)
   
   # Long text for translation
   long_text = """This is a very long paragraph that will be used to test how context window 
   size affects translation quality. We expect that as we truncate the text to shorter and 
   shorter lengths, some context will be lost, potentially affecting the translation quality, 
   especially for words that depend on earlier context. This is particularly important for 
   ambiguous terms, pronouns, or words that have different meanings based on context."""
   
   # Reference translation (you would normally have this professionally translated)
   reference = ["Your professional translation of the long text would go here"]
   
   # Test different truncation lengths
   truncation_lengths = [512, 256, 128, 64, 32]
   
   for max_length in truncation_lengths:
       # Tokenize with truncation
       inputs = tokenizer(long_text, return_tensors="pt", max_length=max_length, truncation=True)
       
       # Generate translation
       outputs = model.generate(**inputs)
       translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
       
       # Calculate BLEU score (if you have a reference translation)
       if reference:
           bleu = corpus_bleu([translation], [reference])
           print(f"Max length: {max_length}, BLEU: {bleu.score:.2f}")
       
       # Print first 100 chars of translation for manual comparison
       print(f"Max length: {max_length}, Translation: {translation[:100]}...")
       print("---")
   ```
   - Analyze which types of sentences suffer most from limited context windows

7. Examine cross-lingual knowledge transfer
   - Hint: Test how well a model trained on one language pair performs on a related language pair
   ```python
   from transformers import MarianMTModel, MarianTokenizer
   import torch
   
   # Load model trained on one language pair (e.g., English-German)
   model_name = "Helsinki-NLP/opus-mt-en-de"
   model = MarianMTModel.from_pretrained(model_name)
   tokenizer = MarianTokenizer.from_pretrained(model_name)
   
   # Test sentences in English
   test_sentences = [
       "Hello, how are you?",
       "The cat is sleeping on the couch.",
       "I would like to order a coffee, please."
   ]
   
   # Target languages (German is what it was trained for, Dutch is related)
   target_languages = {
       "German": "de",  # What the model was trained for
       "Dutch": "nl"    # Related language (Germanic, similar to German)
   }
   
   for lang_name, lang_code in target_languages.items():
       print(f"Translations to {lang_name}:")
       for text in test_sentences:
           # Try to set the target language if the model supports it
           try:
               if hasattr(tokenizer, "set_tgt_lang_id"):
                   tokenizer.set_tgt_lang_id(lang_code)
               
               # Tokenize and translate
               inputs = tokenizer(text, return_tensors="pt")
               with torch.no_grad():
                   outputs = model.generate(**inputs)
               
               translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
               print(f"  '{text}' → '{translation}'")
           except Exception as e:
               print(f"  Error translating to {lang_name}: {e}")
       print("---")
   
   # Alternative: Visualize embeddings using PCA or t-SNE
   from sklearn.decomposition import PCA
   import matplotlib.pyplot as plt
   
   # Get embeddings for words in different languages
   word_pairs = [
       ("house", "haus", "huis"),  # English, German, Dutch
       ("water", "wasser", "water"),
       ("book", "buch", "boek"),
       ("day", "tag", "dag")
   ]
   
   # Extract embeddings from model
   embeddings = []
   labels = []
   
   for en, de, nl in word_pairs:
       for word, lang in zip([en, de, nl], ["en", "de", "nl"]):
           # Get token ID
           token_id = tokenizer.encode(word, add_special_tokens=False)[0]
           # Get embedding from the model
           embedding = model.get_input_embeddings()(torch.tensor([token_id]))
           
           embeddings.append(embedding.detach().numpy().flatten())
           labels.append(f"{word} ({lang})")
   
   # Apply PCA for visualization
   pca = PCA(n_components=2)
   embeddings_2d = pca.fit_transform(embeddings)
   
   # Plot embeddings
   plt.figure(figsize=(10, 8))
   for i, (label, (x, y)) in enumerate(zip(labels, embeddings_2d)):
       plt.scatter(x, y, label=label)
       plt.text(x, y, label)
   
   plt.title("Cross-lingual embeddings visualization")
   plt.xlabel("PCA dimension 1")
   plt.ylabel("PCA dimension 2")
   plt.grid(True)
   plt.show()
   ```
   - Visualize embedding spaces to detect clustering of related languages 
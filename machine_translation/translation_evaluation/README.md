# 📊 Translation Evaluation Activity

This activity teaches you how to evaluate machine translation quality using metrics like BLEU score.

## 🎯 Learning Objectives

- Understand machine translation evaluation metrics
- Calculate and interpret BLEU scores
- Compare translations from different models
- Learn about the strengths and limitations of automatic evaluation

## 🚀 Quick Start

```bash
# Run the translation demo with evaluation
python machine_translation_demo.py
```

## 📋 Contents

- `machine_translation_demo.py` - Script for translating and evaluating translations
- `bleu_scores_distribution.png` - Visualization of BLEU score distributions

## 📈 Understanding BLEU Scores

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating machine translations:

- **0.0**: No match between reference and translation
- **0.0-0.3**: Low quality translation
- **0.3-0.5**: Medium quality translation
- **0.5-0.7**: High quality translation
- **0.7-1.0**: Very high quality, potentially human-level translation

## 💻 Example Code

```python
from nltk.translate.bleu_score import sentence_bleu

reference = ["The cat is sitting on the mat."]
candidate = "The cat sits on the mat."

# Calculate BLEU score
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score:.2f}")
```

## 🔍 Assignment Ideas

1. Implement additional evaluation metrics (METEOR, TER, ROUGE)
2. Create visualizations comparing different models' BLEU scores
3. Design a human evaluation study and compare with BLEU scores
4. Analyze BLEU score differences across language pairs 
"""
Evaluate the baseline NLLB-200-distilled-600M model on English–Irish translation
Author: Anonymous (LREC submission)
"""

import os
import json
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Base model (no fine-tuning)
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Domain-specific evaluation datasets
EVAL_FOLDERS = {
    "general": "data/test_3domains_embedding/general",
    "medical": "data/test_3domains_embedding/medical",
    "wikinews": "data/test_3domains_embedding/wikinews",
    "legal": "data/test_3domains_embedding/legal",
}

# Output files
OUTPUT_TXT = "results/bleu_scores_baseline_nllb200_600m.txt"
OUTPUT_JSON = "results/bleu_scores_baseline_nllb200_600m.json"

# ---------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
print(f"\nLoading baseline model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
model.eval()

tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "gle_Latn"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_texts(file_path: str):
    """Load non-empty lines from a text file."""
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def translate_batch(model, tokenizer, sentences, batch_size=32, max_length=256):
    """Translate a list of sentences in batches."""
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)
    results = {}

    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        for domain, eval_path in EVAL_FOLDERS.items():
            print(f"\n--- Evaluating {domain} domain ---")
            fout.write(f"\n--- Evaluating {domain} domain ---\n")

            en_path = os.path.join(eval_path, "english_clean.txt")
            ga_path = os.path.join(eval_path, "irish_clean.txt")

            english_texts = load_texts(en_path)
            irish_texts = load_texts(ga_path)
            assert len(english_texts) == len(irish_texts), f"Length mismatch in {domain}"

            # Translate and compute BLEU
            translations = translate_batch(model, tokenizer, english_texts)
            bleu = sacrebleu.corpus_bleu(translations, [irish_texts])
            results[domain] = bleu.score

            print(f"{domain}: BLEU = {bleu.score:.2f}")
            fout.write(f"{domain}: BLEU = {bleu.score:.2f}\n")

    # Save JSON results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    print("\nAll evaluations complete.")
    print(f"Results saved to:\n- {OUTPUT_TXT}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

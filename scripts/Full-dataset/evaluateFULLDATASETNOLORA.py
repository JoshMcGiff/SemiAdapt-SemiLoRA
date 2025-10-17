"""
Evaluation script for the fully fine-tuned (no-LoRA) model trained on the combined multi-domain English–Irish dataset.
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

# Model path (fully fine-tuned on all domains)
MODEL_PATH = "models/nllb_fulldataset_finetuned/final_model"

# Evaluation dataset (combined domains)
EVAL_FOLDER = "data/test_fulldataset"
EN_FILE = "all_domains_english.txt"
GA_FILE = "all_domains_irish.txt"

# Output files
OUTPUT_TXT = "results/bleu_scores_fulldataset_nolora.txt"
OUTPUT_JSON = "results/bleu_scores_fulldataset_nolora.json"

# ---------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_model(model_path: str):
    """Load a fully fine-tuned model (no adapters)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


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

    print("\n--- Evaluating full dataset fine-tuned model ---")

    # Load data
    en_path = os.path.join(EVAL_FOLDER, EN_FILE)
    ga_path = os.path.join(EVAL_FOLDER, GA_FILE)
    english_texts = load_texts(en_path)
    irish_texts = load_texts(ga_path)
    assert len(english_texts) == len(irish_texts), "Length mismatch in test dataset"

    # Load model and tokenizer
    tokenizer, model = load_model(MODEL_PATH)

    # Translate and compute BLEU
    translations = translate_batch(model, tokenizer, english_texts)
    bleu = sacrebleu.corpus_bleu(translations, [irish_texts])
    results["full_dataset"] = bleu.score

    print(f"BLEU = {bleu.score:.2f}")

    # Write to file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        fout.write(f"--- Full Dataset Evaluation ---\nBLEU = {bleu.score:.2f}\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print("\nAll evaluations complete.")
    print(f"Results saved to:\n- {OUTPUT_TXT}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

"""
Evaluation script for the fully fine-tuned (no-LoRA) model trained on the combined four-domain English–Irish dataset.
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

# Model trained on all domains combined (fully fine-tuned)
MODEL_PATH = "models/nllb_fulldataset_finetuned/final_model"

# Domain-specific test sets
EVAL_FOLDERS = {
    "general": "data/test/general",
    "medical": "data/test/medical",
    "wikinews": "data/test/wikinews",
    "legal": "data/test/legal",
}

# Output files
OUTPUT_TXT = "results/bleu_scores_fulldataset_nolora_perdomain.txt"
OUTPUT_JSON = "results/bleu_scores_fulldataset_nolora_perdomain.json"

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

    print("\n--- Evaluating Full-Dataset Fine-Tuned Model Across Domains ---")

    # Load model once (used for all domains)
    tokenizer, model = load_model(MODEL_PATH)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        for domain, eval_path in EVAL_FOLDERS.items():
            print(f"\n--- Evaluating {domain} ---")
            fout.write(f"\n--- Evaluating {domain} ---\n")

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

    # Save results as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # Clean up
    del model
    torch.cuda.empty_cache()

    print("\nAll evaluations complete.")
    print(f"Results saved to:\n- {OUTPUT_TXT}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

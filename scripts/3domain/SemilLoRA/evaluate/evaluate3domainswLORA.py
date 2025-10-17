"""
Evaluation script for using LoRA adapters for domain adaptation on English–Irish translation.
Author: Anonymous (LREC submission)
"""

import os
import json
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_MODEL = "models/nllb_3domain_general/final_model"

# map domain names to model checkpoints
NLLB_MODELS = {
    "general": "models/nllb_3domain_general/final_model",
    "medical": "models/nllb_3domain_medical/final_model",
    "wikinews": "models/nllb_3domain_wikinews/final_model",
    "legal": "models/nllb_3domain_legal/final_model",
}

# evaluation data folders
EVAL_FOLDERS = {
    "general": "data/test_3domains/general",
    "medical": "data/test_3domains/medical",
    "wikinews": "data/test_3domains/wikinews",
    "legal": "data/test_3domains/legal",
}

OUTPUT_TXT = "results/bleu_scores.txt"
OUTPUT_JSON = "results/bleu_scores.json"

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
def load_tokenizer_model(model_path: str, base_model: str = None):
    """
    Load a tokenizer and model. If the model uses LoRA adapters,
    attach them to the specified base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"

    # for general domain use fine-tuned general base model, otherwise load LoRA adapters for domain adaptation
    if base_model and model_path != base_model:
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model, local_files_only=True)
        model = PeftModel.from_pretrained(base, model_path, local_files_only=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    model.to(device)
    model.eval()
    return tokenizer, model


def translate_batch(model, tokenizer, sentences, max_length=256, batch_size=32):
    """Translate a list of sentences in batches."""
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend([d.strip() for d in decoded])
    return translations


def load_texts(path):
    """Load lines from a text file, stripping whitespace and skipping blanks."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)
    results = {}

    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        for domain, model_path in NLLB_MODELS.items():
            print(f"\n--- Evaluating {domain} ---")
            fout.write(f"\n--- Evaluating {domain} ---\n")

            eval_path = EVAL_FOLDERS[domain]
            en_texts = load_texts(os.path.join(eval_path, "english_clean.txt"))
            ga_texts = load_texts(os.path.join(eval_path, "irish_clean.txt"))
            assert len(en_texts) == len(ga_texts), f"Length mismatch in {domain}"

            tokenizer, model = load_tokenizer_model(model_path, base_model=BASE_MODEL)
            translations = translate_batch(model, tokenizer, en_texts, batch_size=16)

            bleu = sacrebleu.corpus_bleu(translations, [ga_texts])
            results[domain] = bleu.score

            print(f"{domain}: BLEU = {bleu.score:.2f}")
            fout.write(f"{domain}: BLEU = {bleu.score:.2f}\n")

            # clean up as sometimes GPU memory gets full/memory doesn't get released
            del model
            torch.cuda.empty_cache()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    print("\nAll evaluations complete.")
    print(f"Results saved to:\n- {OUTPUT_TXT}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

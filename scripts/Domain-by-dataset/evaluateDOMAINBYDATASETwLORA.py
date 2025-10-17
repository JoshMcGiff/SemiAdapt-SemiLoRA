"""
Evaluation script for domain-by-dataset LoRA fine-tuned NLLB models
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

# Base model (fully fine-tuned on general domain)
BASE_MODEL = "models/nllb_domainbydataset_general/final_model"

# Model checkpoints: general = full fine-tuned, others = LoRA adapters
NLLB_MODELS = {
    "general": "models/nllb_domainbydataset_general/final_model",
    "medical": "models/nllb_domainbydataset_lora_medical/final_model",
    "wikinews": "models/nllb_domainbydataset_lora_wikinews/final_model",
    "legal": "models/nllb_domainbydataset_lora_legal/final_model",
}

# Domain-specific evaluation datasets
EVAL_FOLDERS = {
    "general": "data/test_dataset_domain/general",
    "medical": "data/test_dataset_domain/medical",
    "wikinews": "data/test_dataset_domain/wikinews",
    "legal": "data/test_dataset_domain/legal",
}

# Output files
OUTPUT_TXT = "results/bleu_scores_domainbydataset_lora.txt"
OUTPUT_JSON = "results/bleu_scores_domainbydataset_lora.json"

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
def load_full_model(model_path: str):
    """Load fully fine-tuned model (no adapters)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_lora_model(base_model: str, adapter_path: str):
    """Load base model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"

    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_texts(path: str):
    """Load non-empty lines from a text file."""
    with open(path, encoding="utf-8") as f:
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
        translations.extend([d.strip() for d in decoded])
    return translations

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)
    results = {}

    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        for domain, model_path in NLLB_MODELS.items():
            print(f"\n--- Evaluating {domain} ---")
            fout.write(f"\n--- Evaluating {domain} ---\n")

            eval_path = EVAL_FOLDERS[domain]
            en_path = os.path.join(eval_path, "english_clean.txt")
            ga_path = os.path.join(eval_path, "irish_clean.txt")

            english_texts = load_texts(en_path)
            irish_texts = load_texts(ga_path)
            assert len(english_texts) == len(irish_texts), f"Length mismatch in {domain}"

            # for general domain use fine-tuned general base model, otherwise load LoRA adapters for domain adaptation
            if domain == "general":
                tokenizer, model = load_full_model(model_path)
            else:
                tokenizer, model = load_lora_model(BASE_MODEL, model_path)

            # Translate and compute BLEU
            translations = translate_batch(model, tokenizer, english_texts, batch_size=16)
            bleu = sacrebleu.corpus_bleu(translations, [irish_texts])
            results[domain] = bleu.score

            print(f"{domain}: BLEU = {bleu.score:.2f}")
            fout.write(f"{domain}: BLEU = {bleu.score:.2f}\n")

            # clean up as sometimes GPU memory gets full/memory doesn't get released
            del model
            torch.cuda.empty_cache()

    # Save results as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    print("\nAll evaluations complete.")
    print(f"Results saved to:\n- {OUTPUT_TXT}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

"""
Tokenize English–Irish parallel text using NLLB-200 tokenizer.
Author: Anonymous (LREC submission)
"""

from datasets import Dataset, DatasetDict
from transformers import NllbTokenizer
import os


def tokenize_dataset(en_file, ga_file, output_dir, num_cpus=8):
    """Tokenize English–Irish parallel data for NLLB fine-tuning."""
    # Load raw texts
    with open(en_file, encoding="utf-8") as f_en, open(ga_file, encoding="utf-8") as f_ga:
        english_sentences = [line.strip() for line in f_en]
        irish_sentences = [line.strip() for line in f_ga]

    assert len(english_sentences) == len(irish_sentences), "Mismatched number of lines!"

    raw_dataset = Dataset.from_dict({
        "source": english_sentences,
        "target": irish_sentences
    })

    dataset_splits = raw_dataset.train_test_split(seed=42)
    train_raw = dataset_splits["train"]

    # Load NLLB tokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = NllbTokenizer.from_pretrained(model_name)

    # Set language codes
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"

    # Tokenize function
    def tokenize_batch(batch):
        return tokenizer(
            batch["source"],
            text_target=batch["target"],
            truncation=True,
            max_length=512
        )

    # Tokenize dataset
    tokenized_train = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_raw.column_names,
        num_proc=num_cpus
    )

    # Set torch format
    columns = ["input_ids", "attention_mask", "labels"]
    tokenized_train.set_format("torch", columns=columns)

    # Save processed dataset
    processed_dataset = DatasetDict({
        "train": tokenized_train,
    })
    processed_dataset.save_to_disk(output_dir)
    print(f"✅ Tokenized dataset saved to: {output_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tokenize_dataset(
        en_file="./train_dataset_domain/wikinews/english_clean.txt",
        ga_file="./train_dataset_domain/wikinews/irish_clean.txt",
        output_dir="nllb_english-irish_tokenized_finetune_DOMAINBYDATASET_wikinews",
        num_cpus=8
    )

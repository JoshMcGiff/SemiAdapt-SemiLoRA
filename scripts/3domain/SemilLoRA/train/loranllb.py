"""
Example LoRA fine-tuning script for NLLB-200-distilled-600M on a particular domain.
Author: Anonymous (LREC submission)
"""

import os
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler

from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load as load_metric


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logging(log_file: str = "logs/training.log"):
    """Configure rotating file and console logging."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------
# Main fine-tuning function
# ---------------------------------------------------------------------
def main():
    logger.info("Initializing LoRA fine-tuning for the Legal domain...")

    # -----------------------------------------------------------------
    # Model and tokenizer setup
    # -----------------------------------------------------------------
    base_model_path = "models/nllb_4domains_withbaselinedata_general/final_model"
    tokenizer = NllbTokenizer.from_pretrained(base_model_path)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "gle_Latn"
    logger.info(f"Loaded tokenizer from: {base_model_path}")

    # -----------------------------------------------------------------
    # Metric setup
    # -----------------------------------------------------------------
    sacrebleu = load_metric("sacrebleu")

    def compute_metrics(eval_preds):
        """Compute BLEU score for evaluation."""
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu = result["score"]
        logger.info(f"🟦 BLEU: {bleu:.2f}")
        return {"bleu": bleu}

    # -----------------------------------------------------------------
    # Dataset loading
    # -----------------------------------------------------------------
    dataset_path = "data/nllb_english-irish_tokenized_4domains_withbaselinedata_legal"
    logger.info(f"Loading tokenized dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    # subset for quick debugging
    # train_dataset = train_dataset.select(range(2000))

    # -----------------------------------------------------------------
    # Base model loading
    # -----------------------------------------------------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path).to("cuda")

    # -----------------------------------------------------------------
    # LoRA configuration
    # -----------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA adapters applied successfully.")

    # -----------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------
    output_dir = "models/nllb_domainbydataset_legal_lora"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",  # Use "steps" if enabling validation
        save_steps=5000,
        logging_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        fp16_backend="auto",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
    )

    # -----------------------------------------------------------------
    # Trainer setup
    # -----------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest"),
        compute_metrics=compute_metrics,
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    logger.info("Starting LoRA fine-tuning on Legal domain dataset...")
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    model.save_pretrained(output_dir)

    logger.info(f"✅ Training complete. Model saved to: {final_model_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger = setup_logging("logs/train_domainbydataset_legal_lora.log")
    logger.info("Fine-tuning NLLB-200 for English → Irish (Legal Domain, LoRA)")
    main()

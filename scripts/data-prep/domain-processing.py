"""
Automatically assign domain labels to English sentences using Facebook's BART-MNLI zero-shot classification model.
Author: Anonymous (LREC submission)
"""

import os
import csv
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline


# ---------------------------------------------------------------------
# Domain Labeling Function
# ---------------------------------------------------------------------
def assign_domain_labels(in_path: str, out_path: str, batch_size: int = 32, generic_threshold: float = 0.45):
    """Classify English sentences into domains using BART-MNLI zero-shot model."""
    print("📂 Loading dataset...")
    df = pd.read_csv(
        in_path,
        sep="\t",
        header=None,
        names=["en", "ga"],
        dtype=str,
        keep_default_na=False,
        engine="python",
        quoting=3,
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    print(f"✅ Loaded {len(df):,} rows")

    # -----------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------
    print("⚙️  Loading zero-shot classifier (this may take ~30s)...")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )
    print("✅ Model loaded and ready!")

    # Candidate domain labels
    candidate_labels = ["medical/COVID-19", "legal", "wiki/news", "general"]
    template = "This example is {}."

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    labels, scores = [], []
    total_batches = (len(df) + batch_size - 1) // batch_size
    print("🚀 Running domain classification...")

    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Classifying"):
        batch = df["en"].iloc[i:i + batch_size].tolist()
        results = clf(
            batch,
            candidate_labels=candidate_labels,
            hypothesis_template=template,
            multi_label=False,
            batch_size=batch_size,
            truncation=True,
        )

        # Normalize output (sometimes a single dict, sometimes list)
        if isinstance(results, dict):
            results = [results]

        for res in results:
            top_label = res["labels"][0]
            top_score = float(res["scores"][0])

            # Confidence threshold for "general" label 
            # if top_score < generic_threshold:
            #     top_label = "general"

            labels.append(top_label)
            scores.append(round(top_score, 4))

    # -----------------------------------------------------------------
    # Save labeled dataset
    # -----------------------------------------------------------------
    df["label"] = labels
    df["label_score"] = scores

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
        encoding="utf-8-sig",
    )

    print(f"✅ Done! Wrote {len(df):,} labeled rows to: {out_path}")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign domain labels to English–Irish sentence pairs using BART-MNLI."
    )
    parser.add_argument("--in", dest="in_path", default="data/combined.tsv", help="Input TSV file path.")
    parser.add_argument("--out", dest="out_path", default="data/combined_4domains_labelled.tsv", help="Output TSV file path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--generic_threshold", type=float, default=0.45, help="Threshold below which sentences default to 'general' domain.")

    args = parser.parse_args()
    assign_domain_labels(args.in_path, args.out_path, args.batch_size, args.generic_threshold)

# SemiAdapt-SemiLoRA
Repository for the paper: "SemiAdapt and SemiLoRA: Efficient Domain Adaptation for Transformer-based Low-Resource Language Translation with a Case Study on Irish"
---

## 🧩 Overview

This repository contains code and models for **SemiAdapt** and **SemiLoRA**, two efficient domain adaptation methods for Transformer-based Neural Machine Translation (NMT).  
Our approaches target **low-resource language translation**, with a case study on **Irish**.  

Traditional full-model fine-tuning of large multilingual models (billions of parameters) is computationally expensive.  
**SemiAdapt** and **SemiLoRA** address this challenge by improving **parameter-efficient fine-tuning (PEFT)** techniques such as LoRA. Our methods achieve strong adaptation while reducing memory and training cost.

---

## 🚀 Key Contributions

- **SemiAdapt**: A lightweight fine-tuning approach that enhances domain adaptation efficiency and can outperform full fine-tuning.
- **SemiLoRA**: A parameter-efficient variant that matches or exceeds full-model fine-tuning performance.
- **Embedding-based inference** for improved performance on large, noisy corpora and efficient inference overall.
- Comprehensive experiments on **Irish translation**, with released open models for reproducibility.

---




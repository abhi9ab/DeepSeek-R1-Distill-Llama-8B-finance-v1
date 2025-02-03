# abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1 (Fine-tuned with LoRA and unsloth)

**You can access the model on [Hugging Face Hub](https://huggingface.co/abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1)**

![Hugging face for deepseek 8b llama](https://github.com/user-attachments/assets/f4556c59-6732-4a27-b644-0b2963cfaf15)

---

**This repository contains the fine-tuned version of the unsloth/DeepSeek-R1-Distill-Llama-8B model for financial tasks, named abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1. The fine-tuning was performed using LoRA (Low-Rank Adaptation) on a subset of the Josephgflowers/Finance-Instruct-500k dataset.**

---

## Model Details

- Base Model: unsloth/DeepSeek-R1-Distill-Llama-8B
- Fine-Tuning Method: LoRA
- Fine-Tuned Model: abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1
- Dataset: Josephgflowers/Finance-Instruct-500k (reduced to 5k JSONL entries)
- Platform: Free-tier Kaggle Notebook
- Libraries: Hugging Face Transformers, Unsloth, Weights and Biases (wandb), pytorch

![wandb for deepseek 8b llama](https://github.com/user-attachments/assets/6456b193-889e-4d66-95ab-a7b3e6ee0fe8)


---

## Objective

The goal of this model is to enhance the base model's performance on financial tasks by fine-tuning it on a specialized financial dataset. Using LoRA, this model has been optimized for low-rank adaptation, allowing efficient fine-tuning with fewer resources.

---

## Dataset

The model was fine-tuned on a subset of the Finance-Instruct-500k dataset from Hugging Face, specifically reduced to 5,000 JSONL entries for the fine-tuning process. This dataset contains financial questions and answers, providing a rich set of examples for training the model.

---

## Setup and Usage

**Requirements:**
- Python >= 3.10
- Hugging Face Transformers library
- Google Colab/Kaggle Notebook (for free-tier usage)
- PyTorch
- Unsloth
- Weights and Biases (wandb)

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1.git
cd DeepSeek-R1-Distill-Llama-8B-finance-v1
```
2. Follow the instructions mentioned in the notebook

---

## Notes

- This fine-tuning was performed on the free-tier of Kaggle Notebook, so training time and available resources are limited.
- Ensure that your runtime in Colab/Kaggle is set to a GPU environment to speed up the training process.
- The reduced 5k dataset is a smaller sample for experimentation. You can scale this up depending on your needs and available resources.

---

## License

[Apache License](https://github.com/abhi9ab/DeepSeek-R1-Distill-Llama-8B-finance-v1/blob/main/LICENSE)

---

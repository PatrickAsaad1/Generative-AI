<div align="center">

**A custom-trained language model built from scratch — fine-tuned on personal data using DistilGPT-2 and Hugging Face Transformers.**

*Produces unique, non-hardcoded responses every single time.*

[![Python](https://img.shields.io/badge/Python-3.11-3572A5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Transformers-FFD21F?style=flat-square)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## ✦ What Is This?

This project is a **generative AI chatbot** trained entirely from scratch on custom data. Rather than prompting an existing API, the model weights are yours — shaped by your own training pipeline and data. Every response is generated fresh through supervised fine-tuning of DistilGPT-2, meaning no two answers are ever the same.

---

## ✦ Features

| | |
|---|---|
| 🧠 **Custom-trained model** | Fine-tuned DistilGPT-2 on your own personal dataset |
| 💬 **Unique responses** | Generative output — never returns hardcoded answers |
| 📝 **Full training pipeline** | Complete `train.py` script included end-to-end |
| 🚀 **Interactive chat loop** | Terminal-based conversation interface out of the box |
| 🔧 **Modular design** | Clean separation between training and inference scripts |

---

## ✦ Tech Stack

```
Python 3.11   ·   PyTorch   ·   Hugging Face Transformers   ·   DistilGPT-2   ·   SentencePiece
```

---

## ✦ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/PatrickAsaad1/Generative-AI.git
cd Generative-AI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ✦ Usage

**Train the model on your data:**
```bash
python scripts/train.py
```

**Start chatting with your AI:**
```bash
python scripts/run.py
```

> Type your questions and the model responds in real time.  
> Type `exit` to quit the chat loop.

---

## ✦ How It Works

```
Your Data  ──►  Fine-tune DistilGPT-2  ──►  Saved Model Weights
                                                      │
                                                      ▼
                                          run.py  ──►  Generated Response
```

The training script loads your custom dataset, fine-tunes the DistilGPT-2 base model via supervised learning, and saves the resulting weights locally. The run script loads those weights and feeds your input through the model to generate contextual, probabilistic responses — no lookup tables, no hardcoded replies.

---

## ✦ License

Released under the **MIT License** — free to use, modify, and distribute.

---

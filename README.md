# SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving 🤖: Finetuning Qwen3-VL model on a custom multi-image VL dataset, using QLoRA 4-bit quantization and Transformer Reinforcement Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/SFT_VLA_colab_notebook.ipynb)

This repository contains an **end-to-end pipeline for Supervised finetuning (SFT) of Qwen3-VL Vision–Language Model (VLM)** for **ADAS and Autonomous Driving video understanding** using multi-image inputs with QLoRA, designed to run efficiently on **Google Colab free-tier (T4 GPU)**.

---
<p align="center">
  <img src="reference/cover_image_ai.jpeg" width="900" alt="Project Cover"/>
</p>
---

## Project Overview

Modern autonomous driving systems require **scene understanding, reasoning, and safety awareness** from visual inputs. This project demonstrates how to:

- Convert **driving videos → structured natural language instructions**
- Build **custom multimodal dataset** for ADAS perception & reasoning
- Finetune **large Vision–Language Model** efficiently on limited hardware resources
- Enable **temporal reasoning** using sequential image inputs instead of full video attention

---

### Features

- Designed for **Google Colab free-tier T4 GPU**
- Resource-efficient finetuning using:
  - **QLoRA (4-bit quantization)**
  - **PEFT adapters**
  - **TRL (Transformer Reinforcement Learning)**
- Full training workflow:
  - Dataset generation and loading
  - Model and QLoRA adapter configuration
  - Training loop
  - Inference pipeline

---
### What It Does

For each driving video, the pipeline generates:

- Scene understanding descriptions
- Driving context & parameters
- Risk and safety assessments
- Outputs in:
  - **Natural language**
  - **Structured JSON format**

| Extracted Frames | Model Answer |
|----------|----------|
| <img src="reference/test_1_frames.png" width="200"/> | <img src="reference/test_1_answer.png" width="200"/> |
| <img src="reference/test_2_frames.png" width="200"/> | <img src="reference/test_2_answer.png" width="200"/> |
| <img src="reference/test_3_frames.png" width="200"/> | <img src="reference/test_3_answer.png" width="200"/> |

---
### Video Understanding Strategy

Instead of processing full video sequences (which typically require **FlashAttention 2** and **A100/Hopper GPUs**), this approach:

- Extracts a **fixed number of frames per video**
- Treats them as **sequential multi-image inputs**
- Enables **temporal reasoning** while remaining compatible with **T4 GPUs**

### Dataset

- **Hugging face link:**  
  [`SarveshBTelang/SFT_VLA_Dataset_1.0`](https://huggingface.co/datasets/SarveshBTelang/SFT_VLA_Dataset_1.0)

- **Source:**  
  Images extracted from **BDD100K driving videos**  
  http://bdd-data.berkeley.edu/

- **Structure:**
  - Multiple images per sample
  - Instruction–completion text pairs
  - Designed for **Supervised Fine-Tuning (SFT)** of VLMs using TRL

- Instructions were generated using the pipeline in [**generate_instruction_dataset.ipynb**](https://github.com/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/generate_instruction_dataset.ipynb)
- This notebook converts driving videos into raw instruction dataset using llm. These outputs serve as **raw instruction data** that are further cleaned, validated and refined via domain rules.
  
---
 
### Citations

- [TRL GitHub Repository](https://github.com/huggingface/trl)
- [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
- [Qwen3-VL Fine-tuning Examples](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)


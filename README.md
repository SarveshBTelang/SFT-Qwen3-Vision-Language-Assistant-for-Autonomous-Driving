# SFT-Qwen3-Vision-Language-Assistant for Autonomous-Driving 

### Fine-tuning Qwen3-VL model on a custom multi-image VL dataset, using QLoRA 4-bit quantization and Transformer Reinforcement Learning

<p align="center">
  <img src="reference/cover_image_ai.png" style="max-width:100%; height:auto;" />
</p>

<p align="right">
  <sup><sub>Image source: https://www.cloudfactory.com/blog/active-learning-and-autonomous-vehicles</sub></sup>
</p>

---

# ▹▹[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/SFT_VLA_colab_notebook.ipynb) ◃◃

<h3>Overview</h3>

This work presents a computationally efficient pipeline for fine-tuning the **Qwen3-VL Vision-Language model** for autonomous driving perception and reasoning tasks. The project demonstrates a parameter-efficient fine-tuning technique, **QLoRA with 4-bit quantization**, for training multi-billion-parameter models on consumer-grade GPU hardware (**Google Colab T4 GPU**) without sacrificing performance. The approach processes driving videos as sequential multi-image inputs, enabling temporal reasoning while circumventing the GPU memory requirements of full video attention mechanisms. The training leverages the **TRL (Transformer Reinforcement Learning)** library, which provides optimized trainers for parameter-efficient methods.

The system generates structured natural language descriptions encompassing scene understanding, contextual awareness, and safety-critical risk assessments from raw driving footage. The approach is validated on the **BDD100K** dataset, with both the fine-tuned model and curated instruction-following dataset released to facilitate reproducible research in this domain.

For each driving video, the pipeline generates:

- **Scene description**
- **Driving parameters**
- **Risk assessment**
- Outputs in:
  - Natural language
  - Structured JSON format

## Results

### Dataset
- Link: [`SarveshBTelang/SFT_VLA_Dataset_1.0`](https://huggingface.co/datasets/SarveshBTelang/SFT_VLA_Dataset_1.0)
- Format: Chat-style / OpenAI-format conversational dataset ([https://huggingface.co/docs/trl/main/en/dataset_formats](https://huggingface.co/docs/trl/main/en/dataset_formats))

- **Source:**  
  Images extracted from **BDD100K driving videos**  
  http://bdd-data.berkeley.edu/

- **Structure:**
  - Multiple images per sample
  - Instruction–completion text pairs
  - Designed for **Supervised Fine-Tuning (SFT)** of VLMs using TRL

- Instructions were generated using the pipeline in [**generate_instruction_dataset.ipynb**](https://github.com/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/generate_instruction_dataset.ipynb)
- This notebook converts driving videos into raw instruction dataset using llm. These outputs serve as **raw instruction data** that are further cleaned, validated and refined via domain rules.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/generate_instruction_dataset.ipynb)



### Checkpoints

12.2025 -->
[https://huggingface.co/SarveshBTelang/SFT_VLA_Qwen3-VL-2B-Instruct-multimage-trl](https://huggingface.co/SarveshBTelang/SFT_VLA_Qwen3-VL-2B-Instruct-multimage-trl)

### Training sample

```
{
    "images": [
        "frame_001.jpg",
        "frame_002.jpg",
        "frame_003.jpg",
        "frame_004.jpg",
        "frame_005.jpg",
        "frame_006.jpg",
        "frame_007.jpg",
        "frame_008.jpg",
        "frame_009.jpg",
        "frame_010.jpg"
    ],

    "prompt": [
        {
            "role": "user",
            "content": """
				You are an ADAS driving assistant. Analyze the scene from a driving and safety perspective, and produce:
				
				1. Scene Description
				2. Driving Parameters (JSON)
				3. Risk Assessment
				
				### Instructions
				
				**Scene Description**
				- Focus only on elements relevant to driving behavior and safety
				- Describe positions, movements, and actions of vehicles, pedestrians, and obstacles
				- Mention traffic signs, lights, road markings, and lane information if relevant
				- Highlight potential hazards requiring ego-vehicle attention
				
				**Driving Parameters (JSON)**
				{
				  "road_type": "...",
				  "lane_count": "...",
				  "ego_lane_position": "...",
				  "traffic_light_state": "...",
				  "pedestrian_on_road": "...",
				  "closest_vehicle_distance": "...",
				  "ego_vehicle_speed": "...",
				  "road_curvature": "...",
				  "weather": "...",
				  "visibility": "...",
				  "traffic_density": "...",
				  "risk_factor": "..."
				}
				
				**Risk Assessment**
				- Consider vehicles, pedestrians, road conditions, traffic rules, visibility, and environment
				
				Do not mention that it is a video or footage.
				Provide precise, actionable observations for an ADAS system.
				"""
        }
    ],

    "completion": [
        {
            "role": "assistant",
            "content": """
			"""
		}
	]
}
```

### Inference Examples

The model demonstrates strong performance across multiple evaluation dimensions:

#### Trial 1

<img src="reference/test_1_frames.png" style="max-width:100%; height:auto;" />
<img src="reference/test_1_answer.png" style="max-width:100%; height:auto;" />

#### Trial 2

<img src="reference/test_2_frames.png" style="max-width:100%; height:auto;" />
<img src="reference/test_2_answer.png" style="max-width:100%; height:auto;" />


#### Trial 3

<img src="reference/test_3_frames.png" style="max-width:100%; height:auto;" />
<img src="reference/test_3_answer.png" style="max-width:100%; height:auto;" />

### Capabilities

- Scene understanding and object detection across temporal sequences
- Contextual awareness of traffic rules and driving norms
- Safety-critical hazard identification (pedestrians, lane changes, etc.)
- Structured output generation adhering to JSON schemas

### Computational Efficiency

(For current checkpoint)

| Metric | Value |
|--------|-------|
| Training Time | ~2-4 hours (T4 GPU) |
| Memory Footprint | <16GB VRAM |
| Inference Latency | ~60 seconds per video |
| Model Size (Base) | ~2B parameters |
| Base model parameters | ~2,127,532,032 |
| LoRA adapter params | ~34,865,152 |
| Adapter Parameters | ~1.6388% of base model |

---

## Features

- Optimized for **Google Colab free-tier T4 GPU** deployment (<16GB VRAM)
- Resource-efficient finetuning using:
  - **QLoRA (4-bit quantization)**
  - **PEFT adapters**
  - **TRL (Transformer Reinforcement Learning)**
- ~2B parameters with 1.64% LoRA adapter overhead (~35M trainable params)
- Generates outputs in natural language and structured JSON format
- Performs temporal reasoning across video frames for context-aware understanding

---
## Video Understanding Strategy

Instead of processing full video sequences (which typically require **FlashAttention 2** and **A100/Hopper GPUs**), this approach:

- Extracts a **fixed number of frames per video**
- Treats them as **sequential multi-image inputs**
- Enables **temporal reasoning** while remaining compatible with **T4 GPUs**

### Workflow

<img src="reference/sft_vla_workflow.png" style="max-width:100%; height:auto;" />

---

## Framework versions

- TRL: 0.26.0
- Transformers: 4.57.3
- Pytorch: 2.9.0+cu126
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citations
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}

@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```
### Additional references
- [TRL GitHub Repository](https://github.com/huggingface/trl)
- [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
- [Qwen3-VL Fine-tuning Examples](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)

---

If you use this code, dataset, or model in your research, please cite it as follows:
```
@misc{telang2025sftqwen3vl,
    author       = {Sarvesh Telang},
    title        = {{SFT-Qwen3-Vision-Language-Assistant for Autonomous Driving}},
    year         = {2025},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving}},
    note         = {Accessed: yyyy-mm-dd}
}
```


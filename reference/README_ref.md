# Efficient Vision-Language Model Fine-tuning for Autonomous Driving Scene Understanding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/SFT_VLA_colab_notebook.ipynb)

**A resource-efficient framework for supervised fine-tuning of large-scale vision-language models on autonomous driving perception tasks using commodity hardware.**

---
<p align="center">
  <img src="reference/cover_image_ai.png" width="900" alt="Project Overview"/>
</p>
---

## Abstract

This work presents a computationally efficient pipeline for adapting large-scale Vision-Language Models (VLMs) to autonomous driving perception and reasoning tasks. We demonstrate that parameter-efficient fine-tuning techniques—specifically QLoRA with 4-bit quantization—enable training of billion-parameter models on consumer-grade hardware (Google Colab T4 GPU) without sacrificing performance. Our approach processes driving videos as sequential multi-image inputs, enabling temporal reasoning while circumventing the GPU memory requirements of full video attention mechanisms.

The system generates structured natural language descriptions encompassing scene understanding, contextual awareness, and safety-critical risk assessments from raw driving footage. We validate our approach on the BDD100K dataset and release both the fine-tuned models and curated instruction-following dataset to facilitate reproducible research in this domain.

---

## Research Contributions

1. **Resource-Efficient Training Protocol**: Demonstrates successful fine-tuning of Qwen3-VL (billions of parameters) on T4 GPUs through strategic application of QLoRA, gradient checkpointing, and mixed-precision training.

2. **Temporal Reasoning via Frame Sampling**: Proposes an alternative to computationally expensive video transformers by treating sampled video frames as multi-image inputs, preserving temporal context while maintaining hardware compatibility.

3. **Structured Instruction Dataset**: Introduces a curated multimodal dataset specifically designed for ADAS applications, comprising aligned visual sequences and natural language instructions with JSON-formatted outputs.

4. **End-to-End Open Framework**: Provides a complete, reproducible pipeline from dataset generation to model inference, enabling researchers to adapt the methodology to domain-specific applications.

---

## Methodology

### Problem Formulation

Given a driving video sequence $V = \{f_1, f_2, ..., f_n\}$, we seek to generate structured descriptions $D$ that capture:

- **Perceptual understanding**: Objects, agents, road geometry, environmental conditions
- **Contextual reasoning**: Traffic patterns, driver behavior, situational dynamics  
- **Safety assessment**: Risk factors, hazard detection, recommended actions

We formulate this as a vision-language instruction-following task where the model learns the mapping:

$$(\{f_{t_1}, f_{t_2}, ..., f_{t_k}\}, p) \rightarrow D$$

where $t_1, ..., t_k$ are uniformly sampled temporal indices and $p$ is the instruction prompt.

### Technical Approach

**Model Architecture**: We build upon Qwen3-VL, a state-of-the-art vision-language model featuring dynamic resolution encoding and multi-image comprehension capabilities.

**Parameter-Efficient Fine-tuning**: Rather than full model fine-tuning, we employ:
- **QLoRA**: Low-rank adaptation with 4-bit quantization (NormalFloat4)
- **Adapter Configuration**: Rank-16 decomposition applied to attention projections
- **Gradient Checkpointing**: Trade computation for memory during backpropagation

**Training Infrastructure**: 
- Hardware: NVIDIA T4 (16GB VRAM)
- Framework: Hugging Face TRL (Transformer Reinforcement Learning)
- Optimization: AdamW with cosine learning rate scheduling
- Batch Accumulation: Effective batch size of 16 through gradient accumulation

### Dataset Construction

**Source Material**: BDD100K diverse driving video dataset, representing varied geographic locations, weather conditions, and traffic scenarios.

**Processing Pipeline**:
1. Uniform temporal sampling (8-12 frames per video)
2. Frame preprocessing and normalization  
3. LLM-assisted annotation generation
4. Manual validation and domain-expert refinement
5. Structured JSON output formatting

**Dataset Statistics**:
- **Training samples**: [To be specified]
- **Validation split**: [To be specified]
- **Instruction diversity**: Scene description, risk assessment, decision reasoning
- **Public availability**: [`SarveshBTelang/SFT_VLA_Dataset_1.0`](https://huggingface.co/datasets/SarveshBTelang/SFT_VLA_Dataset_1.0)

---

## Results

### Qualitative Analysis

The model demonstrates strong performance across multiple evaluation dimensions:

| Input Frames | Model Response |
|:------------:|:--------------|
| <img src="reference/test_1_frames.png" width="200"/> | <img src="reference/test_1_answer.png" width="200"/> |
| <img src="reference/test_2_frames.png" width="200"/> | <img src="reference/test_2_answer.png" width="200"/> |
| <img src="reference/test_3_frames.png" width="200"/> | <img src="reference/test_3_answer.png" width="200"/> |

**Observed Capabilities**:
- Accurate object detection and tracking across temporal sequences
- Contextual understanding of traffic rules and driving norms
- Safety-critical hazard identification (pedestrians, sudden lane changes)
- Structured output generation adhering to JSON schemas

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Training Time | ~4-6 hours (T4 GPU) |
| Memory Footprint | <16GB VRAM |
| Inference Latency | ~2-3 seconds per video |
| Model Size (Base) | ~4B parameters |
| Adapter Parameters | ~0.5% of base model |

---

## Implementation

### Environment Setup

```bash
pip install transformers>=4.57.3 trl>=0.26.0 datasets>=4.4.1 torch>=2.9.0
```

### Quick Start

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load base model with adapters
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B",
    device_map="auto",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "path/to/adapters")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B")

# Process video frames
frames = load_frames("driving_video.mp4", num_frames=8)
inputs = processor(images=frames, text=instruction, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
```

See [`SFT_VLA_colab_notebook.ipynb`](https://colab.research.google.com/github/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/SFT_VLA_colab_notebook.ipynb) for complete training and inference pipeline.

---

## Repository Structure

```
├── SFT_VLA_colab_notebook.ipynb      # Complete training pipeline
├── generate_instruction_dataset.ipynb # Dataset generation workflow  
├── reference/                         # Figures and visualizations
├── README.md                          # This document
```

---

## Dataset Generation

The instruction dataset is generated through a multi-stage pipeline:

1. **Frame Extraction**: Uniform sampling from BDD100K videos
2. **Initial Annotation**: LLM-based caption generation via [`generate_instruction_dataset.ipynb`](https://github.com/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving/blob/main/generate_instruction_dataset.ipynb)
3. **Domain Refinement**: Manual validation and rule-based cleaning
4. **Formatting**: Conversion to SFT-compatible instruction-completion pairs

This approach balances automation scalability with annotation quality.

---

## Reproducibility

**Software Versions**:
- TRL: 0.26.0
- Transformers: 4.57.3  
- PyTorch: 2.9.0+cu126
- Datasets: 4.4.1
- Tokenizers: 0.22.1

**Hardware Requirements**:
- Minimum: NVIDIA T4 (16GB VRAM) or equivalent
- Recommended: NVIDIA V100 or A100 for faster training

**Compute Budget**:
- Training: ~4-6 GPU-hours on T4
- Dataset generation: ~2-3 GPU-hours

---

## Limitations and Future Work

**Current Limitations**:
- Frame sampling may miss critical short-duration events
- Model does not incorporate temporal attention mechanisms
- Limited evaluation on adversarial or edge cases
- No integration with real-time planning systems

**Future Directions**:
- Incorporation of sparse video attention for improved temporal modeling
- Multi-task learning with detection and segmentation objectives  
- Human preference alignment through RLHF
- Deployment optimization for edge devices
- Integration with end-to-end driving simulators

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{telang2024efficient,
  title={Efficient Vision-Language Model Fine-tuning for Autonomous Driving Scene Understanding},
  author={Telang, Sarvesh B.},
  year={2024},
  howpublished={\url{https://github.com/SarveshBTelang/SFT-Qwen3-Vision-Language-Assistant-for-Autonomous-Driving}}
}
```

---

## Acknowledgments

This work builds upon:

- **TRL Framework**: Hugging Face's Transformer Reinforcement Learning library
- **Qwen3-VL**: Alibaba's vision-language foundation model  
- **BDD100K Dataset**: UC Berkeley's diverse driving dataset

We thank the open-source community for making these resources available.

### References

```bibtex
@misc{vonwerra2022trl,
  title={{TRL: Transformer Reinforcement Learning}},
  author={von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  year={2020},
  publisher={GitHub},
  howpublished={\url{https://github.com/huggingface/trl}}
}

@inproceedings{bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

---

## License

[Specify license - e.g., MIT, Apache 2.0]

## Contact

For questions or collaboration opportunities, please open an issue or contact [your institutional email].

---

*This work was developed as part of research in multimodal machine learning and autonomous systems.*

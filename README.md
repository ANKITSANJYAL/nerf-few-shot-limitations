# Limitations of NeRF with Pre-trained Vision Features for Few-Shot 3D Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2506.18208-b31b1b.svg)](https://arxiv.org/abs/2506.18208)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of our paper **"Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction"** published at arXiv.

## 📖 Paper

**Title:** Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction  
**Authors:** Ankit Sanjyal  
**Conference:** arXiv preprint  
**Paper:** [arXiv:2506.18208](https://arxiv.org/abs/2506.18208)  
**PDF:** [Download PDF](https://arxiv.org/pdf/2506.18208.pdf)

## 🎯 Abstract

Neural Radiance Fields (NeRF) have revolutionized 3D scene reconstruction from sparse image collections. Recent work has explored integrating pre-trained vision features, particularly from DINO, to enhance few-shot reconstruction capabilities. However, the effectiveness of such approaches remains unclear, especially in extreme few-shot scenarios. In this paper, we present a systematic evaluation of DINO-enhanced NeRF models, comparing baseline NeRF, frozen DINO features, LoRA fine-tuned features, and multi-scale feature fusion. Surprisingly, our experiments reveal that all DINO variants perform worse than the baseline NeRF, achieving PSNR values around 12.9 to 13.0 compared to the baseline's 14.71. This counterintuitive result suggests that pre-trained vision features may not be beneficial for few-shot 3D reconstruction and may even introduce harmful biases.

## 🚨 Key Findings

**All DINO variants perform worse than a baseline NeRF in few-shot scenarios.**

Our systematic evaluation reveals that integrating pre-trained vision features (DINO) with NeRF actually degrades performance in few-shot 3D reconstruction scenarios, challenging common assumptions in the field.

## 📊 Experimental Results

| Method                   | PSNR   | SSIM  | LPIPS |
|------------------------- |--------|-------|-------|
| **Baseline NeRF**        | 14.71  | 0.46  | 0.53  |
| DINO-NeRF (frozen)       | 12.99  | 0.46  | 0.54  |
| LoRA-NeRF (fine-tuned)   | 12.97  | 0.45  | 0.54  |
| Multi-Scale LoRA-NeRF    | 12.94  | 0.44  | 0.54  |

## 🏗️ Architecture

Our implementation includes several variants of NeRF enhanced with DINO features:

- **Baseline NeRF**: Standard NeRF implementation
- **DINO-NeRF**: NeRF with frozen DINO features
- **LoRA-NeRF**: NeRF with LoRA fine-tuned DINO features  
- **Multi-Scale LoRA-NeRF**: Multi-scale feature fusion approach

## 📁 Repository Structure

```
├── src/
│   ├── models/
│   │   ├── dino_feature_model.py      # DINO feature extraction and integration
│   │   ├── multi_scale_dino.py        # Multi-scale DINO feature model
│   │   ├── nerf_mlp.py                # NeRF MLP with volume rendering
│   │   ├── nerf_model.py              # Basic NeRF MLP implementation
│   │   ├── ray_sampler.py             # Ray sampling utilities
│   │   ├── data_loader.py             # Data loading for Blender dataset
│   │   ├── volume_renderer.py         # Volume rendering functions
│   │   ├── positional_encoding.py     # Positional encoding for NeRF
│   │   ├── dino_lora.py               # LoRA integration with DINO
│   │   └── lora_dino.py               # Alternative LoRA implementation
│   ├── training/
│   │   ├── train.py                   # Main training script for all experiments
│   │   ├── evaluate.py                # Evaluation script for trained models
│   │   ├── train_baseline.py          # Specialized baseline training
│   │   ├── train_lora.py              # Specialized LoRA training
│   │   ├── train_multiscale.py        # Specialized multi-scale training
│   │   └── extract_features.py        # Feature extraction utilities
│   └── utils/
│       └── ray_utils.py               # Ray generation and projection utilities
├── experiments/                        # YAML configuration files
│   ├── baseline.yaml                  # Baseline NeRF configuration
│   ├── dino_nerf.yaml                 # NeRF with frozen DINO features
│   ├── lora.yaml                      # NeRF with LoRA-tuned DINO features
│   ├── multiscale.yaml                # Multi-scale LoRA-DINO-NeRF config
│   └── projection.yaml                # Projection-based configuration
├── results/                           # Experiment outputs
│   ├── baseline_lego/                 # Baseline NeRF results
│   ├── dino_nerf_lego/                # DINO-NeRF results
│   ├── lora_nerf_lego/                # LoRA-NeRF results
│   └── multiscale_nerf_lego/          # Multi-scale results
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ANKITSANJYAL/nerf-few-shot-limitations.git
   cd nerf-few-shot-limitations
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

1. **Train a baseline NeRF:**
   ```bash
   python src/training/train.py --config experiments/baseline.yaml
   ```

2. **Train with DINO features:**
   ```bash
   python src/training/train.py --config experiments/dino_nerf.yaml
   ```

3. **Train with LoRA fine-tuning:**
   ```bash
   python src/training/train.py --config experiments/lora.yaml
   ```

4. **Train multi-scale model:**
   ```bash
   python src/training/train.py --config experiments/multiscale.yaml
   ```

### Evaluation

```bash
python src/training/evaluate.py --config experiments/baseline.yaml
```

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{sanjyal2024limitations,
  title={Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction},
  author={Sanjyal, Ankit},
  journal={arXiv preprint arXiv:2506.18208},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or feedback, please contact:
- **Ankit Sanjyal**: [Email](mailto:contact@example.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the NeRF and DINO communities for their foundational work that made this research possible.

---

**Note:** This work demonstrates that pre-trained vision features can actually harm few-shot 3D reconstruction performance, challenging common assumptions in the field and suggesting that simpler architectures focusing on geometric consistency may be more effective for few-shot scenarios.
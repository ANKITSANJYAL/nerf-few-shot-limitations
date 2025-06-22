# Lora-Nerf: Limitations of NeRF with Pre-trained Vision Features for Few-Shot 3D Reconstruction

This repository contains the code and experiment configurations for our study on the limitations of combining pre-trained vision features (DINO) with Neural Radiance Fields (NeRF) for few-shot 3D reconstruction.

## 🚨 Key Finding

**All DINO variants perform worse than a baseline NeRF in few-shot scenarios.**

## 📊 Experimental Results

| Method                   | PSNR   | SSIM  | LPIPS |
|------------------------- |--------|-------|-------|
| **Baseline NeRF**        | 14.71  | 0.46  | 0.53  |
| DINO-NeRF (frozen)       | 12.99  | 0.46  | 0.54  |
| LoRA-NeRF (fine-tuned)   | 12.97  | 0.45  | 0.54  |
| Multi-Scale LoRA-NeRF    | 12.94  | 0.44  | 0.54  |

## 📁 Repository Structure

- `src/`
  - `models/`
    - `dino_feature_model.py`: DINO feature extraction and integration
    - `multi_scale_dino.py`: Multi-scale DINO feature model
    - `nerf_mlp.py`: NeRF MLP with volume rendering
    - `nerf_model.py`: Basic NeRF MLP implementation
    - `ray_sampler.py`: Ray sampling utilities
    - `data_loader.py`: Data loading for Blender dataset
    - `volume_renderer.py`: Volume rendering functions
    - `positional_encoding.py`: Positional encoding for NeRF
    - `dino_lora.py`, `lora_dino.py`: LoRA integration with DINO
  - `training/`
    - `train.py`: Main training script for all experiments
    - `evaluate.py`: Evaluation script for trained models
    - `train_baseline.py`, `train_lora.py`, `train_multiscale.py`: Specialized training scripts
    - `extract_features.py`: Feature extraction utilities
  - `utils/`
    - `ray_utils.py`: Ray generation and projection utilities
- `experiments/`: YAML configuration files for each experiment
  - `baseline.yaml`: Baseline NeRF config
  - `dino_nerf.yaml`: NeRF with frozen DINO features
  - `lora.yaml`: NeRF with LoRA-tuned DINO features
  - `multiscale.yaml`: Multi-scale LoRA-DINO-NeRF config
  - `projection.yaml`: Projection-based config
- `results/`: Output folders for each experiment (contains checkpoints and rendered images)
  - `baseline_lego/`, `dino_nerf_lego/`, `lora_nerf_lego/`, `multiscale_nerf_lego/`
- `requirements.txt`: Python dependencies
- `LICENSE`: License file
- `.gitignore`: Standard ignores for code, data, and experiment outputs

**Note:** The `paper/` folder is not included in this repository. The final paper and figures are available on arXiv (see link below).

## 🚀 Running Experiments

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model:**
   ```bash
   # Main training script (supports all experiments)
   python src/training/train.py --config experiments/baseline.yaml
   
   # Or use specialized scripts
   python src/training/train_baseline.py --config experiments/baseline.yaml
   python src/training/train_lora.py --config experiments/lora.yaml
   python src/training/train_multiscale.py --config experiments/multiscale.yaml
   ```

3. **Evaluate a model:**
   ```bash
   python src/training/evaluate.py --config experiments/baseline.yaml
   ```

## 📄 Paper

The full paper, including methodology, results, and analysis, is available on arXiv:

**[arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)**

## 📝 Citation

If you use this code or results, please cite our paper (see arXiv for BibTeX).

## 🤝 Contributing

Contributions and issues are welcome! Please open an issue or pull request.

## 📧 Contact

For questions, contact the authors via the email listed in the paper.

---

**Note:** This work demonstrates that pre-trained vision features can actually harm few-shot 3D reconstruction performance, challenging common assumptions in the field.
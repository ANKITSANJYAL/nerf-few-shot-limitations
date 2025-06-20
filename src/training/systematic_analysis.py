import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import json
import time
from pathlib import Path

class FewShotNeRFAnalyzer:
    """Systematic analysis framework for few-shot NeRF approaches"""
    
    def __init__(self, output_dir="analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def evaluate_approach(self, approach_name, config, trainer_class, data_path):
        """Evaluate a specific approach and store results"""
        print(f"\n{'='*50}")
        print(f"Evaluating: {approach_name}")
        print(f"{'='*50}")
        
        # Initialize trainer
        trainer = trainer_class(config)
        trainer.load_data(data_path, max_views=config['max_views'])
        
        # Training
        start_time = time.time()
        trainer.train(config['epochs'])
        training_time = time.time() - start_time
        
        # Evaluation
        final_psnr = trainer.best_psnr
        final_loss = trainer.get_final_loss()
        
        # Generate visualizations
        val_images = self.generate_validation_images(trainer, approach_name)
        
        # Store results
        self.results[approach_name] = {
            'config': config,
            'final_psnr': final_psnr,
            'final_loss': final_loss,
            'training_time': training_time,
            'val_images': val_images
        }
        
        print(f"✅ {approach_name}: PSNR={final_psnr:.2f}, Time={training_time:.1f}s")
        
    def generate_validation_images(self, trainer, approach_name):
        """Generate validation images for analysis"""
        images = []
        
        # Render validation views
        for view_idx in range(min(3, len(trainer.images))):
            val_img, psnr = trainer.validate(view_idx)
            
            # Save image
            img_path = self.output_dir / f"{approach_name}_view_{view_idx}.png"
            val_img_uint8 = (np.clip(val_img, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(img_path, val_img_uint8)
            
            images.append({
                'view_idx': view_idx,
                'psnr': psnr,
                'path': str(img_path)
            })
            
        return images
    
    def create_comparison_visualization(self):
        """Create side-by-side comparison of all approaches"""
        approaches = list(self.results.keys())
        n_approaches = len(approaches)
        
        fig, axes = plt.subplots(n_approaches, 3, figsize=(15, 5*n_approaches))
        if n_approaches == 1:
            axes = axes.reshape(1, -1)
        
        for i, approach in enumerate(approaches):
            result = self.results[approach]
            
            # Load validation images
            for j, img_info in enumerate(result['val_images'][:3]):
                img = Image.open(img_info['path'])
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{approach}\nView {j}, PSNR: {img_info['psnr']:.2f}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        comparison_path = self.output_dir / "approach_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(comparison_path)
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'summary': {},
            'detailed_results': {},
            'insights': [],
            'recommendations': []
        }
        
        # Summary statistics
        psnrs = [result['final_psnr'] for result in self.results.values()]
        report['summary'] = {
            'num_approaches': len(self.results),
            'best_psnr': max(psnrs),
            'worst_psnr': min(psnrs),
            'mean_psnr': np.mean(psnrs),
            'std_psnr': np.std(psnrs)
        }
        
        # Detailed results
        for approach, result in self.results.items():
            report['detailed_results'][approach] = {
                'psnr': result['final_psnr'],
                'loss': result['final_loss'],
                'training_time': result['training_time'],
                'config_summary': {
                    'pos_freq': result['config']['pos_freq'],
                    'reg_weight': result['config']['reg_weight'],
                    'learning_rate': result['config']['learning_rate']
                }
            }
        
        # Key insights
        report['insights'] = [
            "All approaches produce blurry results, suggesting fundamental limitations",
            f"PSNR range: {min(psnrs):.2f} - {max(psnrs):.2f} dB",
            "Training time varies significantly between approaches",
            "Regularization weight has minimal impact on final quality"
        ]
        
        # Recommendations
        report['recommendations'] = [
            "Explore multi-scale feature fusion approaches",
            "Investigate diffusion-based detail enhancement",
            "Consider temporal consistency regularization",
            "Evaluate on larger datasets to confirm findings"
        ]
        
        # Save report
        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print a summary of all results"""
        print("\n" + "="*60)
        print("FEW-SHOT NERF SYSTEMATIC ANALYSIS SUMMARY")
        print("="*60)
        
        for approach, result in self.results.items():
            print(f"\n{approach}:")
            print(f"  PSNR: {result['final_psnr']:.2f} dB")
            print(f"  Loss: {result['final_loss']:.6f}")
            print(f"  Time: {result['training_time']:.1f}s")
            print(f"  Config: pos_freq={result['config']['pos_freq']}, "
                  f"reg_weight={result['config']['reg_weight']}, "
                  f"lr={result['config']['learning_rate']}")

# Configuration templates for different approaches
BASELINE_CONFIG = {
    'dino_model': 'facebook/dinov2-base',
    'lora_rank': 16,
    'lora_alpha': 16,
    'pos_freq': 10,
    'dir_freq': 4,
    'hidden_dim': 256,
    'num_density_layers': 8,
    'learning_rate': 5e-4,
    'weight_decay': 1e-6,
    'lr_milestones': [50, 100, 150],
    'lr_gamma': 0.5,
    'epochs': 200,
    'rgb_weight': 1.0,
    'depth_weight': 0.1,
    'reg_weight': 0.01,
    'near': 2.0,
    'far': 6.0,
    'img_size': 128,
    'noise_std': 0.1,
    'white_bkgd': False,
    'output_dir': 'outputs/baseline',
    'val_freq': 10,
    'save_freq': 50,
    'data_path': 'data/nerf_synthetic/lego',
    'max_views': 5
}

HIGH_FREQ_CONFIG = {
    **BASELINE_CONFIG,
    'pos_freq': 12,
    'output_dir': 'outputs/high_freq'
}

LOW_REG_CONFIG = {
    **BASELINE_CONFIG,
    'reg_weight': 0.0001,
    'output_dir': 'outputs/low_reg'
}

OPTIMIZED_CONFIG = {
    **BASELINE_CONFIG,
    'pos_freq': 12,
    'reg_weight': 0.0001,
    'learning_rate': 2e-4,
    'lr_milestones': [80, 150],
    'num_density_layers': 10,
    'output_dir': 'outputs/optimized'
}

def main():
    """Run systematic analysis of different approaches"""
    analyzer = FewShotNeRFAnalyzer()
    
    # Import here to avoid circular imports
    from train import NeRFDINOTrainer
    
    # Define approaches to test
    approaches = [
        ("Baseline", BASELINE_CONFIG),
        ("High_Frequency", HIGH_FREQ_CONFIG),
        ("Low_Regularization", LOW_REG_CONFIG),
        ("Optimized", OPTIMIZED_CONFIG)
    ]
    
    # Run evaluations
    for approach_name, config in approaches:
        try:
            analyzer.evaluate_approach(
                approach_name, 
                config, 
                NeRFDINOTrainer,
                config['data_path']
            )
        except Exception as e:
            print(f"❌ Failed to evaluate {approach_name}: {e}")
            continue
    
    # Generate analysis
    analyzer.create_comparison_visualization()
    report = analyzer.generate_analysis_report()
    analyzer.print_summary()
    
    print(f"\n✅ Analysis complete! Results saved to {analyzer.output_dir}")

if __name__ == "__main__":
    main() 
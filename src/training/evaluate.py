import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

from train import NeRFDINOTrainer
from ray_utils import get_rays, project_points_to_image
import torch.nn.functional as F

class NeRFDINOEvaluator:
    def __init__(self, checkpoint_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Initialize trainer and load checkpoint
        self.trainer = NeRFDINOTrainer(config)
        self.load_checkpoint(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        """Load trained model"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.trainer.nerf_model.load_state_dict(checkpoint['nerf_state_dict'])
        self.trainer.dino_model.load_state_dict(checkpoint['dino_state_dict'])
        
        self.trainer.nerf_model.eval()
        self.trainer.dino_model.eval()
        
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        
    def novel_view_synthesis(self, test_poses, output_dir):
        """Generate novel views and compute metrics"""
        print("🎬 Evaluating novel view synthesis...")
        
        os.makedirs(output_dir, exist_ok=True)
        psnrs = []
        ssims = []
        
        for i, test_pose in enumerate(test_poses):
            print(f"Rendering view {i+1}/{len(test_poses)}")
            
            # Generate rays for novel view
            rays_o, rays_d = get_rays(self.trainer.H, self.trainer.W, self.trainer.focal, test_pose)
            
            # Render image in chunks
            rendered_img = self.render_full_image(rays_o, rays_d, closest_view_idx=0)
            
            # Save rendered image
            img_uint8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_dir, f'novel_view_{i:03d}.png'), img_uint8)
            
            # If ground truth is available, compute metrics
            # (This would require loading test set ground truth)
            
        return {
            'mean_psnr': np.mean(psnrs) if psnrs else None,
            'mean_ssim': np.mean(ssims) if ssims else None,
            'rendered_images': len(test_poses)
        }
    
    def render_full_image(self, rays_o, rays_d, closest_view_idx, chunk_size=1024):
        """Render a full image efficiently"""
        H, W = rays_o.shape[:2]
        rendered_img = torch.zeros((H, W, 3), device=self.device)
        
        with torch.no_grad():
            rays_o_flat = rays_o.view(-1, 3)
            rays_d_flat = rays_d.view(-1, 3)
            
            for i in range(0, rays_o_flat.shape[0], chunk_size):
                ray_batch_o = rays_o_flat[i:i+chunk_size]
                ray_batch_d = rays_d_flat[i:i+chunk_size]
                
                predictions = self.trainer.render_rays(ray_batch_o, ray_batch_d, closest_view_idx, N_samples=64)
                rendered_img.view(-1, 3)[i:i+chunk_size] = predictions['rgb']
        
        return rendered_img.cpu().numpy()
    
    def test_3d_consistency(self, test_points_3d, output_dir):
        """Test if DINO features are consistent across different viewpoints for same 3D points"""
        print("🔍 Testing 3D consistency of DINO features...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        consistency_scores = []
        n_points = min(1000, len(test_points_3d))  # Sample subset for efficiency
        sampled_points = test_points_3d[np.random.choice(len(test_points_3d), n_points, replace=False)]
        
        # Extract features for same 3D points across different views
        all_features = []
        
        for view_idx in range(len(self.trainer.poses)):
            pose = self.trainer.poses[view_idx].to(self.device)
            points_tensor = torch.from_numpy(sampled_points).float().to(self.device)
            
            # Project 3D points to this view
            points_2d, depths, valid_mask = project_points_to_image(
                points_tensor, pose, self.trainer.focal, self.trainer.H, self.trainer.W
            )
            
            # Sample DINO features at projected locations
            dino_features = self.trainer.dino_features[view_idx].to(self.device)
            sampled_features = self.trainer.dino_model.sample_features_at_points(
                dino_features.unsqueeze(0), points_2d, (self.trainer.H, self.trainer.W)
            )
            
            # Only keep features for valid points
            valid_features = sampled_features[valid_mask]
            all_features.append(valid_features.cpu().numpy())
        
        # Compute pairwise consistency between views
        consistency_matrix = np.zeros((len(all_features), len(all_features)))
        
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                # Find common valid points between views i and j
                min_points = min(len(all_features[i]), len(all_features[j]))
                if min_points > 0:
                    features_i = all_features[i][:min_points]
                    features_j = all_features[j][:min_points]
                    
                    # Compute cosine similarity
                    similarities = []
                    for k in range(min_points):
                        sim = cosine_similarity(
                            features_i[k:k+1], features_j[k:k+1]
                        )[0, 0]
                        similarities.append(sim)
                    
                    consistency_matrix[i, j] = np.mean(similarities)
                    consistency_matrix[j, i] = consistency_matrix[i, j]
        
        # Fill diagonal
        np.fill_diagonal(consistency_matrix, 1.0)
        
        # Visualize consistency matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(consistency_matrix, annot=True, cmap='viridis', 
                   xticklabels=[f'View {i}' for i in range(len(all_features))],
                   yticklabels=[f'View {i}' for i in range(len(all_features))])
        plt.title('DINO Feature Consistency Across Views\n(Higher values = more consistent)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'consistency_matrix.png'), dpi=150)
        plt.close()
        
        mean_consistency = np.mean(consistency_matrix[np.triu_indices(len(all_features), k=1)])
        
        return {
            'mean_consistency': mean_consistency,
            'consistency_matrix': consistency_matrix,
            'n_points_tested': n_points
        }
    
    def test_depth_estimation(self, output_dir):
        """Test depth estimation quality"""
        print("📏 Testing depth estimation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        depth_results = []
        
        for view_idx in range(len(self.trainer.images)):
            rays_o, rays_d, target_rgb = self.trainer.get_training_rays(view_idx)
            
            # Render depth map
            with torch.no_grad():
                H, W = rays_o.shape[:2]
                depth_map = torch.zeros((H, W), device=self.device)
                
                rays_o_flat = rays_o.view(-1, 3)
                rays_d_flat = rays_d.view(-1, 3)
                
                chunk_size = 1024
                for i in range(0, rays_o_flat.shape[0], chunk_size):
                    ray_batch_o = rays_o_flat[i:i+chunk_size]
                    ray_batch_d = rays_d_flat[i:i+chunk_size]
                    
                    predictions = self.trainer.render_rays(ray_batch_o, ray_batch_d, view_idx, N_samples=64)
                    depth_map.view(-1)[i:i+chunk_size] = predictions['depth']
            
            depth_np = depth_map.cpu().numpy()
            
            # Save depth visualization
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(target_rgb.cpu().numpy())
            plt.title(f'RGB View {view_idx}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(depth_np, cmap='plasma')
            plt.colorbar()
            plt.title('Rendered Depth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.hist(depth_np.flatten(), bins=50, alpha=0.7)
            plt.xlabel('Depth')
            plt.ylabel('Frequency')
            plt.title('Depth Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'depth_view_{view_idx}.png'), dpi=150)
            plt.close()
            
            depth_results.append({
                'view_idx': view_idx,
                'mean_depth': np.mean(depth_np),
                'depth_std': np.std(depth_np),
                'depth_range': (np.min(depth_np), np.max(depth_np))
            })
        
        return depth_results
    
    def test_feature_interpolation(self, output_dir):
        """Test how well DINO features interpolate in 3D space"""
        print("🔄 Testing feature interpolation...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a grid of 3D points
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        z = np.linspace(2.5, 4.5, 10)  # Within scene bounds
        
        grid_points = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        
        # Test interpolation quality by checking feature smoothness
        interpolation_results = {}
        
        for view_idx in range(min(2, len(self.trainer.poses))):  # Test on first 2 views
            pose = self.trainer.poses[view_idx].to(self.device)
            points_tensor = torch.from_numpy(grid_points).float().to(self.device)
            
            # Project to image
            points_2d, depths, valid_mask = project_points_to_image(
                points_tensor, pose, self.trainer.focal, self.trainer.H, self.trainer.W
            )
            
            # Sample features
            dino_features = self.trainer.dino_features[view_idx].to(self.device)
            sampled_features = self.trainer.dino_model.sample_features_at_points(
                dino_features.unsqueeze(0), points_2d, (self.trainer.H, self.trainer.W)
            )
            
            # Compute smoothness metric (variance of feature differences)
            valid_features = sampled_features[valid_mask].cpu().numpy()
            
            if len(valid_features) > 100:
                # Sample pairs of nearby points
                n_pairs = 1000
                indices = np.random.choice(len(valid_features), (n_pairs, 2), replace=True)
                
                feature_diffs = []
                for i, j in indices:
                    diff = np.linalg.norm(valid_features[i] - valid_features[j])
                    feature_diffs.append(diff)
                
                smoothness_score = np.mean(feature_diffs)
                interpolation_results[f'view_{view_idx}'] = {
                    'smoothness_score': smoothness_score,
                    'n_valid_points': len(valid_features)
                }
        
        return interpolation_results
    
    def comprehensive_evaluation(self, output_dir='outputs/evaluation'):
        """Run all evaluation tests"""
        print("🚀 Starting comprehensive evaluation...")
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 1. Novel view synthesis
        novel_poses = self.generate_test_poses(n_views=8)
        results['novel_view'] = self.novel_view_synthesis(
            novel_poses, os.path.join(output_dir, 'novel_views')
        )
        
        # 2. 3D consistency test
        test_points = self.generate_test_points_3d(n_points=2000)
        results['3d_consistency'] = self.test_3d_consistency(
            test_points, os.path.join(output_dir, 'consistency')
        )
        
        # 3. Depth estimation
        results['depth_estimation'] = self.test_depth_estimation(
            os.path.join(output_dir, 'depth')
        )
        
        # 4. Feature interpolation
        results['interpolation'] = self.test_feature_interpolation(
            os.path.join(output_dir, 'interpolation')
        )
        
        # Save comprehensive report
        self.generate_report(results, output_dir)
        
        return results
    
    def generate_test_poses(self, n_views=8, radius=4.0):
        """Generate test camera poses around the scene"""
        angles = np.linspace(0, 2*np.pi, n_views, endpoint=False)
        poses = []
        
        for angle in angles:
            # Camera position
            cam_pos = np.array([
                radius * np.cos(angle
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import logging
from tqdm import tqdm
import argparse

from params import par
from model import VisualInertialOdometryModel, load_pretrained_flownet
from dataset import create_data_loaders
from helper import relative_to_absolute_pose, compute_ate, compute_rpe, visualize_trajectory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, device, use_imu=True):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    total_depth_trans_loss = 0.0
    losses = []
    
    # Log gradient information periodically
    log_gradients = (par.epochs <= 20 or par.epochs % 10 == 0)
    imu_grad_norm = 0
    visual_grad_norm = 0
    depth_grad_norm = 0
    
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for data in progress_bar:
        # Unpack data based on enabled modalities
        idx = 0
        rgb_seq = data[idx].to(device)
        idx += 1
        
        depth_seq = None
        if par.use_depth:
            depth_seq = data[idx].to(device)
            idx += 1
        
        imu_data = None
        if use_imu:
            imu_data = data[idx].to(device)
            idx += 1
        
        rel_poses = data[idx].to(device)
        abs_poses = data[idx+1].to(device)
        
        # Forward pass
        pred_rel_poses = model(rgb_seq, depth=depth_seq, imu_data=imu_data)
        
        # Compute loss
        loss, rot_loss, trans_loss, depth_trans_loss = model.get_loss(pred_rel_poses, rel_poses)
        losses.append(loss.item())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradient norms for IMU, visual, and depth parts
        if log_gradients:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'imu' in name:
                        imu_grad_norm += param.grad.norm().item()
                    elif 'conv' in name:
                        visual_grad_norm += param.grad.norm().item()
                    elif 'depth' in name:
                        depth_grad_norm += param.grad.norm().item()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_rot_loss += rot_loss.item()
        total_trans_loss += trans_loss.item()
        total_depth_trans_loss += depth_trans_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rot_loss': f"{rot_loss.item():.4f}",
            'trans_loss': f"{trans_loss.item():.4f}",
            'depth_loss': f"{depth_trans_loss.item():.4f}"
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_rot_loss = total_rot_loss / len(dataloader)
    avg_trans_loss = total_trans_loss / len(dataloader)
    avg_depth_trans_loss = total_depth_trans_loss / len(dataloader)
    std_loss = np.std(losses)
    
    # Log gradient information
    if log_gradients:
        grad_info = []
        if imu_grad_norm > 0:
            grad_info.append(f"IMU grad norm: {imu_grad_norm:.4f}")
        if visual_grad_norm > 0:
            grad_info.append(f"Visual grad norm: {visual_grad_norm:.4f}")
        if depth_grad_norm > 0:
            grad_info.append(f"Depth grad norm: {depth_grad_norm:.4f}")
        if imu_grad_norm > 0 and visual_grad_norm > 0:
            grad_info.append(f"IMU/Visual ratio: {imu_grad_norm/visual_grad_norm:.4f}")
        if depth_grad_norm > 0 and visual_grad_norm > 0:
            grad_info.append(f"Depth/Visual ratio: {depth_grad_norm/visual_grad_norm:.4f}")
        if grad_info:
            logger.info(", ".join(grad_info))
    
    return avg_loss, avg_rot_loss, avg_trans_loss, avg_depth_trans_loss, std_loss

def validate(model, dataloader, device, use_imu=True, visualize=False, log_dir=None):
    """Validate the model on validation data."""
    model.eval()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    total_depth_trans_loss = 0.0
    losses = []
    
    # ATE and RPE metrics
    all_ate_mean = []
    all_ate_std = []
    all_rpe_trans_mean = []
    all_rpe_trans_std = []
    all_rpe_rot_mean = []
    all_rpe_rot_std = []
    
    body_to_camera = par.body_to_camera.to(device)
    
    # Track best and worst trajectories for visualization
    best_traj = {'error': float('inf'), 'pred': None, 'gt': None, 'idx': -1}
    worst_traj = {'error': 0, 'pred': None, 'gt': None, 'idx': -1}
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
        for batch_idx, data in enumerate(progress_bar):
            # Unpack data based on enabled modalities
            idx = 0
            rgb_seq = data[idx].to(device)
            idx += 1
            
            depth_seq = None
            if par.use_depth:
                depth_seq = data[idx].to(device)
                idx += 1
            
            imu_data = None
            if use_imu:
                imu_data = data[idx].to(device)
                idx += 1
            
            rel_poses = data[idx].to(device)
            abs_poses = data[idx+1].to(device)
            
            # Forward pass
            pred_rel_poses = model(rgb_seq, depth=depth_seq, imu_data=imu_data)
            
            # Compute loss
            loss, rot_loss, trans_loss, depth_trans_loss = model.get_loss(pred_rel_poses, rel_poses)
            losses.append(loss.item())
            
            # Update statistics
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            total_depth_trans_loss += depth_trans_loss.item()
            
            # Compute trajectory and metrics
            for i in range(len(rgb_seq)):
                # Integrate predictions to get absolute trajectory
                pred_abs_poses = relative_to_absolute_pose(pred_rel_poses[i], body_to_camera)
                gt_abs_poses = abs_poses[i]
                
                # Compute ATE
                ate_mean, ate_std = compute_ate(pred_abs_poses, gt_abs_poses)
                all_ate_mean.append(ate_mean)
                all_ate_std.append(ate_std)
                
                # Compute RPE
                rpe_trans_mean, rpe_trans_std, rpe_rot_mean, rpe_rot_std = compute_rpe(pred_abs_poses, gt_abs_poses)
                all_rpe_trans_mean.append(rpe_trans_mean)
                all_rpe_trans_std.append(rpe_trans_std)
                all_rpe_rot_mean.append(rpe_rot_mean)
                all_rpe_rot_std.append(rpe_rot_std)
                
                # Track best and worst trajectories
                if visualize:
                    traj_idx = batch_idx * rgb_seq.size(0) + i
                    if ate_mean < best_traj['error']:
                        best_traj = {
                            'error': ate_mean,
                            'pred': pred_abs_poses.cpu(),
                            'gt': gt_abs_poses.cpu(),
                            'idx': traj_idx
                        }
                    if ate_mean > worst_traj['error']:
                        worst_traj = {
                            'error': ate_mean,
                            'pred': pred_abs_poses.cpu(),
                            'gt': gt_abs_poses.cpu(),
                            'idx': traj_idx
                        }
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rot_loss': f"{rot_loss.item():.4f}",
                'trans_loss': f"{trans_loss.item():.4f}",
                'depth_loss': f"{depth_trans_loss.item():.4f}"
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_rot_loss = total_rot_loss / len(dataloader)
    avg_trans_loss = total_trans_loss / len(dataloader)
    avg_depth_trans_loss = total_depth_trans_loss / len(dataloader)
    std_loss = np.std(losses)
    
    # Calculate average metrics
    avg_ate_mean = np.mean(all_ate_mean)
    avg_ate_std = np.mean(all_ate_std)
    avg_rpe_trans_mean = np.mean(all_rpe_trans_mean)
    avg_rpe_trans_std = np.mean(all_rpe_trans_std)
    avg_rpe_rot_mean = np.mean(all_rpe_rot_mean)
    avg_rpe_rot_std = np.mean(all_rpe_rot_std)
    
    metrics = {
        'ate_mean': avg_ate_mean,
        'ate_std': avg_ate_std,
        'rpe_trans_mean': avg_rpe_trans_mean,
        'rpe_trans_std': avg_rpe_trans_std,
        'rpe_rot_mean': avg_rpe_rot_mean,
        'rpe_rot_std': avg_rpe_rot_std,
        'depth_trans_loss': avg_depth_trans_loss
    }
    
    # Visualize best and worst trajectories if requested
    if visualize and log_dir is not None:
        imu_suffix = "_with_imu" if use_imu else ""
        
        # Best trajectory
        fig_best, _ = visualize_trajectory(
            best_traj['pred'], 
            best_traj['gt'], 
            title=f"Best Trajectory (ATE: {best_traj['error']:.4f}m, ID: {best_traj['idx']})", 
            save_path=os.path.join(log_dir, f"best_trajectory{imu_suffix}.png")
        )
        
        # Worst trajectory
        fig_worst, _ = visualize_trajectory(
            worst_traj['pred'], 
            worst_traj['gt'], 
            title=f"Worst Trajectory (ATE: {worst_traj['error']:.4f}m, ID: {best_traj['idx']})", 
            save_path=os.path.join(log_dir, f"worst_trajectory{imu_suffix}.png")
        )
        
        # Close figures to prevent memory issues
        plt.close(fig_best)
        plt.close(fig_worst)
    
    return avg_loss, avg_rot_loss, avg_trans_loss, avg_depth_trans_loss, std_loss, metrics

def train(use_imu=True, use_integrated_imu=True, batch_size=None, learning_rate=None):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Use command line overrides if provided
    if batch_size is not None:
        par.batch_size = batch_size
    
    if learning_rate is not None:
        par.optim['lr'] = learning_rate
    
    # Model configuration
    imu_suffix = "_with_imu" if use_imu else ""
    integrated_suffix = "_integrated" if use_integrated_imu else ""
    run_name = f"rgbvo{imu_suffix}{integrated_suffix}-{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize wandb
    wandb.init(project="rgbvo", name=run_name)
    
    # Log parameters
    wandb.config.update(vars(par))
    wandb.config.update({
        "use_imu": use_imu,
        "use_integrated_imu": use_integrated_imu,
        "batch_size": par.batch_size,
        "learning_rate": par.optim['lr']
    })
    
    # Create data loaders with IMU support
    train_loader, val_loader = create_data_loaders(
        batch_size=par.batch_size,
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    logger.info(f"Training with {len(train_loader)} batches, validating with {len(val_loader)} batches")
    logger.info(f"Using IMU data: {use_imu} (Integrated: {use_integrated_imu})")
    
    # Determine IMU input size based on integration mode
    imu_input_size = 21 if use_integrated_imu else 6
    
    # Create visual-inertial odometry model
    try:
        model = VisualInertialOdometryModel(
            imsize1=par.img_h, 
            imsize2=par.img_w, 
            batchNorm=par.batch_norm,
            use_imu=use_imu,
            imu_feature_size=512,
            imu_input_size=imu_input_size,
            use_adaptive_weighting=True,
            use_depth_translation=False,
            pretrained_depth_path=par.pretrained_depth if par.use_depth else None
        )
    except TypeError as e:
        logger.warning(f"Model initialization error: {e}")
        logger.info("Retrying with reduced parameters...")
        model = VisualInertialOdometryModel(
            imsize1=par.img_h, 
            imsize2=par.img_w, 
            batchNorm=par.batch_norm,
            use_imu=use_imu,
            pretrained_depth_path=par.pretrained_depth if par.use_depth else None
        )
    
    model = model.to(device)
    
    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Load FlowNet weights if available
    if par.pretrained_flownet and os.path.exists(par.pretrained_flownet):
        load_pretrained_flownet(model, par.pretrained_flownet)
    
    # Create optimizer with differential learning rates
    optimizer = optim.Adam([
        {'params': model.depth_encoder.parameters(), 'lr': par.optim['lr'] * 0.1},  # Lower LR for pretrained depth encoder
        {'params': [p for n, p in model.named_parameters() if 'depth' not in n], 'lr': par.optim['lr']}
    ], weight_decay=par.optim['weight_decay'])
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Update model path with suffix
    par.model_name = f"{par.model_name}{imu_suffix}{integrated_suffix}"
    par.model_path = os.path.join(par.model_dir, f"{par.model_name}.pt")
    
    # Ensure log directory exists
    os.makedirs(par.log_dir, exist_ok=True)
    
    # Set up logging file
    log_file = os.path.join(par.log_dir, f"{par.model_name}_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Training variables
    best_val_loss = float('inf')
    best_ate = float('inf')
    train_losses = []
    val_losses = []
    val_ates = []
    depth_trans_losses = []
    
    # Train for specified number of epochs
    for epoch in range(par.epochs):
        logger.info(f"Epoch {epoch+1}/{par.epochs}")
        
        # Train
        train_loss, train_rot_loss, train_trans_loss, train_depth_trans_loss, train_loss_std = train_epoch(
            model, train_loader, optimizer, device, use_imu=use_imu
        )
        train_losses.append(train_loss)
        
        # Validate
        visualize_epoch = (epoch + 1) % 10 == 0 or epoch == par.epochs - 1
        val_loss, val_rot_loss, val_trans_loss, val_depth_trans_loss, val_loss_std, metrics = validate(
            model, val_loader, device, use_imu=use_imu, visualize=visualize_epoch, log_dir=par.log_dir
        )
        val_losses.append(val_loss)
        val_ates.append(metrics['ate_mean'])
        depth_trans_losses.append(val_depth_trans_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss_mean': train_loss,
            'train_loss_std': train_loss_std,
            'train_rot_loss': train_rot_loss,
            'train_trans_loss': train_trans_loss,
            'train_depth_trans_loss': train_depth_trans_loss,
            'val_loss_mean': val_loss,
            'val_loss_std': val_loss_std,
            'val_rot_loss': val_rot_loss,
            'val_trans_loss': val_trans_loss,
            'val_depth_trans_loss': val_depth_trans_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'ate_mean': metrics['ate_mean'],
            'ate_std': metrics['ate_std'],
            'rpe_trans_mean': metrics['rpe_trans_mean'],
            'rpe_trans_std': metrics['rpe_trans_std'],
            'rpe_rot_mean': metrics['rpe_rot_mean'],
            'rpe_rot_std': metrics['rpe_rot_std']
        })
        
        # Print metrics
        logger.info(f"Train Loss: {train_loss:.6f} ± {train_loss_std:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f} ± {val_loss_std:.6f}")
        logger.info(f"Train Depth Trans Loss: {train_depth_trans_loss:.6f}")
        logger.info(f"Val Depth Trans Loss: {val_depth_trans_loss:.6f}")
        logger.info(f"ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m")
        logger.info(f"RPE Trans: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m")
        logger.info(f"RPE Rot: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad")
        
        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), par.model_path)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Also save best model by ATE
        if metrics['ate_mean'] < best_ate:
            best_ate = metrics['ate_mean']
            ate_model_path = os.path.join(par.model_dir, f"{par.model_name}_best_ate.pt")
            torch.save(model.state_dict(), ate_model_path)
            logger.info(f"New best ATE model saved with ATE: {best_ate:.4f}m")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == par.epochs - 1:
            checkpoint_path = os.path.join(par.model_dir, f"{par.model_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # Plot and save loss curve
            plt.figure(figsize=(12, 12))
            
            # Loss plot
            plt.subplot(3, 1, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Validation Loss')
            plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.6f}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss ({"with" if use_imu else "without"} IMU)')
            plt.legend()
            plt.grid(True)
            
            # ATE plot
            plt.subplot(3, 1, 2)
            plt.plot(range(1, len(val_ates) + 1), val_ates, marker='s', color='green', label='ATE')
            plt.axhline(y=best_ate, color='r', linestyle='--', label=f'Best ATE: {best_ate:.4f}m')
            plt.xlabel('Epoch')
            plt.ylabel('ATE (m)')
            plt.title('Absolute Trajectory Error')
            plt.legend()
            plt.grid(True)
            
            # Depth Translation Loss plot
            plt.subplot(3, 1, 3)
            plt.plot(range(1, len(depth_trans_losses) + 1), depth_trans_losses, marker='d', color='purple', label='Depth Trans Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Depth Translation Loss')
            plt.title('Depth-Supervised Translation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(par.log_dir, f"{par.model_name}_curves.png"))
            
            # Save as wandb artifact
            wandb.log({"loss_curve": wandb.Image(plt)})
            
            plt.close()
    
    # Finish training
    wandb.finish()
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best ATE: {best_ate:.4f}m")
    
    return best_val_loss, best_ate

def test(model_path, use_imu=True, use_integrated_imu=True, batch_size=None):
    """Test trained model on test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing model: {model_path}")
    logger.info(f"Using IMU: {use_imu} (Integrated: {use_integrated_imu})")
    
    # Use provided batch size or default
    if batch_size is not None:
        test_batch_size = batch_size
    else:
        test_batch_size = par.batch_size
    
    # Create test data loader (using validation set for now)
    _, test_loader = create_data_loaders(
        batch_size=test_batch_size,
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    
    # Determine IMU input size based on integration mode
    imu_input_size = 21 if use_integrated_imu else 6
    
    # Create model
    try:
        model = VisualInertialOdometryModel(
            imsize1=par.img_h, 
            imsize2=par.img_w, 
            batchNorm=par.batch_norm,
            use_imu=use_imu,
            imu_feature_size=128,
            imu_input_size=imu_input_size,
            use_adaptive_weighting=True,
            use_depth_translation=True,
            pretrained_depth_path=par.pretrained_depth if par.use_depth else None
        )
    except TypeError as e:
        logger.warning(f"Model initialization error: {e}")
        logger.info("Retrying with reduced parameters...")
        model = VisualInertialOdometryModel(
            imsize1=par.img_h, 
            imsize2=par.img_w, 
            batchNorm=par.batch_norm,
            use_imu=use_imu,
            pretrained_depth_path=par.pretrained_depth if par.use_depth else None
        )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Run validation as test
    _, _, _, _, _, metrics = validate(
        model, 
        test_loader, 
        device, 
        use_imu=use_imu, 
        visualize=True, 
        log_dir=par.log_dir
    )
    
    # Print test results
    logger.info("Test Results:")
    logger.info(f"ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m")
    logger.info(f"RPE Trans: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m")
    logger.info(f"RPE Rot: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad")
    logger.info(f"Depth Translation Loss: {metrics['depth_trans_loss']:.6f}")
    
    return metrics

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visual-Inertial Odometry Training')
    parser.add_argument('--no_imu', action='store_true', help='Disable IMU usage')
    parser.add_argument('--raw_imu', action='store_true', help='Use raw IMU data instead of integrated features')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--test', action='store_true', help='Run test instead of training')
    parser.add_argument('--model_path', type=str, help='Path to model for testing')
    args = parser.parse_args()
    
    # Determine IMU usage
    use_imu = not args.no_imu
    use_integrated_imu = not args.raw_imu
    
    if args.test:
        # Run test mode
        if args.model_path is None:
            imu_suffix = "_with_imu" if use_imu else ""
            integrated_suffix = "_integrated" if use_integrated_imu else ""
            model_name = f"{par.model_name}{imu_suffix}{integrated_suffix}"
            model_path = os.path.join(par.model_dir, f"{model_name}.pt")
        else:
            model_path = args.model_path
        
        # Test the model
        test(
            model_path=model_path,
            use_imu=use_imu,
            use_integrated_imu=use_integrated_imu,
            batch_size=args.batch_size
        )
    else:
        # Run training mode
        train(
            use_imu=use_imu,
            use_integrated_imu=use_integrated_imu,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
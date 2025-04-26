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

from params import par
from model import RGBVO, load_pretrained_flownet
from dataset import create_data_loaders
from helper import relative_to_absolute_pose, compute_ate, compute_rpe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    losses = []
    
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for rgb_seq, rel_poses, abs_poses in progress_bar:
        # Move data to device
        rgb_seq = rgb_seq.to(device)
        rel_poses = rel_poses.to(device)
        
        # Forward pass
        pred_rel_poses = model(rgb_seq)
        
        # Compute loss
        loss, rot_loss, trans_loss = model.get_loss(pred_rel_poses, rel_poses)
        losses.append(loss.item())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_rot_loss += rot_loss.item()
        total_trans_loss += trans_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rot_loss': f"{rot_loss.item():.4f}",
            'trans_loss': f"{trans_loss.item():.4f}"
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_rot_loss = total_rot_loss / len(dataloader)
    avg_trans_loss = total_trans_loss / len(dataloader)
    std_loss = np.std(losses)
    
    return avg_loss, avg_rot_loss, avg_trans_loss, std_loss

def validate(model, dataloader, device):
    """Validate the model on validation data."""
    model.eval()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    losses = []
    
    # ATE and RPE metrics
    all_ate_mean = []
    all_ate_std = []
    all_rpe_trans_mean = []
    all_rpe_trans_std = []
    all_rpe_rot_mean = []
    all_rpe_rot_std = []
    
    body_to_camera = par.body_to_camera.to(device)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
        for rgb_seq, rel_poses, abs_poses in progress_bar:
            # Move data to device
            rgb_seq = rgb_seq.to(device)
            rel_poses = rel_poses.to(device)
            abs_poses = abs_poses.to(device)
            
            # Forward pass
            pred_rel_poses = model(rgb_seq)
            
            # Compute loss
            loss, rot_loss, trans_loss = model.get_loss(pred_rel_poses, rel_poses)
            losses.append(loss.item())
            
            # Update statistics
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rot_loss': f"{rot_loss.item():.4f}",
                'trans_loss': f"{trans_loss.item():.4f}"
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_rot_loss = total_rot_loss / len(dataloader)
    avg_trans_loss = total_trans_loss / len(dataloader)
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
        'rpe_rot_std': avg_rpe_rot_std
    }
    
    return avg_loss, avg_rot_loss, avg_trans_loss, std_loss, metrics

def train():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="rgbvo", name=f"rgbvo-{time.strftime('%Y%m%d-%H%M%S')}")
    wandb.config.update(vars(par))
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders()
    logger.info(f"Training with {len(train_loader)} batches, validating with {len(val_loader)} batches")
    
    # Create model
    model = RGBVO(imsize1=par.img_h, imsize2=par.img_w, batchNorm=par.batch_norm)
    model = model.to(device)
    
    # Load FlowNet weights if available
    if par.pretrained_flownet and os.path.exists(par.pretrained_flownet):
        load_pretrained_flownet(model, par.pretrained_flownet)
    
    # Create optimizer
    if par.optim['opt'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=par.optim['lr'], weight_decay=par.optim['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=par.optim['lr'], momentum=0.9, weight_decay=par.optim['weight_decay'])
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Train for specified number of epochs
    for epoch in range(par.epochs):
        logger.info(f"Epoch {epoch+1}/{par.epochs}")
        
        # Train
        train_loss, train_rot_loss, train_trans_loss, train_loss_std = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_rot_loss, val_trans_loss, val_loss_std, metrics = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss_mean': train_loss,
            'train_loss_std': train_loss_std,
            'train_rot_loss': train_rot_loss,
            'train_trans_loss': train_trans_loss,
            'val_loss_mean': val_loss,
            'val_loss_std': val_loss_std,
            'val_rot_loss': val_rot_loss,
            'val_trans_loss': val_trans_loss,
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
        logger.info(f"ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m")
        logger.info(f"RPE Trans: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m")
        logger.info(f"RPE Rot: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), par.model_path)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(par.model_dir, f"{par.model_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # Plot and save loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Validation Loss')
            plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.6f}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(par.log_dir, f"{par.model_name}_loss_curve.png"))
            plt.close()
    
    # Finish training
    wandb.finish()
    logger.info("Training completed!")

if __name__ == "__main__":
    train()
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime

from config import config, train_config
from data import get_data_loaders
from models import get_model
from losses import dRMAE, align_svd_mae, batched_dRMAE, batched_align_svd_mae, align_svd_mae

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_wandb(config):
    """Initialize wandb for experiment tracking"""
    run_name = f"RNA3D_finetuned_3090_600e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="RNA_Structure_Prediction",
        name=run_name,
        config={
            **config,
            **train_config
        }
    )
    
    # Define metrics to track
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="step")
    wandb.define_metric("train/learning_rate", step_metric="step")
    wandb.define_metric("val/loss", step_metric="epoch")

def create_3d_structure_plot(gt_coords, pred_coords, title):
    """
    Create a 3D interactive plot comparing ground truth and predicted RNA structures,
    with coloring based on sequence position
    
    Args:
        gt_coords: Ground truth coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Align predicted coordinates to ground truth for better visualization
    # Convert to torch tensors if they are numpy arrays
    if isinstance(gt_coords, np.ndarray):
        gt_coords_tensor = torch.tensor(gt_coords)
    else:
        gt_coords_tensor = gt_coords
        
    if isinstance(pred_coords, np.ndarray):
        pred_coords_tensor = torch.tensor(pred_coords)
    else:
        pred_coords_tensor = pred_coords
    
    # Handle NaN values if any
    valid_mask = ~torch.isnan(gt_coords_tensor.sum(dim=1))
    
    if valid_mask.sum() > 0:
        gt_valid = gt_coords_tensor[valid_mask]
        pred_valid = pred_coords_tensor[valid_mask]
        
        # Compute centroids
        gt_centroid = gt_valid.mean(dim=0, keepdim=True)
        pred_centroid = pred_valid.mean(dim=0, keepdim=True)
        
        # Center the structures
        gt_centered = gt_valid - gt_centroid
        pred_centered = pred_valid - pred_centroid
        
        # Compute optimal rotation
        H = pred_centered.t() @ gt_centered
        U, S, Vt = torch.svd(H)
        R = Vt.t() @ U.t()
        
        # Apply rotation to align prediction with ground truth
        aligned_pred = (pred_centered @ R) + gt_centroid
        
        # Use the aligned prediction
        pred_coords_plot = aligned_pred.numpy()
        gt_coords_plot = gt_valid.numpy()
        
        # Create position indices for coloring (normalized to 0-1 range)
        seq_positions = np.arange(len(gt_coords_plot))
        position_colors = seq_positions / max(1, len(seq_positions) - 1)
    else:
        # If no valid coordinates, just use the raw coordinates
        pred_coords_plot = pred_coords_tensor.numpy()
        gt_coords_plot = gt_coords_tensor.numpy()
        
        # Create position indices for coloring
        seq_positions = np.arange(len(gt_coords_plot))
        position_colors = seq_positions / max(1, len(seq_positions) - 1)
    
    # Create figure with two subplots (side by side)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Ground Truth", "Predicted")
    )
    
    # Add ground truth trace with position-based coloring
    fig.add_trace(
        go.Scatter3d(
            x=gt_coords_plot[:, 0],
            y=gt_coords_plot[:, 1],
            z=gt_coords_plot[:, 2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=position_colors,
                colorscale='Turbo',  # Rainbow-like colorscale
                opacity=0.8,
                colorbar=dict(
                    title="Sequence Position",
                    x=0.45,
                    thickness=20
                )
            ),
            line=dict(
                color='blue',
                width=2
            ),
            name='Ground Truth'
        ),
        row=1, col=1
    )
    
    # Add predicted trace with same position-based coloring
    fig.add_trace(
        go.Scatter3d(
            x=pred_coords_plot[:, 0],
            y=pred_coords_plot[:, 1],
            z=pred_coords_plot[:, 2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=position_colors,
                colorscale='Turbo',  # Same colorscale as ground truth
                opacity=0.8,
                showscale=False  # Don't show duplicate colorbar
            ),
            line=dict(
                color='red',
                width=2
            ),
            name='Predicted'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        width=1200,
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        scene2=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    return fig

def log_structure_predictions(val_preds, epoch):
    """Log 3D visualization of first 5 RNA structure predictions"""
    if epoch % 5 == 0 or epoch == 0:
        for i in range(min(5, len(val_preds))):
            gt_coords, pred_coords = val_preds[i]
            title = f"RNA Structure #{i+1} - Epoch {epoch+1}"
            fig = create_3d_structure_plot(gt_coords, pred_coords, title)
            wandb.log({f"structure_{i+1}_epoch_{epoch+1}": wandb.Html(fig.to_html())})

def train_epoch(model, train_loader, optimizer, scheduler, batch_size, grad_clip, epoch, cos_epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    step = epoch * len(train_loader)
    tbar = tqdm(train_loader)
    
    torch.autograd.set_detect_anomaly(True)
    
    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].cuda()
        gt_xyz = batch['xyz'].cuda()
        mask = batch['mask'].cuda()

        pred_xyzs = model(sequence, mask)

        loss = 0
        for pred_xyz in pred_xyzs:
            loss += batched_dRMAE(pred_xyz, gt_xyz, mask) 
            loss += batched_align_svd_mae(pred_xyz, gt_xyz, mask)
        
        loss.backward()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate if we're past the cosine annealing start epoch
        if (epoch + 1) > cos_epoch:
            scheduler.step()
        
        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "step": step + idx,
        })
                
        total_loss += loss.item()
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)}")
    
    return total_loss / len(tbar)

def validate(model, val_loader):
    """Validate the model on the validation set"""
    model.eval()
    val_loss = 0
    val_preds = []
    tbar = tqdm(val_loader)
    
    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].cuda()
        gt_xyz = batch['xyz'].cuda()
        mask = batch['mask'].cuda()

        with torch.no_grad():
            pred_xyzs = model(sequence, mask)
            pred_xyz = pred_xyzs[-1]  # Take the final prediction
            loss = batched_dRMAE(pred_xyz, gt_xyz, mask)
            
        val_loss += loss.item()
        
        # Save predictions for later analysis
        for b in range(sequence.shape[0]):
            val_preds.append([
                gt_xyz[b, mask[b]].cpu().numpy(),
                pred_xyz[b, mask[b]].cpu().numpy()
            ])
    
    val_loss = val_loss / len(tbar)
    return val_loss, val_preds

def train_model():
    """Main training function"""
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Initialize wandb
    init_wandb(config)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(config)
    
    # Initialize model
    model = get_model(config['model_config_path'], pretrained=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(train_config['epochs'] - train_config['cos_epoch']) * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_preds = None
    
    for epoch in range(train_config['epochs']):
        val_loss, val_preds = validate(model, val_loader)

        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=train_config['batch_size'],
            grad_clip=train_config['grad_clip'],
            epoch=epoch,
            cos_epoch=train_config['cos_epoch']
        )
        
        # Validate
        val_loss, val_preds = validate(model, val_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
        })
        
        # Log 3D structure visualizations
        if epoch == 0 or (epoch + 1) % 5 == 0:
            log_structure_predictions(val_preds, epoch)
        
        print(f"Epoch {epoch + 1}/{train_config['epochs']}, "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds
            torch.save(model.state_dict(), 'RibonanzaNet-3D.pt')
            
            # Log best model to wandb
            # wandb.log({"best_model": wandb.Artifact(
            #     name=f"best_model_{wandb.run.id}",
            #     type="model",
            #     description=f"Best model with validation loss {best_val_loss:.6f}"
            # )})
            
            print(f"New best model saved with val loss: {best_val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'RibonanzaNet-3D-final.pt')
    print(f"Final model saved. Best validation loss: {best_val_loss:.6f}")
    
    # Finish wandb run
    wandb.finish()
    
    return best_preds

if __name__ == "__main__":
    train_model() 
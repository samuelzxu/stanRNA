import torch

def calculate_distance_matrix(X, Y, epsilon=1e-4):
    """Calculate pairwise distance matrix between two sets of points"""
    return (torch.square(X[:, None] - Y[None, :]) + epsilon).sum(-1).sqrt()

def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    """Distance Root Mean Square Deviation loss"""
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()] = False

    if d_clamp is not None:
        rmsd = (torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon).clip(0, d_clamp**2)
    else:
        rmsd = torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon

    return rmsd.sqrt().mean() / Z

def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    """Local Distance Root Mean Square Deviation loss"""
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) * (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0]).bool()] = False

    rmsd = torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon
    return rmsd.sqrt().mean() / Z

def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    """Distance Root Mean Absolute Error loss"""
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()] = False

    rmsd = torch.abs(pred_dm[mask] - gt_dm[mask])
    return rmsd.mean() / Z

def align_svd_mae(input, target, Z=10):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes MAE loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    # Mask
    mask = ~torch.isnan(target.sum(-1))
    input = input[mask]
    target = target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    return torch.abs(aligned_input - target).mean() / Z 

def batched_dRMAE(pred_xyz, gt_xyz, mask, epsilon=1e-4, Z=10):
    """
    Batched distance root mean absolute error loss
    
    Args:
        pred_xyz: [batch_size, seq_len, 3] predicted coordinates
        gt_xyz: [batch_size, seq_len, 3] ground truth coordinates
        mask: [batch_size, seq_len] boolean mask for valid positions
        
    Returns:
        Loss value
    """
    batch_losses = []
    
    for b in range(pred_xyz.shape[0]):
        # Extract valid positions
        valid_pred = pred_xyz[b, mask[b]]
        valid_gt = gt_xyz[b, mask[b]]
        
        # Skip empty sequences
        if valid_pred.shape[0] == 0:
            continue
            
        # Calculate loss for this sequence
        loss = dRMAE(valid_pred, valid_pred, valid_gt, valid_gt, epsilon, Z)
        batch_losses.append(loss)
    
    # Return average loss across the batch
    if len(batch_losses) == 0:
        return torch.tensor(0.0).to(pred_xyz.device)
    
    return torch.stack(batch_losses).mean()

def batched_align_svd_mae(pred_xyz, gt_xyz, mask, Z=10):
    """
    Batched SVD alignment and MAE loss
    
    Args:
        pred_xyz: [batch_size, seq_len, 3] predicted coordinates
        gt_xyz: [batch_size, seq_len, 3] ground truth coordinates
        mask: [batch_size, seq_len] boolean mask for valid positions
        
    Returns:
        Loss value
    """
    batch_losses = []
    
    for b in range(pred_xyz.shape[0]):
        # Extract valid positions
        valid_pred = pred_xyz[b, mask[b]]
        valid_gt = gt_xyz[b, mask[b]]
        
        # Skip empty sequences
        if valid_pred.shape[0] == 0:
            continue
            
        # Calculate loss for this sequence
        loss = align_svd_mae(valid_pred, valid_gt, Z)
        batch_losses.append(loss)
    
    # Return average loss across the batch
    if len(batch_losses) == 0:
        return torch.tensor(0.0).to(pred_xyz.device)
    
    return torch.stack(batch_losses).mean() 
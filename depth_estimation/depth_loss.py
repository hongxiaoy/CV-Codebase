# Adapted from https://github.com/NVlabs/FB-BEV/blob/main/mmdet3d/models/fbbev/modules/depth_net.py#L437
import torch
import torch.nn.functional as F

from get_downsampled_gt_depth import get_downsampled_gt_depth


def get_depth_loss(depth_labels, depth_preds, depth_channels, loss_depth_weight, downsample, grid_config):
    """Compute the depth loss between depth labels and predictions.  
  
    Args:  
        depth_labels (torch.Tensor): The ground truth depth labels, with shape [B, N, H, W], should be a full 
                                        scale depth map.
        depth_preds (torch.Tensor): The predicted depth values, with shape [B, N, C, H', W'], can be a downsampled 
                                        scale of depth map.
        depth_channels (int): The number of depth channels.
        loss_depth_weight (float): The weight for the depth loss.
        downsample (int): The downsampled factor of depth map.
        grid_config (dict): The config of depth discretization.
  
    Returns:  
        dict: A dictionary containing the depth loss, with the key 'loss_depth'.  
  
    """
    # [B*N*H'*W', C]
    depth_labels = get_downsampled_gt_depth(depth_labels, downsample, depth_channels, grid_config)
    depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, depth_channels)
    fg_mask = torch.max(depth_labels, dim=1).values > 0.0
    depth_labels = depth_labels[fg_mask]
    depth_preds = depth_preds[fg_mask]
    
    depth_loss = F.binary_cross_entropy(
        depth_preds,
        depth_labels,
        reduction='none',
    ).sum() / max(1.0, fg_mask.sum())
    return dict(loss_depth=loss_depth_weight * depth_loss)

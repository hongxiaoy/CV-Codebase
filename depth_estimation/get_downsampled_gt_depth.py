# Adapted from https://github.com/NVlabs/FB-BEV/blob/main/mmdet3d/models/fbbev/modules/depth_net.py#L396
import torch
import torch.nn.functional as F


def get_downsampled_gt_depth(gt_depths, downsample, depth_channels, grid_config, sid=False):
        """Performs downsampling on a depth map and applies subsequent processing.  
  
        Args:  
            gt_depths (torch.Tensor): A 4D tensor of shape [B, N, H, W], where B is the batch size,  
                                        N is the sequence length, and H and W are the height and width, 
                                        respectively.
            downsample (int): The factor for downsampling the depth map.
            depth_channels (int): The number of depth channels in the output tensor.
            grid_config (dict): A dictionary containing configuration for the depth grid.  
                - depth (list[float]): A list of three elements representing the minimum depth value,  
                                    maximum depth value, and the depth interval, respectively.
            sid (bool): A flag indicating whether to apply Spatial Increasing Discretization (SID) method.
    
        Returns:  
            torch.Tensor: A 2D tensor of shape [B*N*h*w, d] after downsampling and subsequent processing, 
                            where h and w are the new height and width after downsampling, respectively.
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        # pixel where depth value is 0 means infinite distance
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
        if not sid:
            gt_depths = (gt_depths - (grid_config['depth'][0] -
                                      grid_config['depth'][2])) / \
                        grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(grid_config['depth'][0]).float())
            gt_depths = gt_depths * (depth_channels - 1) / torch.log(
                torch.tensor(grid_config['depth'][1] - 1.).float() /
                grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        
        gt_depths = torch.where((gt_depths < depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=depth_channels + 1).view(-1, depth_channels + 1)[:,
                                                                           1:]
        return gt_depths.float()

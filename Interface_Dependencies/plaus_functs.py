import torch
import numpy as np
from plot_functs import * 
from plot_functs import imshow
from plot_functs import normalize_tensor
import math   
import time
import matplotlib.path as mplPath
from matplotlib.path import Path
from utils.general import scale_coords, xyxy2xywh
import torchvision

def get_gradient(img, grad_wrt, norm=True, absolute=True, grayscale=True, keepmean=False):
    """
    Compute the gradient of an image with respect to a given tensor.

    Args:
        img (torch.Tensor): The input image tensor.
        grad_wrt (torch.Tensor): The tensor with respect to which the gradient is computed.
        norm (bool, optional): Whether to normalize the gradient. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the gradient to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean value of the attribution map. Defaults to False.

    Returns:
        torch.Tensor: The computed attribution map.

    """
    grad_wrt_outputs = torch.ones_like(grad_wrt)
    gradients = torch.autograd.grad(grad_wrt, img, 
                                    grad_outputs=grad_wrt_outputs, 
                                    retain_graph=True,
                                    # create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                    )
    attribution_map = gradients[0]
    if absolute:
        attribution_map = torch.abs(attribution_map) # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        attribution_map = torch.sum(attribution_map, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(attribution_map)
            attmin = torch.min(attribution_map)
            attmax = torch.max(attribution_map)
        attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
        if keepmean:
            attribution_map -= attribution_map.mean()
            attribution_map += (attmean / (attmax - attmin))
        
    return attribution_map

def get_gaussian(img, grad_wrt, norm=True, absolute=True, grayscale=True, keepmean=False):
    """
    Generate Gaussian noise based on the input image.

    Args:
        img (torch.Tensor): Input image.
        grad_wrt: Gradient with respect to the input image.
        norm (bool, optional): Whether to normalize the generated noise. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the noise to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean of the noise. Defaults to False.

    Returns:
        torch.Tensor: Generated Gaussian noise.
    """
    
    gaussian_noise = torch.randn_like(img)
    
    if absolute:
        gaussian_noise = torch.abs(gaussian_noise) # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        gaussian_noise = torch.sum(gaussian_noise, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(gaussian_noise)
            attmin = torch.min(gaussian_noise)
            attmax = torch.max(gaussian_noise)
        gaussian_noise = normalize_batch(gaussian_noise) # Normalize attribution maps per image in batch
        if keepmean:
            gaussian_noise -= gaussian_noise.mean()
            gaussian_noise += (attmean / (attmax - attmin))
        
    return gaussian_noise

#Corners must be true here
def get_plaus_score(imgs, targets_out, attr, debug=False, corners=True):
    """
    Calculates the plausibility score based on the given inputs. This is a 
    metric that evaluates how well the model's attribution maps align with 
    its own predictions of where the bounding boxes are.

    Args:
        imgs (torch.Tensor): The input images.
        targets_out (torch.Tensor): The output targets.
        attr (torch.Tensor): The attribute tensor.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        torch.Tensor: The plausibility score.
    """
    
    # Resize images
    imgs_resized = torch.nn.functional.interpolate(imgs, size=(480, 480), mode='bilinear', align_corners=False)

    # Resize attribution maps
    attr_resized = torch.nn.functional.interpolate(attr, size=(480, 480), mode='bilinear', align_corners=False)

    target_inds = targets_out[:, 0].int().to(imgs_resized.device)
    xyxy_batch = targets_out[:, 2:6].to(imgs_resized.device)# * pre_gen_gains[out_num]
    num_pixels = torch.tile(torch.tensor([imgs_resized.shape[2], imgs_resized.shape[3], imgs_resized.shape[2], imgs_resized.shape[3]], device=imgs_resized.device), (xyxy_batch.shape[0], 1))
    xyxy_corners = (corners_coords_batch(xyxy_batch) * num_pixels).int()
    co = xyxy_corners
    if corners:
        co = targets_out[:, 2:6].int()
    coords_map = torch.zeros_like(attr_resized, dtype=torch.bool)
    x1, x2 = co[:,1], co[:,3]
    y1, y2 = co[:,0], co[:,2]
    
    for ic in range(co.shape[0]):
        coords_map[target_inds[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True
    
    if torch.isnan(attr_resized).any():
        attr_resized = torch.nan_to_num(attr_resized, nan=0.0)
    if debug:
        for i in range(len(coords_map)):
            coords_map3ch = torch.cat([coords_map[i][:1], coords_map[i][:1], coords_map[i][:1]], dim=0)
            coords_map3ch = coords_map3ch.to(imgs_resized[i].device)
            test_bbox = torch.zeros_like(imgs_resized[i])
            test_bbox[coords_map3ch] = imgs_resized[i][coords_map3ch]
            imshow(test_bbox, save_path='figs/test_bbox')
            imshow(imgs_resized[i], save_path='figs/im0')
            imshow(attr_resized[i], save_path='figs/attr')
    
    IoU_num = (torch.sum(attr_resized[coords_map]))
    IoU_denom = torch.sum(attr_resized)
    IoU_ = (IoU_num / IoU_denom)
    plaus_score = IoU_
    if torch.isnan(plaus_score).any():
        plaus_score = torch.tensor(0.0, device=imgs_resized.device)
    return plaus_score

def point_in_polygon(poly, grid):
    # t0 = time.time()
    num_points = poly.shape[0]
    j = num_points - 1
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    for i in range(num_points):
        cond1 = (poly[i, 1] < grid[..., 1]) & (poly[j, 1] >= grid[..., 1])
        cond2 = (poly[j, 1] < grid[..., 1]) & (poly[i, 1] >= grid[..., 1])
        cond3 = (grid[..., 0] - poly[i, 0]) < (poly[j, 0] - poly[i, 0]) * (grid[..., 1] - poly[i, 1]) / (poly[j, 1] - poly[i, 1])
        oddNodes = oddNodes ^ (cond1 | cond2) & cond3
        j = i
    # t1 = time.time()
    # print(f'point in polygon time: {t1-t0}')
    return oddNodes
    
def point_in_polygon_gpu(poly, grid):
    num_points = poly.shape[0]
    i = torch.arange(num_points)
    j = (i - 1) % num_points
    # Expand dimensions
    # t0 = time.time()
    poly_expanded = poly.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid.shape[0], grid.shape[0])
    # t1 = time.time()
    cond1 = (poly_expanded[i, 1] < grid[..., 1]) & (poly_expanded[j, 1] >= grid[..., 1])
    cond2 = (poly_expanded[j, 1] < grid[..., 1]) & (poly_expanded[i, 1] >= grid[..., 1])
    cond3 = (grid[..., 0] - poly_expanded[i, 0]) < (poly_expanded[j, 0] - poly_expanded[i, 0]) * (grid[..., 1] - poly_expanded[i, 1]) / (poly_expanded[j, 1] - poly_expanded[i, 1])
    # t2 = time.time()
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    cond = (cond1 | cond2) & cond3
    # t3 = time.time()
    # efficiently perform xor using gpu and avoiding cpu as much as possible
    c = []
    while len(cond) > 1: 
        if len(cond) % 2 == 1: # odd number of elements
            c.append(cond[-1])
            cond = cond[:-1]
        cond = torch.bitwise_xor(cond[:int(len(cond)/2)], cond[int(len(cond)/2):])
    for c_ in c:
        cond = torch.bitwise_xor(cond, c_)
    oddNodes = cond
    # t4 = time.time()
    # for c in cond:
    #     oddNodes = oddNodes ^ c
    # print(f'expand time: {t1-t0} | cond123 time: {t2-t1} | cond logic time: {t3-t2} |  bitwise xor time: {t4-t3}')
    # print(f'point in polygon time gpu: {t4-t0}')
    # oddNodes = oddNodes ^ (cond1 | cond2) & cond3
    return oddNodes


def bitmap_for_polygon(poly, h, w):
    y = torch.arange(h).to(poly.device).float()
    x = torch.arange(w).to(poly.device).float()
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    bitmap = point_in_polygon(poly, grid)
    return bitmap.unsqueeze(0)


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])

def corners_coords_batch(center_xywh):
    center_x, center_y = center_xywh[:,0], center_xywh[:,1]
    w, h = center_xywh[:,2], center_xywh[:,3]
    x = center_x - w/2
    y = center_y - h/2
    return torch.stack([x, y, x+w, y+h], dim=1)
    
def normalize_batch(x):
    """
    Normalize a batch of tensors along each channel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input.
    """
    mins = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    maxs = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    for i in range(x.shape[0]):
        mins[i] = x[i].min()
        maxs[i] = x[i].max()
    x_ = (x - mins) / (maxs - mins)
    
    return x_
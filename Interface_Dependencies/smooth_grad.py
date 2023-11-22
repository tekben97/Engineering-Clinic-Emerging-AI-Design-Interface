## Original Implementation of Smooth_grad **NOT USED**

import torch
import numpy as np

def generate_vanilla_grad(model, input_tensor, outputNum, targets=None, norm=False, device='cpu'):
    """
    Generates an attribution map using vanilla gradient method.

    Args:
        model (torch.nn.Module): The PyTorch model to generate the attribution map for.
        input_tensor (torch.Tensor): The input tensor to the model.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to False.
        device (str, optional): The device to use for the computation. Defaults to 'cpu'.

    Returns:
        numpy.ndarray: The attribution map.
    """
    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True
    
    # Forward pass
    train_out = model(input_tensor) # training outputs (no inference outputs in train mode)

    num_classes = 2
    
    # Zero gradients
    model.zero_grad()
    
    import torch

    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) #anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) #anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) #anchorx1) obj (objectness score or confidence)
    
    gradients = torch.autograd.grad(train_out[outputNum-1].requires_grad_(True), input_tensor, 
                                    grad_outputs=torch.ones_like(train_out[outputNum-1]).requires_grad_(True), 
                                    retain_graph=True, create_graph=True)
    
    # Convert gradients to numpy array
    gradients = gradients[0].detach().cpu().numpy()

    if norm:
        # Take absolute values of gradients
        gradients = np.absolute(gradients)

        # Sum across color channels
        attribution_map = np.sum(gradients, axis=0)

        # Normalize attribution map
        attribution_map /= np.max(attribution_map)
    else:
        # Sum across color channels
        attribution_map = gradients
    
    return torch.tensor(attribution_map, dtype=torch.float32, device=device)

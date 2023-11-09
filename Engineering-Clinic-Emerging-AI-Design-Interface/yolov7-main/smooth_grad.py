import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np



# Gradient Generating Method
def returnGrad(img, model, criterion, device = 'cpu'):
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True).retain_grad()
    #ya mama
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
    loss.backward()
        
    #    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
        
    return Sc_dx

# Gradient Generating Method for SmoothGrad
def returnSmoothGrad(img, model, augment=None, num_samples=2, sigma=0.2, device='cpu'):
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True)
    
    total_gradient = torch.zeros_like(img)
    # model.train()
    for _ in range(num_samples):
        noisy_img = img + torch.randn_like(img) * sigma
        noisy_img.requires_grad_(True)
        # model.train()
        train_out = model(noisy_img, augment=augment)[1]
        model.zero_grad()
        # loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
        # loss.backward()
        train_out.requires_grad_(True)
        grad = torch.autograd.grad(train_out,
                                    noisy_img,
                                    grad_outputs=torch.ones_like(train_out),create_graph=True,retain_graph=True)
        total_gradient += grad[0]
    model.eval()
    avg_gradient = total_gradient / num_samples
    return avg_gradient

def generate_vanilla_grad(model, input_tensor, targets=None, norm=False, device='cpu'):
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
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    gradients = torch.autograd.grad(train_out[1].requires_grad_(True), input_tensor, 
                                    grad_outputs=torch.ones_like(train_out[1]).requires_grad_(True), 
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

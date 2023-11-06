import torch
import torchvision
import torchvision.transforms as transforms



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

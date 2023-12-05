import torch
import numpy as np
from plot_functs import * 
from plot_functs import normalize_tensor
import math   
import time

def generate_other_grad(model, input_tensor, loss_func = None, 
                          targets=None, metric=None, out_num = 1, 
                          norm=False, device='cpu'):    
    """
    Computes the vanilla gradient of the input tensor with respect to the output of the given model.

    Args:
        model (torch.nn.Module): The model to compute the gradient with respect to.
        input_tensor (torch.Tensor): The input tensor to compute the gradient for.
        loss_func (callable, optional): The loss function to use. If None, the gradient is computed with respect to the output tensor.
        targets (torch.Tensor, optional): The target tensor to use with the loss function. Defaults to None.
        metric (callable, optional): The metric function to use with the loss function. Defaults to None.
        out_num (int, optional): The index of the output tensor to compute the gradient with respect to. Defaults to 1.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to False.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: The attribution map computed as the gradient of the input tensor with respect to the output tensor.
    """
    # Set model.train() at the beginning and revert back to original mode (model.eval() or model.train()) at the end
    train_mode = model.training
    if not train_mode:
        model.train()

    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True
    
    # Zero gradients
    model.zero_grad()

    # Forward pass
    train_out = model(input_tensor) # training outputs (no inference outputs in train mode)
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    if loss_func is None:
        grad_wrt = train_out[out_num]
        grad_wrt_outputs = torch.ones_like(grad_wrt)
    else:
        loss, loss_items = loss_func(train_out, targets.to(device), input_tensor, metric=metric)  # loss scaled by batch_size
        grad_wrt = loss
        grad_wrt_outputs = None
        # loss.backward(retain_graph=True, create_graph=True)
        # gradients = input_tensor.grad
    
    gradients = torch.autograd.grad(grad_wrt, input_tensor, 
                                        grad_outputs=grad_wrt_outputs, 
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

    # Set model back to original mode
    if not train_mode:
        model.eval()
    
    return torch.tensor(attribution_map, dtype=torch.float32, device=device)

def eval_plausibility(imgs, targets, attr_tensor, device, debug=False):
    """
    Evaluate the plausibility of an object detection prediction by computing the Intersection over Union (IoU) between
    the predicted bounding box and the ground truth bounding box.

    Args:
        im0 (numpy.ndarray): The input image.
        targets (list): A list of targets, where each target is a list containing the class label and the ground truth
            bounding box coordinates in the format [class_label, x1, y1, x2, y2].
        attr (torch.Tensor): A tensor containing the normalized attribute values for the predicted
            bounding box.

    Returns:
        float: The total IoU score for all predicted bounding boxes.
    """
    # if len(targets) == 0:
    #     return 0
    # MIGHT NEED TO NORMALIZE OR TAKE ABS VAL OF ATTR
    # ALSO MIGHT NORMALIZE FOR THE SIZE OF THE BBOX
    eval_totals = 0
    plaus_num_nan = 0
    eval_individual_data = []
    # targets_ = [[targets[i] for i in range(len(targets)) if int(targets[i][0]) == j] for j in range(int(max(targets[:,0])))]
    for i, im0 in enumerate(imgs):
        if len(targets[i]) == 0:
            eval_individual_data.append([torch.tensor(0).to(device),])
        else:
            IoU_list = []
            xyxy_pred = targets[i][-4:] # * torch.tensor([im0.shape[2], im0.shape[1], im0.shape[2], im0.shape[1]])
            print(xyxy_pred)
            xyxy_center = corners_coords(xyxy_pred) * torch.tensor([im0.shape[1], im0.shape[2], im0.shape[1], im0.shape[2]])
            c1, c2 = (int(xyxy_center[0]), int(xyxy_center[1])), (int(xyxy_center[2]), int(xyxy_center[3]))
            attr = (normalize_tensor(torch.abs(attr_tensor[i].clone().detach())))
            if torch.isnan(attr).any():
                attr = torch.nan_to_num(attr, nan=0.0)
            IoU_num = (torch.sum(attr[:,c1[1]:c2[1], c1[0]:c2[0]]))
            IoU_denom = torch.sum(attr)
            
            IoU_ = (IoU_num / IoU_denom)
            if debug:
                IoU = IoU_ if not math.isnan(IoU_) else 0.0
                plaus_num_nan += 1 if math.isnan(IoU_) else 0
            else:
                IoU = IoU_
            IoU_list.append(IoU.clone().detach().cpu())
        list_mean = torch.mean(torch.tensor(IoU_list))
        eval_totals += list_mean if not math.isnan(list_mean) else 0.0
        eval_individual_data.append(IoU_list)

    if debug:
        return torch.tensor(eval_totals).requires_grad_(True), plaus_num_nan
    else:
        return torch.tensor(eval_totals).requires_grad_(True), eval_individual_data


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])
    
def generate_vanilla_grad(model, input_tensor, loss_func = None, 
                          targets=None, metric=None, out_num = 1, 
                          norm=False, device='cpu'):    
    """
    Computes the vanilla gradient of the input tensor with respect to the output of the given model.

    Args:
        model (torch.nn.Module): The model to compute the gradient with respect to.
        input_tensor (torch.Tensor): The input tensor to compute the gradient for.
        loss_func (callable, optional): The loss function to use. If None, the gradient is computed with respect to the output tensor.
        targets (torch.Tensor, optional): The target tensor to use with the loss function. Defaults to None.
        metric (callable, optional): The metric function to use with the loss function. Defaults to None.
        out_num (int, optional): The index of the output tensor to compute the gradient with respect to. Defaults to 1.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to False.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The attribution map computed as the gradient of the input tensor with respect to the output tensor.
    """
    # maybe add model.train() at the beginning and model.eval() at the end of this function

    # Set requires_grad attribute of tensor. Important for computing gradients
    input_tensor.requires_grad = True
    
    # Zero gradients
    model.zero_grad()

    # Forward pass
    train_out = model(input_tensor) # training outputs (no inference outputs in train mode)
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    out_num = out_num - 1
    
    if loss_func is None:
        grad_wrt = train_out[out_num]
        grad_wrt_outputs = torch.ones_like(grad_wrt)
    else:
        loss, loss_items = loss_func(train_out, targets.to(device), input_tensor, metric=metric)  # loss scaled by batch_size
        grad_wrt = loss
        grad_wrt_outputs = None
        # loss.backward(retain_graph=True, create_graph=True)
        # gradients = input_tensor.grad
    
    gradients = torch.autograd.grad(grad_wrt, input_tensor, 
                                        grad_outputs=grad_wrt_outputs, 
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

    # Set model back to training mode
    # model.train()
    
    return torch.tensor(attribution_map, dtype=torch.float32, device=device)
import torch
import torch.nn as nn
import numpy as np

def loss_delay_detection_2(p_slice, device):
    n = p_slice.size(0)
    prod = torch.ones(n).to(device)
    p_slice = p_slice.to(device)
    prod[1:] -= p_slice[:-1]
    cumprod = torch.cumprod(prod, dim=0).to(device)
    loss = torch.arange(1, n + 1).to(device) * p_slice * cumprod
    loss = torch.sum(loss)
    return loss


def loss_delay_detection(p_slice, w, device):
    n = p_slice.size(0) 
    prod = torch.ones(n).to(device)
    p_slice = p_slice.to(device)
    
    prod[1:] -= p_slice[:-1].to(device)
    cumprod = torch.cumprod(prod, dim=0).to(device)
    # TODO drop either n or w
    loss = (torch.arange(1, n + 1).to(device) * p_slice * cumprod 
            + (w + 1) * torch.prod(prod[1:]) * (1 - p_slice[-1]))
    loss = torch.sum(loss)
    return loss

def loss_false_alarms(p_slice, device):
    length = len(p_slice)
    loss = 0
    
    start_ind = 0
    end_ind = 0
    
    #while end_ind < length:
    #    start_ind = end_ind - 1
    #    if start_ind < 0:
    #        start_ind = 0
    #    end_ind = np.random.randint(start_ind, length + 2)
    #    if end_ind == start_ind:
    #        end_ind = end_ind + 1
    #    loss += 1 - loss_delay_detection_2(p_slice[start_ind: end_ind + 1], device)   
    loss = 1 - loss_delay_detection_2(p_slice[start_ind: length], device)            
    loss = torch.sum(loss)
    return loss

class CustomLoss(nn.Module):
    
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, outputs, labels):
        loss = torch.zeros(labels.size(0))
        device = outputs.device
        for i, label in enumerate(labels):
            ind = torch.where(label != label[0])[0]
            if ind.size()[0] == 0:
                loss[i] = loss_false_alarms(outputs[i, :], device)
            else:
                w = 8
                alpha = 0.5
                loss[i] = (alpha * loss_delay_detection(outputs[i, ind[0]:(ind[0] + w)], w, device) + 
                           (1 - alpha) * loss_false_alarms(outputs[i, :ind[0]], device))
        loss = torch.mean(loss)
        return loss
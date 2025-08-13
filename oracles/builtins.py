"""
Quantum oracles: built-in (logistic, mse, softmax), custom user API.
"""
import torch

def logistic_oracle(batch, params, indices, bounds):
    # Example: logistic regression loss mapped to [0,1]
    X, y = batch
    logits = X @ params
    pred = torch.sigmoid(logits)
    loss = -(y * torch.log(pred+1e-8) + (1 - y) * torch.log(1 - pred + 1e-8))
    minv, maxv = bounds
    scaled = (loss - minv) / (maxv - minv)
    return torch.clamp(scaled, 0, 1)

def mse_oracle(batch, params, indices, bounds):
    X, y = batch
    preds = X @ params
    loss = (preds - y) ** 2
    minv, maxv = bounds
    scaled = (loss - minv) / (maxv - minv)
    return torch.clamp(scaled, 0, 1)

def softmax_oracle(batch, params, indices, bounds):
    X, y = batch
    logits = X @ params
    pred = torch.softmax(logits, dim=-1)
    loss = -torch.log(pred[range(len(y)), y])
    minv, maxv = bounds
    scaled = (loss - minv) / (maxv - minv)
    return torch.clamp(scaled, 0, 1)

def custom_oracle(batch, params, indices, bounds, fn):
    # Calls user fn(batch, params, indices, bounds) → [0,1]
    return fn(batch, params, indices, bounds)

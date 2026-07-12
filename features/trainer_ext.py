import torch
from torch.optim.lr_scheduler import StepLR

def make_scheduler(optimizer, step_size: int = 1, gamma: float = 0.5):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def clip_grads(model, max_norm: float = 1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
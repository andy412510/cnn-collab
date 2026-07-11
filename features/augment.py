import torch
from torchvision import transforms

def random_rotation(degrees: int = 15):
    return transforms.RandomRotation(degrees)

def random_hflip(p: float = 0.5):
    return transforms.RandomHorizontalFlip(p)

def add_gaussian_noise(std: float = 0.1):
    return transforms.Lambda(lambda x: x + std * torch.randn_like(x))
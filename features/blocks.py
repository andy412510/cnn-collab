import torch.nn as nn
def dropout_block(p: float = 0.25):
 return nn.Dropout(p)
def bn_block(num_features: int = 32):
 return nn.BatchNorm2d(num_features)
def extra_conv_block(in_channels: int = 32, out_channels: int = 64):
 return nn.Sequential(
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
)
"""極簡 CNN 模型定義（MNIST 手寫數字辨識）。

這是協作練習的基礎模型，刻意保持簡單，方便同學在上面擴充功能。
"""
import torch.nn as nn


class MinimalCNN(nn.Module):
    """2 個卷積層 + 1 個全連接層，用於 MNIST（10 類）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Backbone（特徵提取）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> (16, 28, 28)
        self.pool = nn.MaxPool2d(2)                                # -> 解析度減半
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> (32, 14, 14)
        # Head（分類）
        self.fc = nn.Linear(32 * 6 * 6, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # (B, 16, 28, 28)
        x = self.pool(x)               # (B, 16, 14, 14)
        x = self.relu(self.conv2(x))   # (B, 32, 14, 14)
        x = self.pool(x)               # (B, 32, 7, 7)
        x = x.view(x.size(0), -1)      # 展平
        return self.fc(x)              # (B, num_classes) logits


if __name__ == "__main__":
    import torch

    model = MinimalCNN()
    dummy = torch.randn(4, 1, 28, 28)
    print("MinimalCNN 輸出形狀:", model(dummy).shape)  # 期望 (4, 10)

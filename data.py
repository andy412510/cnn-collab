"""MNIST 資料載入與前處理。"""
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def build_transforms():
    """回傳訓練用的前處理管道。

    同學1（資料組）會在這個函式裡加入資料增強。
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的 mean / std
    ])


def get_dataloaders(batch_size: int = 64, root: str = "./data"):
    """下載 MNIST 並回傳 (train_loader, test_loader)。"""
    train_tf = build_transforms()
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    training_set = MNIST(root=root, train=True, download=True, transform=train_tf)
    test_set = MNIST(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, test_loader

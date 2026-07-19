"""訓練主程式。

執行方式：
    python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim

from model import MinimalCNN
from data import get_dataloaders
from utils import accuracy

# =====================================================================
# 功能註冊區
#
# 每位同學完成自己的功能後，必須在下面這個 list 加入一行字串，
# 代表「啟用」自己的功能。因為大家都改同一個位置，
#
# 範例： "augment"  /  "deep_model"  /  "lr_schedule"  /  "eval_tools"
# =====================================================================
ACTIVE_FEATURES = [
    # 各組在此加入自己的功能字串
    "augment",
    "blocks",
    "trainer_ext",
    "evaluate"
]
# =====================================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_acc += accuracy(logits, labels) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        total_acc += accuracy(logits, labels) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_acc / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    print(f"已啟用功能: {ACTIVE_FEATURES or '（無）'}")

    train_loader, test_loader = get_dataloaders(batch_size=64)
    model = MinimalCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}")


if __name__ == "__main__":
    main()

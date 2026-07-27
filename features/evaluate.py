import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes: int = 10):
    """計算混淆矩陣"""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        
        for t, p in zip(labels, preds):
            cm[t.long(), p.long()] += 1
            
    return cm

def plot_curves(history: dict, save_path: str = "curves.png"):
    """繪製 Loss 與 Accuracy 曲線"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # 繪製 Loss
    ax[0].plot(history.get("train_loss", []), label="train")
    ax[0].plot(history.get("val_loss", []), label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    
    # 繪製 Accuracy
    ax[1].plot(history.get("val_acc", []), label="val acc")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"曲線已儲存至: {save_path}")

def save_model(model, path: str = "model.pt"):
    """儲存模型權重"""
    torch.save(model.state_dict(), path)
    print(f"模型已儲存至: {path}")

def load_model(model, path: str = "model.pt", device="cpu"):
    """載入模型權重"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
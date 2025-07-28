import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from config import CFG, label_map
from dataset_txt import CustomTxtDataset  # 你之前写好的 dataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import time


# 标签索引映射
idx2name = {v: k for k, v in label_map.items()}

# 图像预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


# 构建模型
def build_model():
    weights = MobileNet_V2_Weights.DEFAULT  # 加载默认预训练权重
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, CFG.num_classes)
    return model.to(CFG.device)


# 验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds += outputs.argmax(1).cpu().tolist()
            targets += labels.cpu().tolist()
    acc = accuracy_score(targets, preds)
    return total_loss / len(dataloader), acc


# 可视化函数
def plot_curves(train_loss, val_loss, train_acc, val_acc, save_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.show()


# 主训练函数
def train():
    # 数据加载
    train_dataset = CustomTxtDataset(CFG.train_txt, transform=transform)
    val_dataset = CustomTxtDataset(CFG.val_txt, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_acc = 0.0
    early_stop_counter = 0
    save_dir = 'mobilenet'
    os.makedirs(save_dir, exist_ok=True)

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(CFG.num_epochs):
        model.train()
        running_loss = 0.0
        preds, targets = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CFG.num_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds += outputs.argmax(1).detach().cpu().tolist()
            targets += labels.cpu().tolist()

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(targets, preds)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"[Epoch {epoch + 1}/{CFG.num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("✅ Best model saved!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= CFG.patience:
                print("⛔ Early stopping triggered.")
                break

    # 绘图
    plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_dir)


if __name__ == '__main__':
    train()

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Windows 多进程必须加
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models
    from dataset_txt import CustomTxtDataset, transform
    from config import CFG
    import os
    from tqdm import tqdm
    import numpy as np

    os.makedirs('models', exist_ok=True)

    train_dataset = CustomTxtDataset(CFG.train_txt, transform=transform)
    val_dataset = CustomTxtDataset(CFG.val_txt, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, CFG.num_classes)
    )
    model = model.to(CFG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=CFG.lr)

    best_acc = 0.0
    early_stop_counter = 0
    history = []

    for epoch in range(CFG.num_epochs):
        print(f"Epoch {epoch+1}/{CFG.num_epochs}")
        model.train()
        train_loss = 0
        train_correct = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CFG.device), labels.to(CFG.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)

        history.append([train_loss / len(train_dataset), val_loss / len(val_dataset), train_acc, val_acc])

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("✅ Saved best model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= CFG.patience:
                print("⏹️ Early stopping")
                break

    np.save('models/history.npy', np.array(history))

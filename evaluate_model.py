import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_txt import CustomTxtDataset, transform
from config import CFG, label_map
from PIL import Image
import time  # ✅ 新增
import matplotlib
matplotlib.use('TkAgg')  # 避免 PyCharm 的 backend 报错
import csv  # ✅ 加在文件顶部的 import 部分


# ✅ 构建反向映射：从类别名 -> 索引，和 索引 -> 类别名
name2idx = label_map
idx2name = {v: k for k, v in name2idx.items()}

# ✅ 推理函数，封装主流程
def run_inference():
    # 加载模型结构
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, CFG.num_classes)
    )
    model.load_state_dict(torch.load('models/best_model_512_padding.pt', map_location=CFG.device))
    model.to(CFG.device)
    model.eval()




    # ✅ 筛选每类一个文件用于推理
    selected_lines = []
    seen_prefix = set()
    with open(CFG.val_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            path, label = line.split(",", 1)
            filename = os.path.basename(path)

            # ✅ 只保留 xxx.jpg 或 xxx_1.jpg 文件
            if "_" in filename:
                name_part = filename.split(".")[0]
                suffix = name_part.split("_")[-1]
                if suffix != "1":
                    continue
            elif "_" in filename:
                continue  # 有下划线但不是 _1
            # ✅ 添加进筛选
            selected_lines.append((path, label))

    print(f"\n📂 共选择 {len(selected_lines)} 张图片用于推理")

    # 构建数据集
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, selected_data, transform):
            self.data = selected_data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            path, label = self.data[idx]
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            label_idx = name2idx[label]  # ✅ 将 label 映射为整数索引
            return image, label_idx, os.path.basename(path)

    dataset = InferenceDataset(selected_lines, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # 推理
    all_preds, all_labels, all_names = [], [], []
    batch_times = []  # ✅ 用于记录每个 batch 的耗时
    with torch.no_grad():
        for batch_idx, (imgs, labels, names) in enumerate(dataloader):
            start_time = time.time()  # ✅ 开始计时

            imgs = imgs.to(CFG.device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            batch_time = time.time() - start_time  # ✅ 当前 batch 耗时
            batch_times.append(batch_time)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_names.extend(names)

            print(f"🔁 Batch {batch_idx + 1}: 推理 {len(imgs)} 张图，耗时 {batch_time:.4f} 秒")

    # 解码预测值
    decoded_preds = [idx2name[p] for p in all_preds]
    decoded_labels = [idx2name[l] for l in all_labels]

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\n✅ Accuracy: {acc:.4f} | F1-score (macro): {f1:.4f}\n")

    print("📊 512*512 分类报告：")
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys()), digits=4))

    output_path = "test_final_inference_all_on_pdfonly.csv"
    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "predicted", "true", "correct"])
        for name, pred, true in zip(all_names, decoded_preds, decoded_labels):
            correct = "1" if pred == true else "0"
            writer.writerow([name, pred, true, correct])
    print(f"\n✅ 推理结果已保存至 CSV 文件：{output_path}")


# ✅ 主函数保护（兼容 Windows 多进程）
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    run_inference()

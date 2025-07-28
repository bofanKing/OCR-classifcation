import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt
from pdf2image import convert_from_path
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import CFG, label_map
idx2label = {v: k for k, v in label_map.items()}


class CustomTxtDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' not in line:
                    continue
                path, label_str = line.strip().rsplit(',', 1)
                if label_str not in label_map:
                    continue
                label = label_map[label_str]
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(path)
        return image, label, filename


class ResNetUserApp:
    def __init__(self, model_path="models/best_model_512.pt"):
        self.device = CFG.device
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((CFG.image_size, CFG.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.poppler_path = r"bin" #C:\Users\王博凡\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin
        self.temp_dir = "temp_images"
        os.makedirs(self.temp_dir, exist_ok=True)

    def build_model(self):
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, CFG.num_classes)
        )
        return model

    def run_inference(self, output_csv="resnet50_user_infer.csv"):
        dataset = CustomTxtDataset(CFG.val_txt, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False,num_workers=8)

        all_preds, all_labels, file_names = [], [], []

        with torch.no_grad():
            for imgs, labels, names in dataloader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds += preds.cpu().numpy().tolist()
                all_labels += labels.cpu().numpy().tolist()
                file_names += names

        decoded_preds = [idx2label[p] for p in all_preds]
        decoded_labels = [idx2label[l] for l in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"\n✅ ResNet50 Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=list(label_map.keys()), digits=4))

        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted", "true", "correct"])
            for name, pred, true in zip(file_names, decoded_preds, decoded_labels):
                correct = int(pred == true)
                writer.writerow([name, pred, true, correct])
        print(f"📄 推理结果已保存至 {output_csv}")

    def launch_gui(self):
        self.gui = QWidget()
        self.gui.setWindowTitle("📄 ResNet 文档识别（User 版）")
        self.gui.resize(600, 700)

        self.result_area = QScrollArea(self.gui)
        self.result_area.setWidgetResizable(True)
        self.result_area.setFixedHeight(500)

        self.result_label = QLabel("预测结果将显示在这里", self.gui)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: Consolas; font-size: 18px;")
        self.result_area.setWidget(self.result_label)

        self.btn_load = QPushButton("📂 批量上传图像/PDF", self.gui)
        self.btn_load.clicked.connect(self.load_input)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        layout.addWidget(self.result_area)
        self.gui.setLayout(layout)
        self.gui.show()

    def load_input(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.gui, "选择图像或PDF文件", "", "Images (*.jpg *.jpeg *.png *.pdf)"
        )
        if not file_paths:
            return

        img_paths = []
        for path in file_paths[:CFG.batch_size]:
            if path.lower().endswith(".pdf"):
                image = convert_from_path(path, dpi=200, first_page=1, last_page=1, poppler_path=self.poppler_path)[0]
                img_path = os.path.join(self.temp_dir, os.path.basename(path) + "_page1.jpg")
                image.save(img_path, "JPEG")
                img_paths.append(img_path)
            else:
                img_paths.append(path)

        self.predict_images(img_paths)

    def predict_images(self, paths):
        try:
            imgs = [self.transform(Image.open(p).convert("RGB")) for p in paths]
            batch = torch.stack(imgs).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            results = ""
            for path, pred in zip(paths, preds):
                label = idx2label[pred.item()]
                confidence = probs[paths.index(path)][pred].item()
                results += f"文件: {os.path.basename(path)}\n"
                results += f"类别: {label} | 概率: {confidence:.4f}\n\n"

            self.result_label.setText(results.strip())
        except Exception as e:
            self.result_label.setText(f"❌ 推理失败: {e}")


class MobileNetUserApp:
    def __init__(self, model_path="mobilenet/best_model.pth"):
        self.device = CFG.device
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((CFG.image_size, CFG.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.poppler_path = r"bin"  #C:\Users\王博凡\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin
        self.temp_dir = "temp_images"
        os.makedirs(self.temp_dir, exist_ok=True)

    def build_model(self):
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, CFG.num_classes)
        return model

    def run_inference(self, output_csv="mobilenet_user_infer.csv"):
        dataset = CustomTxtDataset(CFG.val_txt, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False,num_workers=8)

        all_preds, all_labels, file_names = [], [], []

        with torch.no_grad():
            for imgs, labels, names in dataloader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds += preds.cpu().numpy().tolist()
                all_labels += labels.cpu().numpy().tolist()
                file_names += names

        decoded_preds = [idx2label[p] for p in all_preds]
        decoded_labels = [idx2label[l] for l in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"\n✅ MobileNet Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=list(label_map.keys()), digits=4))

        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted", "true", "correct"])
            for name, pred, true in zip(file_names, decoded_preds, decoded_labels):
                correct = int(pred == true)
                writer.writerow([name, pred, true, correct])
        print(f"📄 推理结果已保存至 {output_csv}")

    def launch_gui(self):
        self.gui = QWidget()
        self.gui.setWindowTitle("📱 MobileNet 文档识别（User 版）")
        self.gui.resize(600, 700)

        self.result_area = QScrollArea(self.gui)
        self.result_area.setWidgetResizable(True)
        self.result_area.setFixedHeight(500)

        self.result_label = QLabel("预测结果将显示在这里", self.gui)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: Consolas; font-size: 18px;")
        self.result_area.setWidget(self.result_label)

        self.btn_load = QPushButton("📂 批量上传图像/PDF", self.gui)
        self.btn_load.clicked.connect(self.load_input)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        layout.addWidget(self.result_area)
        self.gui.setLayout(layout)
        self.gui.show()

    def load_input(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.gui, "选择图像或PDF文件", "", "Images (*.jpg *.jpeg *.png *.pdf)"
        )
        if not file_paths:
            return

        img_paths = []
        for path in file_paths[:CFG.batch_size]:
            if path.lower().endswith(".pdf"):
                image = convert_from_path(path, dpi=200, first_page=1, last_page=1, poppler_path=self.poppler_path)[0]
                img_path = os.path.join(self.temp_dir, os.path.basename(path) + "_page1.jpg")
                image.save(img_path, "JPEG")
                img_paths.append(img_path)
            else:
                img_paths.append(path)

        self.predict_images(img_paths)

    def predict_images(self, paths):
        try:
            imgs = [self.transform(Image.open(p).convert("RGB")) for p in paths]
            batch = torch.stack(imgs).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            results = ""
            for path, pred in zip(paths, preds):
                label = idx2label[pred.item()]
                confidence = probs[paths.index(path)][pred].item()
                results += f"文件: {os.path.basename(path)}\n"
                results += f"类别: {label} | 概率: {confidence:.4f}\n\n"

            self.result_label.setText(results.strip())
        except Exception as e:
            self.result_label.setText(f"❌ 推理失败: {e}")
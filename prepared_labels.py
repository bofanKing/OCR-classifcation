import os
import random

# 合并类别规则
merge_mapping = {
    'photos': 'vehicle_photos',
    'front_photo': 'vehicle_photos',
    'back_photo': 'vehicle_photos',
    'left_photo': 'vehicle_photos',
    'right_photo': 'vehicle_photos',
    'registration_certificate': 'registration_certificate',
    'driving_license': 'driving_license',
    'passport_files': 'passport_files',
    'contract_files': 'contract_files',
    'invoice_files': 'invoice_files',
    'receipt_files': 'receipt_files'
}

base_dir = 'output_images'
output_train_txt = 'train.txt'
output_val_txt = 'test.txt'
split_ratio = 0.8

all_data = []

# 收集所有图片路径和对应标签
for folder, label in merge_mapping.items():
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, filename)
            all_data.append((full_path, label))

# 打乱并划分
random.shuffle(all_data)
split_index = int(len(all_data) * split_ratio)
train_data = all_data[:split_index]
val_data = all_data[split_index:]

# 写入 train.txt
with open(output_train_txt, 'w', encoding='utf-8') as f:
    for path, label in train_data:
        f.write(f"{path},{label}\n")

# 写入 val.txt
with open(output_val_txt, 'w', encoding='utf-8') as f:
    for path, label in val_data:
        f.write(f"{path},{label}\n")

print(f"✅ 标签文件生成完毕！训练集：{len(train_data)}，验证集：{len(val_data)}")

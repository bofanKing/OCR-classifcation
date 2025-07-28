import os
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置路径
input_root = "original_data"              # 原始图像路径
output_root = "output_images_512_padding" # 输出路径
target_size = (512, 512)
num_threads = 16  # 可根据 CPU 核心数调整
bg_color = (255, 255, 255)  # 填充颜色（白色）

# 确保输出目录存在
os.makedirs(output_root, exist_ok=True)

supported_exts = (".jpg", ".jpeg", ".png")

# 自定义：带 padding 的等比例缩放函数
def resize_with_padding(img, target_size=(512, 512), bg_color=(255, 255, 255)):
    original_width, original_height = img.size
    target_width, target_height = target_size

    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    img_resized = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # 创建背景图（白色）
    new_img = Image.new("RGB", target_size, bg_color)
    paste_position = ((target_width - new_width) // 2, (target_height - new_height) // 2)
    new_img.paste(img_resized, paste_position)
    return new_img

# 收集图像任务
image_tasks = []
folder_image_count = {}

for folder_name in os.listdir(input_root):
    input_folder = os.path.join(input_root, folder_name)
    output_folder = os.path.join(output_root, folder_name)

    if not os.path.isdir(input_folder):
        continue

    os.makedirs(output_folder, exist_ok=True)

    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_exts)]
    folder_image_count[folder_name] = len(img_files)

    for img_name in img_files:
        input_img_path = os.path.join(input_folder, img_name)
        new_img_name = os.path.splitext(img_name)[0] + ".jpg"
        output_img_path = os.path.join(output_folder, new_img_name)
        image_tasks.append((input_img_path, output_img_path, folder_name))

# 图像处理函数
def process_image(task):
    input_path, output_path, folder = task
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img_padded = resize_with_padding(img, target_size, bg_color)
            img_padded.save(output_path, format="JPEG")
        return (True, folder)
    except Exception as e:
        return (False, f"{input_path} 错误: {e}")

# 开始处理
start_time = time.time()
print(f"🔧 准备开始处理 {len(image_tasks)} 张图片，共 {len(folder_image_count)} 个类别")

error_list = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_image, task) for task in image_tasks]

    for future in tqdm(as_completed(futures), total=len(futures), desc="📦 处理中", ncols=100):
        success, info = future.result()
        if not success:
            error_list.append(info)

# 输出错误信息
if error_list:
    print("\n❌ 以下图片处理失败：")
    for e in error_list:
        print(e)
else:
    print("\n✅ 所有图片处理完成！")

# 总结信息
end_time = time.time()
duration = end_time - start_time
print(f"\n⏱️ 总处理时间：{duration:.2f} 秒 ≈ {duration/60:.2f} 分钟")
print(f"📂 每类图片数量：")
for k, v in folder_image_count.items():
    print(f" - {k}: {v} 张")

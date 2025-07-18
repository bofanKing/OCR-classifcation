import os
import json
import pandas as pd
import requests
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ========= ✅ 配置参数 =========
excel_path = '资产.xlsx'
poppler_path = r'C:\Users\王博凡\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'
log_file = 'error_log.txt'

# 图像/PDF输出文件夹
img_root = 'output_images'
os.makedirs(img_root, exist_ok=True)

# PDF列（需转换为图像）
pdf_columns = [
    'registration_certificate', 'driving_license', 'passport_files',
    'contract_files', 'invoice_files', 'receipt_files'
]

# 图片列
img_columns = [
    'photos', 'front_photo', 'left_photo',
    'right_photo', 'back_photo'
]

# 合并列名
all_columns = pdf_columns + img_columns

# 日志
def log_error(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ========= ✅ 工具函数 =========

def download_and_convert_pdf(idx, col, file_info):
    """下载并转换 PDF 为 JPG"""
    name = file_info['name'].replace("，", "_").replace(" ", "_")
    url = file_info['url']
    save_dir = os.path.join(img_root, col)
    os.makedirs(save_dir, exist_ok=True)

    pdf_path = os.path.join(save_dir, f"{idx}_{name}.pdf")
    if not os.path.exists(pdf_path):
        try:
            r = requests.get(url, timeout=10)
            with open(pdf_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"[PDF转换失败] {col} 第{idx}行: {e}")
            log_error(f"[PDF转换失败] {col} 第{idx}行: {e}")
        # except Exception as e:
        #     print(f"[PDF下载失败] {col} 第{idx}行: {e}")
            return

    try:
        images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
        for i, img in enumerate(images):
            img_path = os.path.join(save_dir, f"{idx}_{i + 1}.jpg")
            img.save(img_path, 'JPEG')
    except Exception as e:
        log_error(f"[PDF转换失败] {col} 第{idx}行: {e}")
        print(f"[PDF转换失败] {col} 第{idx}行: {e}")


def download_image(idx, col, url):
    """下载图片"""
    save_dir = os.path.join(img_root, col)
    os.makedirs(save_dir, exist_ok=True)

    try:
        ext = url.split('.')[-1].split('?')[0].lower()
        img_path = os.path.join(save_dir, f"{idx}.{ext}")
        r = requests.get(url, timeout=10)
        with open(img_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"[图片下载失败] {col} 第{idx}行: {e}")
        log_error(f"[图片下载失败] {col} 第{idx}行: {e}")


# ========= ✅ 主处理函数 =========
def process_row(idx, row):
    for col in all_columns:
        if pd.isna(row.get(col, None)):
            continue
        try:
            items = json.loads(row[col])
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if isinstance(item, dict) and 'url' in item and 'name' in item:
                    if col in pdf_columns:
                        download_and_convert_pdf(idx, col, item)
                    else:
                        download_image(idx, col, item['url'])
                elif isinstance(item, str):
                    download_image(idx, col, item)
        except Exception as e:
            print(f"[解析失败] {col} 第{idx}行: {e}")
            log_error(f"[解析失败] {col} 第{idx}行: {e}")


# ========= ✅ 并发执行 =========
if __name__ == "__main__":
    df = pd.read_excel(excel_path)
    with ThreadPoolExecutor(max_workers=6) as executor:  # 你可以改为 8 或更多线程
        list(tqdm(executor.map(lambda i_row: process_row(*i_row), df.iterrows()), total=len(df), desc="批量处理"))

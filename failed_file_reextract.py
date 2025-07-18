import os
import re
import json
import requests
import pandas as pd
from collections import defaultdict
from pdf2image import convert_from_bytes
import pypdfium2
from PIL import Image

# 配置路径
excel_path = '资产.xlsx'
log_path = 'error_log.txt'
output_dir = 'output_images'
poppler_path = r'C:\Users\王博凡\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'
re_failed_log = 're_failed_log.txt'

os.makedirs(output_dir, exist_ok=True)


# 解析错误日志，返回 {列名: set(行号)}
def parse_error_log(path):
    failed_map = defaultdict(set)
    pattern = re.compile(r'\[.*?失败\]\s+(.*?)\s+第(\d+)行')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                col = match.group(1).strip()
                idx = int(match.group(2))
                failed_map[col].add(idx)
    return failed_map


# 尝试将 PDF 二进制转换为多张 JPG 图片
def convert_pdf_to_images(pdf_bytes, save_path_prefix, col, idx):
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200, poppler_path=poppler_path)
        for i, img in enumerate(images):
            img.save(f"{save_path_prefix}_{i + 1}.jpg", "JPEG")
        print(f"✅ pdf2image 成功: {col} 第{idx}行")
        return True
    except Exception as e:
        print(f"⚠️ pdf2image 失败，尝试 pypdfium2: {e}")

    try:
        pdf = pypdfium2.PdfDocument(pdf_bytes)
        for i in range(len(pdf)):
            image = pdf[i].render(scale=2).to_pil()
            image.save(f"{save_path_prefix}_{i + 1}.jpg", "JPEG")
        print(f"✅ pypdfium2 成功: {col} 第{idx}行")
        return True
    except Exception as e:
        print(f"❌ 所有 PDF 转换失败: {col} 第{idx}行: {e}")
        return False


# 主处理逻辑：重处理 PDF 和 图片
def reprocess_failed_files():
    df = pd.read_excel(excel_path)
    failed_map = parse_error_log(log_path)

    with open(re_failed_log, 'w', encoding='utf-8') as logf:
        for col, indices in failed_map.items():
            for idx in sorted(indices):
                try:
                    row = df.loc[idx]
                    files = row[col]
                    if pd.isna(files): continue
                    items = json.loads(files)
                    if not isinstance(items, list): items = [items]

                    for item in items:
                        if not isinstance(item, dict) or 'url' not in item or 'name' not in item:
                            continue

                        url = item['url']
                        name = item['name']
                        ext = os.path.splitext(name)[1].lower()
                        save_dir = os.path.join(output_dir, col)
                        os.makedirs(save_dir, exist_ok=True)
                        file_base = os.path.join(save_dir, f"{idx}")

                        # 如果已存在处理过的 JPG 图像，跳过
                        already = any(f.startswith(f"{idx}_") or f.startswith(f"{idx}.") for f in os.listdir(save_dir))
                        if already:
                            print(f"⚠️ 已存在图像，跳过: {col} 第{idx}行")
                            continue

                        try:
                            r = requests.get(url, timeout=10)
                            r.raise_for_status()
                            content = r.content
                        except Exception as e:
                            print(f"❌ 下载失败: {col} 第{idx}行: {e}")
                            logf.write(f"{col} 第{idx}行 下载失败: {name}\n")
                            continue

                        # PDF 文件
                        if ext == '.pdf':
                            success = convert_pdf_to_images(content, file_base, col, idx)
                        # 图片文件
                        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            try:
                                img_path = f"{file_base}{ext}"
                                with open(img_path, 'wb') as f:
                                    f.write(content)
                                print(f"✅ 图片保存成功: {col} 第{idx}行")
                                success = True
                            except Exception as e:
                                print(f"❌ 图片保存失败: {col} 第{idx}行: {e}")
                                success = False
                        else:
                            print(f"⚠️ 不支持的文件类型: {name}")
                            success = False

                        if not success:
                            logf.write(f"{col} 第{idx}行 处理失败: {name}\n")

                except Exception as e:
                    print(f"⚠️ JSON 解析失败: {col} 第{idx}行: {e}")
                    logf.write(f"{col} 第{idx}行 JSON解析失败\n")


if __name__ == "__main__":
    reprocess_failed_files()

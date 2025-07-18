import os

def rename_pdf_suffix(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg.pdf') or filename.lower().endswith('.pdf.pdf'):
            new_filename = filename[:-4]  # 删除最后的 .pdf
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            if not os.path.exists(new_path):  # 避免覆盖
                os.rename(old_path, new_path)
                print(f"✅ 重命名: {filename} → {new_filename}")
            else:
                print(f"⚠️ 已存在同名文件，跳过: {new_filename}")
        else:
            print(f"🔸 不符合条件，跳过: {filename}")

# 调用方式（改成你自己的路径）
rename_pdf_suffix("output_images/contract_files")


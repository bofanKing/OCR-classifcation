import pandas as pd
import json
import re
from collections import defaultdict

excel_path = '资产.xlsx'
log_path = 'error_log.txt'
output_failed_info = 'to_retry_list.txt'

# 提取失败记录
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

# 输出失败文件名和URL
def dump_failed_file_info():
    df = pd.read_excel(excel_path)
    failed_map = parse_error_log(log_path)
    with open(output_failed_info, 'w', encoding='utf-8') as out:
        for col, indices in failed_map.items():
            for idx in sorted(indices):
                try:
                    row = df.loc[idx]
                    files = row[col]
                    if pd.isna(files): continue
                    items = json.loads(files)
                    if not isinstance(items, list): items = [items]
                    for item in items:
                        if isinstance(item, dict) and 'name' in item and 'url' in item:
                            out.write(f"{col} 第{idx}行\n")
                            out.write(f"文件名: {item['name']}\n")
                            out.write(f"URL: {item['url']}\n\n")
                except Exception as e:
                    out.write(f"{col} 第{idx}行 JSON解析失败: {str(e)}\n\n")
    print(f"✅ 已输出失败信息至 {output_failed_info}")

if __name__ == "__main__":
    dump_failed_file_info()

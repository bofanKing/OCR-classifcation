import os
import matplotlib.pyplot as plt


def count_and_plot_jpg_distribution(data_dir):
    """
    统计每个子目录中 .jpg 文件数量，并绘制分布图。

    参数：
        data_dir (str): 包含子类别目录的根目录路径

    返回：
        dict: 每个类别及其 jpg 数量的字典
    """
    class_counts = {}

    # 遍历每个子文件夹（类别）
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        # 统计 .jpg 文件
        jpg_count = sum(1 for f in os.listdir(cls_path) if f.lower().endswith('.jpg'))
        class_counts[cls] = jpg_count

    # 输出统计信息
    print("\n📊 每个类别中 .jpg 图像数量：")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {count} 张")

    # 可视化部分
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # 按数量排序
    sorted_items = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
    sorted_classes = [x[0] for x in sorted_items]
    sorted_counts = [x[1] for x in sorted_items]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_classes, sorted_counts, color='skyblue')

    # 添加数量标签
    for bar, count in zip(bars, sorted_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=9)

    plt.title("📊 每个类别的 JPG 图像数量分布", fontsize=14)
    plt.xlabel("类别名称（文件夹名）", fontsize=12)
    plt.ylabel("图像数量", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    return class_counts

# ✅ 示例调用方式（修改为你的数据路径）
count_and_plot_jpg_distribution("output_images")

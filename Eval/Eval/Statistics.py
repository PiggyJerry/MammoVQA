import json
from collections import defaultdict

def count_statistics(json_file_path):
    # 初始化统计数据结构
    total_images = set()
    total_items = 0
    dataset_stats = defaultdict(lambda: {"images": set(), "items": 0})
    question_topic_stats = defaultdict(lambda: {"images": set(), "items": 0})

    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 遍历每个 item
    for item_id, item in data.items():
        total_items += 1
        image_path = item['Path']
        dataset = item['Dataset']
        question_topic = item['Question topic']

        # 统计图片数
        total_images.add(image_path)
        dataset_stats[dataset]["images"].add(image_path)
        question_topic_stats[question_topic]["images"].add(image_path)

        # 统计样本数
        dataset_stats[dataset]["items"] += 1
        question_topic_stats[question_topic]["items"] += 1

    # 转换图片集合为图片数
    total_images_count = len(total_images)
    for dataset in dataset_stats:
        dataset_stats[dataset]["images"] = len(dataset_stats[dataset]["images"])
    for question_topic in question_topic_stats:
        question_topic_stats[question_topic]["images"] = len(question_topic_stats[question_topic]["images"])

    return total_images_count, total_items, dataset_stats, question_topic_stats

# 使用示例
if __name__ == "__main__":
    json_file_path = '/home/jiayi/MammoVQA/Benchmark/MammoVQA-Image-Bench.json'
    total_images_count, total_items, dataset_stats, question_topic_stats = count_statistics(json_file_path)

    # 打印统计结果
    print(f"Total Images: {total_images_count}")
    print(f"Total Items: {total_items}")

    print("\nDataset Statistics:")
    for dataset, stats in dataset_stats.items():
        print(f"Dataset: {dataset}")
        print(f"  Images: {stats['images']}")
        print(f"  Items: {stats['items']}")

    print("\nQuestion Topic Statistics:")
    for question_topic, stats in question_topic_stats.items():
        print(f"Question Topic: {question_topic}")
        print(f"  Images: {stats['images']}")
        print(f"  Items: {stats['items']}")
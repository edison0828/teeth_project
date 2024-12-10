import json
from collections import Counter


def check_coco_category_distribution(coco_json_path):
    """
    檢查 COCO 格式 JSON 檔案中各類別的數量分佈
    :param coco_json_path: COCO JSON 文件路徑
    """
    # 加載 COCO JSON 文件
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # 提取類別名稱和 ID
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # 統計各類別的標註數量
    category_count = Counter(ann["category_id"]
                             for ann in coco_data["annotations"])

    # 輸出結果
    print(f"檔案: {coco_json_path}")
    print("\n類別數量分佈：")
    for category_id, count in category_count.items():
        category_name = categories.get(category_id, "Unknown")
        print(f"類別 {category_id} ({category_name}): {count} 框")

    # 總數量
    total_annotations = sum(category_count.values())
    print(f"\n總標註數量: {total_annotations}")


# 使用範例
coco_json_path = "../data/coco/disease/annotations/instances_train2017.json"
check_coco_category_distribution(coco_json_path)

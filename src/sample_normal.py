import json
import random

# 文件路徑
normal_annotations_path = "../data/coco/disease/annotations/normal_annotations_train.json"
output_train_path = "../data/coco/disease/annotations/normal_sample_train.json"
output_test_path = "../data/coco/disease/annotations/normal_sample_val.json"

# 目標下採樣數量（例如與有問題牙齒數量接近）
target_sample_size = 5000

# 加載正常牙齒的標註文件
with open(normal_annotations_path, "r") as f:
    normal_data = json.load(f)

# 提取正常牙齒的圖像和標註信息
normal_annotations = normal_data["annotations"]
normal_images = normal_data["images"]

# 隨機下採樣
random.seed(42)  # 固定隨機種子以保證結果可復現
sampled_annotations = random.sample(normal_annotations, target_sample_size)

# 獲取選中的圖像 ID
sampled_image_ids = {annotation["image_id"]
                     for annotation in sampled_annotations}

# 篩選對應的圖像信息
sampled_images = [
    img for img in normal_images if img["id"] in sampled_image_ids]

# 重新分配連續的 annotation id
for new_id, annotation in enumerate(sampled_annotations):
    annotation["id"] = new_id  # 重新分配連續 ID

# 劃分數據（80:20 比例）
train_size = int(0.8 * target_sample_size)  # 訓練集的數量
train_annotations = sampled_annotations[:train_size]
test_annotations = sampled_annotations[train_size:]

# 獲取訓練和測試集對應的圖像 ID
train_image_ids = {annotation["image_id"] for annotation in train_annotations}
test_image_ids = {annotation["image_id"] for annotation in test_annotations}

# 篩選對應的圖像信息
train_images = [img for img in sampled_images if img["id"] in train_image_ids]
test_images = [img for img in sampled_images if img["id"] in test_image_ids]

# 組裝訓練集和測試集的標註數據
train_data = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": normal_data["categories"],  # 保留原始的類別信息
}

test_data = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": normal_data["categories"],  # 保留原始的類別信息
}

# 保存訓練集和測試集的標註文件
with open(output_train_path, "w") as f:
    json.dump(train_data, f)

with open(output_test_path, "w") as f:
    json.dump(test_data, f)

print(f"完成下採樣和劃分，訓練集包含 {len(train_images)} 張圖像和 {len(train_annotations)} 條標註，"
      f"測試集包含 {len(test_images)} 張圖像和 {len(test_annotations)} 條標註。")

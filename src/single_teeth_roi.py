import json
import cv2
import os
import csv

# 替換為訓練集標註文件的路徑
disease_annotations_path = "../data/coco/disease/annotations/instances_train2017.json"
normal_annotations_path = "../data/coco/disease/annotations/normal_sample_train.json"

# # 替換為驗證集標註文件的路徑
# disease_annotations_path = "../data/coco/disease/annotations/instances_val2017.json"
# normal_annotations_path = "../data/coco/disease/annotations/normal_sample_val.json"

# 圖像資料夾路徑train
disease_images_dir = "../data/coco/disease/train2017"  # 有問題牙齒的影像
normal_images_dir = "../data/coco/disease/train2017"  # 正常牙齒的影像訓練集

# # 圖像資料夾路徑val
# disease_images_dir = "../data/coco/disease/val2017"  # 有問題牙齒的影像
# normal_images_dir = "../data/coco/disease/train2017"  # 正常牙齒的影像訓練集


output_dir = "../data/train_single_tooth"  # 存放裁剪後訓練牙齒ROI的目錄
# output_dir = "../data/test_single_tooth"  # 存放裁剪後驗證牙齒ROI的目錄
os.makedirs(output_dir, exist_ok=True)

# 加載標註文件
with open(disease_annotations_path, "r") as f:
    disease_data = json.load(f)

with open(normal_annotations_path, "r") as f:
    normal_data = json.load(f)

# 創建圖像 ID 到文件名的映射
disease_image_dict = {img["id"]: img for img in disease_data["images"]}
normal_image_dict = {img["id"]: img for img in normal_data["images"]}


new_annotations = []  # 用於存放新標註數據
new_images = []  # 用於存放新圖像數據

new_image_id = 0


def process_annotations(data, image_dict, images_dir, category_override=None):
    global new_image_id
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]  # COCO bbox 格式: [x, y, width, height]
        category_id = annotation["category_id"] if category_override is None else category_override

        # 加載對應的影像
        image_info = image_dict[image_id]
        image_path = os.path.join(images_dir, image_info["file_name"])
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        # 獲取 bbox 信息，並轉換為整數座標
        x, y, w, h = map(int, bbox)
        roi = image[y:y+h, x:x+w]  # 裁剪邊界框區域

        # 調整保存路徑，根據圖像和牙齒類別命名
        output_path = os.path.join(output_dir, f"tooth_{new_image_id}.jpg")
        cv2.imwrite(output_path, roi)
        print(f"Saved: {output_path}")

        # 添加新圖像信息
        new_images.append({
            "id": new_image_id,
            "file_name": f"tooth_{new_image_id}.jpg",
            "width": w,
            "height": h
        })

        # 添加新標註信息
        new_annotations.append({
            "id": new_image_id,
            "image_id": new_image_id,
            "category_id": category_id
        })

        new_image_id += 1


# 處理有問題的牙齒數據
process_annotations(disease_data, disease_image_dict, disease_images_dir)

# 處理正常牙齒數據，並分配類別 ID = 4
process_annotations(normal_data, normal_image_dict,
                    normal_images_dir, category_override=4)
# 合併並保存新標註數據到 JSON 文件
new_coco_data = {
    "images": new_images,
    "annotations": new_annotations,
    # 添加正常牙齒類別
    "categories": disease_data["categories"] + [{"id": 4, "name": "normal"}]
}
# 保存到新 JSON 文件
with open("../data/train_annotations.json", "w") as f:
    json.dump(new_coco_data, f)
# with open("../data/test_annotations.json", "w") as f:
#     json.dump(new_coco_data, f)

# CSV 文件路徑
csv_file = "../data/train_annotations.csv"
# csv_file = "../data/test_annotations.csv"

# 寫入裁剪圖像的標註
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "category_id"])  # 標題行
    for annotation in new_annotations:
        file_name = next(img["file_name"]
                         for img in new_images if img["id"] == annotation["image_id"])
        category_id = annotation["category_id"]
        writer.writerow([file_name, category_id])

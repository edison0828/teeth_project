import json
import cv2
import os
import csv

# 替換為訓練集標註文件的路徑
annotations_path = "../data/coco/disease/annotations/instances_train2017.json"
# annotations_path = "../data/coco/disease/annotations/instances_val2017.json"  # 替換為驗證集標註文件的路徑
# 加載 COCO JSON 標註文件
with open(annotations_path, "r") as f:
    data = json.load(f)

# 文件夾路徑
images_dir = "../data/coco/disease_all/train2017"  # 原始全口X光片的圖像目錄

output_dir = "../data/train_single_tooth"  # 存放裁剪後訓練牙齒ROI的目錄
# output_dir = "../data/test_single_tooth"  # 存放裁剪後驗證牙齒ROI的目錄
os.makedirs(output_dir, exist_ok=True)

# 創建圖像 ID 到文件名的映射
image_dict = {img["id"]: img for img in data["images"]}

new_annotations = []  # 用於存放新標註數據
new_images = []  # 用於存放新圖像數據
# new_image_id = 0  # 新的圖像 ID 計數器

# 遍歷標註文件中的每個標註
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    new_image_id = annotation["id"]  # 新的圖像 ID
    bbox = annotation["bbox"]  # COCO bbox 格式: [x, y, width, height]
    category_id = annotation["category_id"]  # 類別 ID

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

    # new_image_id += 1


new_coco_data = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": data["categories"]  # 保留原始的類別標籤
}

# 保存到新 JSON 文件
with open("data/train_annotations.json", "w") as f:
    json.dump(new_coco_data, f)
# with open("data/test_annotations.json", "w") as f:
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

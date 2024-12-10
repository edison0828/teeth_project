import os
import json
from ultralytics import YOLO
from PIL import Image


def get_image_info(image_path, image_id):
    """
    獲取影像的基本信息
    :param image_path: 影像文件路徑
    :param image_id: 影像的唯一 ID
    :return: COCO 格式的影像信息字典
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height
    }


def yolo_to_coco_annotations(detected_boxes, image_id, start_annotation_id):
    """
    將 YOLO 推論結果轉換為 COCO 格式的標註
    :param detected_boxes: YOLO 檢測結果 [{bbox: [x1, y1, x2, y2], confidence: 0.9, category_id: 0}, ...]
    :param image_id: 影像 ID
    :param start_annotation_id: 起始標註 ID
    :return: COCO 格式的標註列表
    """
    annotations = []
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = box["bbox"]
        annotations.append({
            "id": start_annotation_id + i,
            "image_id": image_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h 格式
            "area": (x2 - x1) * (y2 - y1),
            "category_id": box["category_id"],
            "iscrowd": 0
        })
    return annotations


def process_dataset(image_folder, model, categories, output_json_path):
    """
    處理整個影像資料集並保存為 COCO 格式
    :param image_folder: 影像資料夾路徑
    :param model: YOLO 模型
    :param categories: COCO 格式的類別信息
    :param output_json_path: 輸出的 COCO JSON 文件路徑
    """
    image_files = [os.path.join(image_folder, f) for f in os.listdir(
        image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_files, start=1):
        print(f"Processing {image_path} ({image_id}/{len(image_files)})...")

        # 加載影像信息
        image_info = get_image_info(image_path, image_id)
        images.append(image_info)

        # 推論影像
        results = model.predict(source=image_path, conf=0.5)
        detected_boxes = []
        for result in results:
            for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detected_boxes.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "category_id": int(cls)
                })

        # YOLO 推論結果轉為 COCO 格式標註
        image_annotations = yolo_to_coco_annotations(
            detected_boxes, image_id, annotation_id)
        annotations.extend(image_annotations)
        annotation_id += len(image_annotations)

    # 組合 COCO 格式的結構
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # 保存為 JSON 文件
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO 格式 JSON 文件已保存到 {output_json_path}")


# 定義類別信息
categories = [
    {"id": 0, "name": "Normal", "supercategory": "Tooth"},
    {"id": 1, "name": "Impacted", "supercategory": "Tooth"},
    {"id": 2, "name": "Caries", "supercategory": "Tooth"},
    {"id": 3, "name": "Deep Caries", "supercategory": "Tooth"}
]

# 加載 YOLO 模型
model = YOLO("../models/best_yolo11x.pt")

# 處理資料集
image_folder = "../data/coco/disease/train2017"
output_json_path = "../data/coco/disease/annotations/all_teeth_train.json"
# # 處理資料集
# image_folder = "../data/coco/disease/val2017"
# output_json_path = "../data/coco/disease/annotations/all_teeth_val.json"
process_dataset(image_folder, model, categories, output_json_path)

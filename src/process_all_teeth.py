import json


def calculate_iou(box1, box2):
    """
    計算兩個框的 IoU
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU 值
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compare_and_classify(yolo_results_path, original_annotations_path, output_path, iou_threshold=0.01):
    """
    比對 YOLO 結果和原始病徵標註，生成新標註文件
    :param yolo_results_path: YOLO 結果 JSON 文件路徑
    :param original_annotations_path: 原始病徵標註 JSON 文件路徑
    :param output_path: 輸出的新標註文件路徑
    :param iou_threshold: IoU 閾值
    """
    # 加載 YOLO 結果和原始標註
    with open(yolo_results_path, "r") as f:
        yolo_data = json.load(f)
    with open(original_annotations_path, "r") as f:
        original_data = json.load(f)

    # 提取病徵標註框
    problematic_boxes = [
        {
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "image_id": ann["image_id"]
        }
        for ann in original_data["annotations"]
    ]

    # 轉換病徵框為 x1, y1, x2, y2 格式
    for box in problematic_boxes:
        x, y, w, h = box["bbox"]
        box["bbox"] = [x, y, x + w, y + h]

    # 整理 YOLO 檢測結果
    final_annotations = []
    annotation_id = 1
    category_count = {}  # 用於統計數據分佈

    for yolo_image in yolo_data["images"]:
        image_id = yolo_image["id"]
        detected_boxes = [
            ann for ann in yolo_data["annotations"] if ann["image_id"] == image_id
        ]

        for detected_box in detected_boxes:
            x, y, w, h = detected_box["bbox"]
            detected_bbox = [x, y, x + w, y + h]

            is_problematic = False
            for problematic_box in problematic_boxes:
                # 確認影像 ID 相同
                if problematic_box["image_id"] == image_id:
                    if calculate_iou(detected_bbox, problematic_box["bbox"]) > iou_threshold:
                        # 重疊，標記為病徵類別
                        category_id = problematic_box["category_id"]
                        final_annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "bbox": detected_box["bbox"],  # 保持 COCO 格式
                            "area": detected_box["area"],
                            "category_id": problematic_box["category_id"],
                            "iscrowd": 0
                        })
                        # 更新類別統計
                        category_count[category_id] = category_count.get(
                            category_id, 0) + 1
                        annotation_id += 1
                        is_problematic = True
                        break

            if not is_problematic:
                # 無重疊，標記為正常牙齒
                category_id = 4  # 假設 4 為正常牙齒類別
                final_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": detected_box["bbox"],  # 保持 COCO 格式
                    "area": detected_box["area"],
                    "category_id": 4,  # 假設 4 為正常牙齒類別
                    "iscrowd": 0
                })
                # 更新類別統計
                category_count[category_id] = category_count.get(
                    category_id, 0) + 1
                annotation_id += 1

    # 整合新標註文件
    new_coco_data = {
        "images": yolo_data["images"],  # 保留原始影像信息
        "annotations": final_annotations,
        # 添加正常牙齒類別
        "categories": original_data["categories"] + [{"id": 4, "name": "Normal", "supercategory": "Tooth"}]
    }

    # 保存新標註文件
    with open(output_path, "w") as f:
        json.dump(new_coco_data, f, indent=4)

    print(f"新標註文件已保存至 {output_path}")
    # 輸出類別分佈統計
    print("\n數據分佈統計：")
    for category_id, count in category_count.items():
        category_name = next(
            (cat["name"] for cat in new_coco_data["categories"] if cat["id"] == category_id), "Unknown")
        print(f"類別 {category_id} ({category_name}): {count} 框")


# 使用範例train
yolo_results_path = "../data/coco/disease/annotations/all_teeth_train.json"
original_annotations_path = "../data/coco/disease/annotations/instances_train2017.json"
output_path = "../data/coco/disease/annotations/all_annotations_train.json"

# # 使用範例val
# yolo_results_path = "../data/coco/disease/annotations/all_teeth_val.json"
# original_annotations_path = "../data/coco/disease/annotations/instances_val2017.json"
# output_path = "../data/coco/disease/annotations/all_annotations_val.json"
compare_and_classify(yolo_results_path, original_annotations_path, output_path)

import json
import cv2
import os
import numpy as np
import csv
import shutil
from collections import defaultdict

CATEGORY_ID_MAPPING = {
    33: 1,  # Caries
    34: 2,  # Impacted
    35: 3,  # Periapical Lesion
    36: 4,  # Retained dental root
}
NORMAL_CLASS_ID = 5

new_categories = [
    {"id": 1, "name": "Caries", "supercategory": "teeth-Ji9p"},
    {"id": 2, "name": "Impacted", "supercategory": "teeth-Ji9p"},
    {"id": 3, "name": "Periapical Lesion", "supercategory": "teeth-Ji9p"},
    {"id": 4, "name": "Retained dental root", "supercategory": "teeth-Ji9p"},
    {"id": 5, "name": "normal", "supercategory": "teeth-Ji9p"}
]


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area else 0


def create_binary_datasets(output_root, output_dir, new_annotations, disease_names, iou_threshold=0.7):
    binary_base_dir = os.path.join(output_root, "binary_datasets")

    annotations_by_original_image = defaultdict(list)
    for ann in new_annotations:
        annotations_by_original_image[ann["original_file_name"]].append(ann)

    for disease_id, disease_name in disease_names.items():
        binary_dir = os.path.join(binary_base_dir, f"{disease_name}_vs_Normal")
        pos_dir = os.path.join(binary_dir, "Positive")
        neg_dir = os.path.join(binary_dir, "Negative")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        binary_csv = os.path.join(
            binary_dir, f"{disease_name}_annotations.csv")
        with open(binary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "label"])

            positive_samples = [
                ann for ann in new_annotations if ann["category_id"] == disease_id]
            for ann in positive_samples:
                shutil.copy(os.path.join(
                    output_dir, ann["file_name"]), pos_dir)
                writer.writerow([ann["file_name"], 1])

            negative_candidates = [
                ann for ann in new_annotations if ann["category_id"] != disease_id]

            overlap_excluded_count = 0

            for neg_ann in negative_candidates:
                overlap = False
                original_image_rois = annotations_by_original_image[neg_ann["original_file_name"]]
                current_disease_rois_same_image = [
                    ann for ann in original_image_rois if ann["category_id"] == disease_id
                ]

                for pos_ann in current_disease_rois_same_image:
                    iou = calculate_iou(pos_ann["bbox"], neg_ann["bbox"])
                    if iou >= iou_threshold:
                        overlap = True
                        overlap_excluded_count += 1
                        break

                if not overlap:
                    shutil.copy(os.path.join(
                        output_dir, neg_ann["file_name"]), neg_dir)
                    writer.writerow([neg_ann["file_name"], 0])

        print(f"Binary dataset created for {disease_name}")
        print(f"因重疊而排除的負樣本數量：{overlap_excluded_count}\n")


def filter_caries_if_retained_root_exists(annotations):
    """
    如果同一張 X 光片的牙齒有 Caries (1) 和 Retained dental root (4)，
    且它們的 bbox 重疊，則刪除 Caries 的標註。
    """
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann["original_file_name"]].append(ann)

    filtered_annotations = []
    total_caries_removed = 0  # 追蹤刪除的 Caries 數量

    for original_file_name, anns in annotations_by_image.items():
        retained_roots = [ann for ann in anns if ann["category_id"] == 4]
        caries = [ann for ann in anns if ann["category_id"] == 1]

        print(
            f"🔍 檢查 X 光片: {original_file_name}, Caries: {len(caries)}, Retained Roots: {len(retained_roots)}")

        caries_to_remove = set()

        for caries_ann in caries:
            for root_ann in retained_roots:
                iou = calculate_iou(caries_ann["bbox"], root_ann["bbox"])
                print(
                    f"👉 比較 Caries {caries_ann['id']} 和 Retained Root {root_ann['id']}，IOU: {iou:.3f}")

                if iou >= 0.5:
                    caries_to_remove.add(caries_ann["id"])

        total_caries_removed += len(caries_to_remove)

        for ann in anns:
            if ann["id"] not in caries_to_remove:
                filtered_annotations.append(ann)

    print(
        f"🔥 最終刪除 {total_caries_removed} 個 Caries 標註，因為同一顆牙齒上有 Retained dental root。")
    return filtered_annotations


def process_annotations(split, annotations_path, images_dir):
    output_root = f"../data/dentex2023 disease.v6i.coco/{split}/roi_bbox"
    output_dir = os.path.join(output_root, "rois")
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    image_dict = {img["id"]: img for img in data["images"]}

    # 先將原始 annotations 轉換 category_id
    original_annotations = []
    for ann in data["annotations"]:
        orig_category_id = ann["category_id"]
        category_id = CATEGORY_ID_MAPPING.get(
            orig_category_id, NORMAL_CLASS_ID)

        original_annotations.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": category_id,
            "bbox": ann["bbox"],
            "original_file_name": image_dict[ann["image_id"]]["file_name"]
        })

    # 🔥 **先過濾 Caries，確保不與 Retained dental root 重疊**
    filtered_annotations = filter_caries_if_retained_root_exists(
        original_annotations)

    new_images, new_annotations = [], []

    for annotation in filtered_annotations:
        image_info = image_dict[annotation["image_id"]]
        image_path = os.path.join(images_dir, image_info["file_name"])
        image = cv2.imread(image_path)

        if image is None:
            continue

        x_min, y_min, width, height = map(int, annotation["bbox"])
        roi = image[y_min:y_min+height, x_min:x_min+width]
        if roi.size == 0:
            continue

        roi_filename = f"tooth_{len(new_images)}.jpg"
        cv2.imwrite(os.path.join(output_dir, roi_filename), roi)

        new_images.append({
            "id": len(new_images),
            "file_name": roi_filename,
            "width": width,
            "height": height
        })

        new_annotations.append({
            "id": len(new_annotations),
            "image_id": len(new_images)-1,
            "file_name": roi_filename,
            "category_id": annotation["category_id"],
            "bbox": [x_min, y_min, width, height],
            "original_file_name": image_info["file_name"]
        })

    json_output_path = os.path.join(output_root, "roi_annotations.json")
    with open(json_output_path, "w") as f:
        json.dump({"images": new_images, "annotations": new_annotations,
                  "categories": new_categories}, f)

    disease_names = {1: "Caries", 2: "Impacted",
                     3: "Periapical_Lesion", 4: "Retained_dental_root"}
    create_binary_datasets(output_root, output_dir,
                           new_annotations, disease_names)


for split in ["train", "valid", "test"]:
    annotations_path = f"../data/dentex2023 disease.v6i.coco/{split}/_annotations.coco.json"
    images_dir = f"../data/dentex2023 disease.v6i.coco/{split}/original"
    process_annotations(split, annotations_path, images_dir)

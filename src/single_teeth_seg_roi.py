import json
import cv2
import os
import numpy as np
import csv
import shutil

# 定義病徵類別ID映射
CATEGORY_ID_MAPPING = {
    33: 1,  # Caries
    34: 2,  # Impacted
    35: 3,  # Periapical Lesion
    36: 4,  # Retained dental root
}
NORMAL_CLASS_ID = 5  # 新的normal類別ID

new_categories = [
    {"id": 1, "name": "Caries", "supercategory": "teeth-Ji9p"},
    {"id": 2, "name": "Impacted", "supercategory": "teeth-Ji9p"},
    {"id": 3, "name": "Periapical Lesion", "supercategory": "teeth-Ji9p"},
    {"id": 4, "name": "Retained dental root", "supercategory": "teeth-Ji9p"},
    {"id": 5, "name": "normal", "supercategory": "teeth-Ji9p"}
]


def process_annotations(split, annotations_path, images_dir):
    output_dir = f"../data/dentex2023 disease.v6i.coco/{split}/rois"
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    image_dict = {img["id"]: img for img in data["images"]}

    new_images, new_annotations = [], []

    for annotation in data["annotations"]:
        image_info = image_dict[annotation["image_id"]]
        image_path = os.path.join(images_dir, image_info["file_name"])
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for seg in annotation["segmentation"]:
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 255)

        ys, xs = np.where(mask == 255)
        if len(xs) == 0 or len(ys) == 0:
            print(f"Empty segmentation for annotation id {annotation['id']}")
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        roi = image[y_min:y_max+1, x_min:x_max+1]
        roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

        roi_segmented = cv2.bitwise_and(roi, roi, mask=roi_mask)

        roi_filename = f"tooth_{len(new_images)}.jpg"
        cv2.imwrite(os.path.join(output_dir, roi_filename), roi_segmented)
        print(f"Saved: {roi_filename}")

        orig_category_id = annotation["category_id"]
        category_id = CATEGORY_ID_MAPPING.get(
            orig_category_id, NORMAL_CLASS_ID)

        new_images.append({
            "id": len(new_images),
            "file_name": roi_filename,
            "width": roi_segmented.shape[1],
            "height": roi_segmented.shape[0]
        })

        new_annotations.append({
            "id": len(new_annotations),
            "image_id": len(new_images)-1,
            "category_id": category_id,
            "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
            "original_file_name": image_info["file_name"]
        })

    new_coco_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(f"../data/dentex2023 disease.v6i.coco/{split}/roi_annotations.json", "w") as f:
        json.dump(new_coco_data, f)

    csv_file = f"../data/dentex2023 disease.v6i.coco/{split}/roi_annotations.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "category_id",
                        "original_file_name", "bbox"])

        for ann in new_annotations:
            writer.writerow([
                new_images[ann["image_id"]]["file_name"],
                ann["category_id"],
                ann["original_file_name"],
                ann["bbox"]
            ])

    disease_names = {1: "Caries", 2: "Impacted",
                     3: "Periapical_Lesion", 4: "Retained_dental_root"}
    binary_base_dir = f"../data/dentex2023 disease.v6i.coco/{split}/binary_datasets"

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

            for ann, img in zip(new_annotations, new_images):
                src_path = os.path.join(output_dir, img["file_name"])

                if ann["category_id"] == disease_id:
                    shutil.copy(src_path, pos_dir)
                    writer.writerow([img["file_name"], 1])
                else:
                    shutil.copy(src_path, neg_dir)
                    writer.writerow([img["file_name"], 0])

        print(f"Binary dataset created for {disease_name}")


for split in ["train", "valid", "test"]:
    annotations_path = f"../data/dentex2023 disease.v6i.coco/{split}/_annotations.coco.json"
    images_dir = f"../data/dentex2023 disease.v6i.coco/{split}"

    process_annotations(split, annotations_path, images_dir)

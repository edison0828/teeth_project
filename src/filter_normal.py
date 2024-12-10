import json


def filter_coco_annotations(input_file, output_file, target_category_id):
    # 讀取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 保留 category 為 target_category_id 的標註資料
    filtered_annotations = [
        annotation for annotation in coco_data['annotations'] if annotation['category_id'] == target_category_id
    ]

    # 找到與這些標註相關的 image_id
    valid_image_ids = set(annotation['image_id']
                          for annotation in filtered_annotations)

    # 篩選出相關的圖像資料
    filtered_images = [
        image for image in coco_data['images'] if image['id'] in valid_image_ids
    ]

    # 重新分配 annotation_id，保留原始的 image_id
    annotation_id_map = {annotation['id']: idx for idx,
                         annotation in enumerate(filtered_annotations)}

    for annotation in filtered_annotations:
        annotation['id'] = annotation_id_map[annotation['id']]
        # image_id 不變，直接保留

    # 更新 coco_data 並保留 categories 資訊
    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': [category for category in coco_data['categories'] if category['id'] == target_category_id]
    }

    # 將處理後的資料寫入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_coco_data, f, ensure_ascii=False, indent=4)

    print(f"Filtered annotations saved to {output_file}")


# 使用範例train
input_file = '../data/coco/disease/annotations/all_annotations_train.json'
output_file = '../data/coco/disease/annotations/normal_annotations_train.json'

# # 使用範例val
# input_file = '../data/coco/disease/annotations/all_annotations_val.json'
# output_file = '../data/coco/disease/annotations/normal_annotations_val.json'

target_category_id = 4

filter_coco_annotations(input_file, output_file, target_category_id)

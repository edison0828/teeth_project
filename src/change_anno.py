import json


def process_coco_annotations(input_file, output_file, unified_category_id=0):
    """
    簡化 COCO 格式 JSON 文件，只保留 bbox 和 category_id，刪除 segmentation 等其他字段。
    :param input_file: 原始 COCO JSON 文件
    :param output_file: 簡化後的 JSON 文件
    :param unified_category_id: 統一的 category_id，默認為 0
    """
    # 讀取原始 COCO 文件
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # 更新 annotations
    for annotation in coco_data['annotations']:
        # 保留 bbox 和 category_id，刪除其他字段
        annotation['category_id'] = unified_category_id  # 統一類別
        annotation.pop('segmentation', None)            # 刪除 segmentation
        # annotation.pop('iscrowd', None)                 # 刪除 iscrowd

    new_categories = [
        {"id": 0, "name": "tooth", "supercategory": "tooth"}  # 單一類別示例
    ]
    # 替換 categories 部分
    coco_data['categories'] = new_categories

    # 保存處理後的文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"處理完成，結果已保存至 {output_file}")


# 使用範例
new_categories = [
    {"id": 0, "name": "tooth", "supercategory": "tooth"}  # 單一類別示例
]
input_file_train = '../data/coco/enumeration32/annotations/instances_train2017.json'
output_file_train = '../data/coco/enumeration32/annotations/only_boxes_train2017.json'
process_coco_annotations(
    input_file_train, output_file_train, unified_category_id=0)
input_file_val = '../data/coco/enumeration32/annotations/instances_val2017.json'
output_file_val = '../data/coco/enumeration32/annotations/only_boxes_val2017.json'
process_coco_annotations(
    input_file_val, output_file_val, unified_category_id=0)

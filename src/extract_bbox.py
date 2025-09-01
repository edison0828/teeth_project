import json
import cv2
import os
import numpy as np

# 路徑設定
IMAGE_PATH = "./train_273_png.jpg"
ANNOTATION_JSON = "./train_273.json"
OUTPUT_DIR = "output_teeth_bbox"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(IMAGE_PATH)
with open(ANNOTATION_JSON, 'r') as f:
    annotations = json.load(f)

for box in annotations['boxes']:
    label = box['label']
    polygon = np.array(box['points'], dtype=np.int32)

    # 用polygon的最小外接矩形來裁剪
    x, y, w, h = cv2.boundingRect(polygon)

    cropped_tooth = image[y:y+h, x:x+w]

    output_filename = f"{label}_{box['id']}_bbox_fixed.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), cropped_tooth)

    print(f"已裁切並儲存修正後bbox影像：{output_filename}")

print("所有修正後的bbox裁剪完成！")

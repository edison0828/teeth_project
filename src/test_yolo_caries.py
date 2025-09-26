from ultralytics import YOLO


def main():
    model = YOLO("./all_caries.pt")
    metrics = model.val(
        data="./caries_yolo11_test/data.yaml",
        split="test",
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,
        device=0,
        save_json=True,
        plots=True
    )

    # 讀取整體指標（box = 偵測）
    print("mAP50-95:", metrics.box.map)      # mAP@0.5:0.95
    print("mAP50   :", metrics.box.map50)    # mAP@0.5
    print("Precision:", metrics.box.mp)       # mean precision
    print("Recall   :", metrics.box.mr)       # mean recall

    # 逐類別 mAP（list，順序對應 model.names）
    per_class_maps = metrics.box.maps
    print("per-class mAP:", list(zip(model.names.values(), per_class_maps)))


if __name__ == "__main__":   # ★ 必加在 Windows
    main()

#!/bin/bash

# # 實驗 1：基線 (啟用資料增強、使用 swav 預訓練、加權重抽樣、Focal Loss、解凍4和fc、使用學習率調度器)
# echo "開始實驗 1: 基線設定"
# python resnet50_ablation.py \
#   --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
#   --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
#   --model_save_path "../models/resnet50_swav_caries_seg_experiment1" \
#   --tensorboard_log "runs/experiment1" \
#   --minority_class 1 \
#   --epochs 300 \
#   --patience 20 \
#   --use_augmentation \
#   --use_pretrained \
#   --use_weighted_sampler \
#   --use_focal_loss \
#   --use_scheduler

# if [ $? -ne 0 ]; then
#   echo "實驗 1 執行錯誤，停止後續實驗。"
#   exit 1
# fi

# # 實驗 2：僅解凍 fc 層 (其他參數同實驗 1)
# echo "開始實驗 2: 僅解凍 fc 層"
# python resnet50_ablation.py \
#   --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
#   --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
#   --model_save_path "../models/resnet50_swav_caries_seg_experiment2" \
#   --tensorboard_log "runs/experiment2" \
#   --minority_class 1 \
#   --epochs 300 \
#   --patience 20 \
#   --use_augmentation \
#   --use_pretrained \
#   --use_weighted_sampler \
#   --use_focal_loss \
#   --finetune_fc_only \
#   --use_scheduler

# if [ $? -ne 0 ]; then
#   echo "實驗 2 執行錯誤，停止後續實驗。"
#   exit 1
# fi

# # 實驗 3：不使用資料增強 (其他參數同基線)
# echo "開始實驗 3: 不使用資料增強"
# python resnet50_ablation.py \
#   --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
#   --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
#   --model_save_path "../models/resnet50_swav_caries_seg_experiment3" \
#   --tensorboard_log "runs/experiment3" \
#   --minority_class 1 \
#   --epochs 300 \
#   --patience 20 \
#   --no_use_augmentation \
#   --use_pretrained \
#   --use_weighted_sampler \
#   --use_focal_loss \
#   --use_scheduler

# if [ $? -ne 0 ]; then
#   echo "實驗 3 執行錯誤，停止後續實驗。"
#   exit 1
# fi

# # 實驗 4：不使用預訓練權重 (使用imagenet，其餘與基線相同)
# echo "開始實驗 4: 不使用預訓練權重"
# python resnet50_ablation.py \
#   --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
#   --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
#   --model_save_path "../models/resnet50_swav_caries_seg_experiment4" \
#   --tensorboard_log "runs/experiment4" \
#   --minority_class 1 \
#   --epochs 300 \
#   --patience 20 \
#   --use_augmentation \
#   --no_use_pretrained \
#   --use_weighted_sampler \
#   --use_focal_loss \
#   --use_scheduler

# if [ $? -ne 0 ]; then
#   echo "實驗 4 執行錯誤，停止後續實驗。"
#   exit 1
# fi

# # 實驗 5：不使用加權重抽樣 (其他參數同基線)
# echo "開始實驗 5: 不使用加權重抽樣"
# python resnet50_ablation.py \
#   --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
#   --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
#   --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
#   --model_save_path "../models/resnet50_swav_caries_seg_experiment5" \
#   --tensorboard_log "runs/experiment5" \
#   --minority_class 1 \
#   --epochs 300 \
#   --patience 20 \
#   --use_augmentation \
#   --use_pretrained \
#   --no_use_weighted_sampler \
#   --use_focal_loss \
#   --use_scheduler \
#   --no_use_weighted_sampler

# if [ $? -ne 0 ]; then
#   echo "實驗 5 執行錯誤，停止後續實驗。"
#   exit 1
# fi

# 實驗 6：使用 CrossEntropy Loss（不使用 Focal Loss，其餘同基線）
echo "開始實驗 6: 使用 CrossEntropy Loss（取消 Focal Loss)"
python resnet50_ablation.py \
  --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
  --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
  --model_save_path "../models/resnet50_swav_caries_seg_experiment6" \
  --tensorboard_log "runs/experiment6" \
  --minority_class 1 \
  --epochs 300 \
  --patience 20 \
  --use_augmentation \
  --use_pretrained \
  --use_weighted_sampler \
  --use_scheduler \
  --no_use_focal_loss

if [ $? -ne 0 ]; then
  echo "實驗 6 執行錯誤，停止後續實驗。"
  exit 1
fi

# 實驗 7：不使用學習率調度器 (其他參數同基線)
echo "開始實驗 7: 不使用學習率調度器"
python resnet50_ablation.py \
  --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
  --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
  --model_save_path "../models/resnet50_swav_caries_seg_experiment7" \
  --tensorboard_log "runs/experiment7" \
  --minority_class 1 \
  --epochs 300 \
  --patience 20 \
  --use_augmentation \
  --use_pretrained \
  --use_weighted_sampler \
  --no_use_scheduler

if [ $? -ne 0 ]; then
  echo "實驗 7 執行錯誤，停止後續實驗。"
  exit 1
fi

# 實驗 8：綜合消融 - 不使用資料增強、預訓練、學習率調度器
echo "開始實驗 8: 綜合消融（取消資料增強、預訓練及學習率調度器，僅解凍 fc 層)"
python resnet50_ablation.py \
  --csv_train "../data/dentex2023 disease.v6i.coco/train/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --csv_val "../data/dentex2023 disease.v6i.coco/valid/binary_datasets/Caries_vs_Normal/Caries_annotations.csv" \
  --img_dir_train "../data/dentex2023 disease.v6i.coco/train/rois" \
  --img_dir_valid "../data/dentex2023 disease.v6i.coco/valid/rois" \
  --model_save_path "../models/resnet50_swav_caries_seg_experiment8" \
  --tensorboard_log "runs/experiment8" \
  --minority_class 1 \
  --epochs 300 \
  --patience 20 \
  --no_use_scheduler \
  --no_use_augmentation \
  --no_use_weighted_sampler \
  --no_use_focal_loss \


if [ $? -ne 0 ]; then
  echo "實驗 8 執行錯誤，停止執行。"
  exit 1
fi

# 實驗 9：efficientnet_v2s
echo "開始實驗 9: efficientnet_v2s"
python effiecientnet_v2s_train.py

if [ $? -ne 0 ]; then
  echo "實驗 9 執行錯誤，停止後續實驗。"
  exit 1
fi

# 實驗 10：rad-dino_v2
echo "開始實驗 10: rad-dino_v2"
python rad-dino_v2_caries_train.py

if [ $? -ne 0 ]; then
  echo "實驗 10 執行錯誤，停止後續實驗。"
  exit 1
fi

echo "所有實驗已成功執行完畢！"



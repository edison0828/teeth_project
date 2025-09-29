# Demo 樣本資產

請在此資料夾放置 Demo 用的靜態影像，檔名會自動轉換成樣本 ID（取決於檔案名稱，不含副檔名）。

例如：

- `anterior.png`
- `posterior_case.jpg`
- `root-caries.jpeg`

服務啟動時會自動掃描支援的影像格式（PNG、JPG、JPEG），並以 `/demo-assets/samples/<檔名>` 的相對路徑提供給前端選取。若要預先提供疊圖或 Grad-CAM，可額外放置：

- `{sample_id}_overlay.png`（或 `.jpg`）
- `{sample_id}_cam_<FDI>.png`

推論結果仍會輸出到 `demo_backend/outputs/` 供下載，若存在上述檔案，API 也會一併回傳對應的靜態路徑。

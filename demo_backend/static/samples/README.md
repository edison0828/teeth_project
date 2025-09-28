# Demo 樣本資產

請在此資料夾放置 Demo 用的靜態影像與對應的 Grad-CAM / 疊圖檔案，檔名需與 `demo_backend/samples/manifest.json` 相符，例如：

- `anterior.png`
- `anterior_overlay.png`
- `anterior_cam_11.png`
- `anterior_cam_21.png`
- `posterior.png`
- `posterior_overlay.png`
- `posterior_cam_16.png`
- `posterior_cam_26.png`

啟動 FastAPI 服務後，這些檔案會透過 `/demo-assets/samples/...` 路徑提供給前端頁面與 API 使用。

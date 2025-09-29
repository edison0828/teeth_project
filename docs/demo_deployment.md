# Cross-Attention Grad-CAM Demo 設計與雲端部署指南

## 1. 目標與範疇
- 在不影響完整臨床系統的前提下，提供一個公開可訪問的跨注意力 Grad-CAM 體驗。
- 直接重用 `src/infer_cross_cam.py` 的研究推論流程，封裝為可容器化的 API 服務並部署到雲端。
- 建置單頁前端，讓使用者能瀏覽預載樣本或上傳影像，即時檢視模型預測與 Grad-CAM 熱力圖。

## 2. 整體架構概觀

```
┌────────────────────────────────────────────────────────────────┐
│                        Demo Frontend (Next.js)                 │
│                                                                │
│  /demo page                                                    │
│    ├─ Sample gallery (reads `/demo/samples`)                    │
│    ├─ Upload widget (POST `/demo/infer`)                        │
│    └─ Result viewer (overlay, findings table, Grad-CAM tiles)  │
└────────────────────────────────────────────────────────────────┘
                 │                              ▲
                 ▼                              │
┌────────────────────────────────────────────────────────────────┐
│                    Demo Backend (FastAPI)                      │
│                                                                │
│  Endpoints                                                     │
│    GET  /demo/samples   → list auto-discovered static samples  │
│    POST /demo/infer     → run inference on uploads or presets  │
│                                                                │
│  Pipeline                                                      │
│    ├─ Lazy-load YOLO + CrossAttention classifier               │
│    ├─ Call `infer_one_image` from research script              │
│    ├─ Persist overlay + CSV in `demo_backend/outputs/`         │
│    └─ Return structured JSON with relative URLs                │
└────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│                Model Assets & Static Samples                   │
│                                                                │
│  - Weights (mounted into container via env-configurable paths) │
│  - `demo_backend/static/samples/` for bundled demo images      │
│  - Optional heatmaps saved alongside the source images         │
└────────────────────────────────────────────────────────────────┘
```

> 提示：ASCII 圖示保留英文元素以利對照程式碼路徑；其餘敘述皆以中文撰寫。

## 3. 後端設計細節

### 3.1 程式結構
- `demo_backend/main.py`：FastAPI 應用程式，統一使用 `/demo` 前綴並掛載靜態資源（`/demo-assets`、`/demo-outputs`）。
- `demo_backend/pipeline.py`：封裝 YOLO 偵測與 CrossAttention 分類器，直接呼叫既有的 `infer_one_image` 函式，確保輸出一致。
- `demo_backend/samples.py`：掃描靜態資料夾取得樣本、overlay、Grad-CAM 對應關係，供 `/demo/samples` 使用。
- `demo_backend/schemas.py`：定義 Pydantic 輸出模型（齒別資訊、Grad-CAM 路徑、警示訊息等）。
- `demo_backend/config.py`：集中管理環境變數設定，例如權重路徑、輸出資料夾與啟動預載選項。

### 3.2 執行期主要參數
| 環境變數 | 說明 | 預設值 |
|----------|------|--------|
| `DEMO_YOLO_WEIGHTS` | YOLO 齒位偵測器權重路徑 | `models/fdi_seg.pt` |
| `DEMO_CLASSIFIER_WEIGHTS` | CrossAttention 分類器權重 | `models/cross_attn_fdi_camAlignA.pth` |
| `DEMO_LAYERED_THRESHOLDS` | 逐層信心門檻 JSON（選用） | 未設定 |
| `DEMO_OUTPUT_DIR` | 推論輸出（疊圖、CSV）存放目錄 | `demo_backend/outputs` |
| `DEMO_STATIC_DIR` | 預載樣本與資產根目錄 | `demo_backend/static` |
| `DEMO_SAMPLES_SUBDIR` | 預設樣本所在子資料夾 | `samples` |
| `DEMO_DEVICE` | 推論裝置（`cuda` 或 `cpu`） | `cuda` |
| `DEMO_AUTOLOAD` | 啟動時是否預先載入權重 | `false` |

### 3.3 推論流程
1. 使用者呼叫 `/demo/infer`，可傳入 `sample_id`（重播預載結果）或直接上傳影像檔。
2. 管線確保 YOLO 與分類模型已載入，並沿用 `src/infer_cross_cam.py` 的前處理與逐齒推論邏輯。
3. 推論產生的疊圖、Grad-CAM、CSV 會存放於 `<output_dir>/<request_id>/...`，並以相對路徑（`/demo-outputs/...`）回傳。
4. 內建樣本與使用者上傳使用相同推論流程；若靜態資料夾內提供對應熱力圖檔案，回傳時會自動帶入路徑。

### 3.4 樣本資產準備
- 將 PNG / JPG 檔放進 `demo_backend/static/samples/`，檔名（不含副檔名）會直接當作 `sample_id` 使用。
- 若同資料夾中存在 `{sample_id}_overlay.png` 或 `{sample_id}_cam_<FDI>.png`，API 會自動帶入對應路徑，方便示範預先產生的疊圖與 Grad-CAM。
- FastAPI 啟動後會將該資料夾掛載為 `/demo-assets/samples/...`，前端選取樣本時若找不到影像檔，會回傳 404 提示。

## 4. 前端設計重點
- 位置：`frontend/app/demo/page.tsx`（Client Component）。
- 頁面以全螢幕獨立樣式呈現，不再共用主系統的 Sidebar/Nav，未登入使用者也能直接存取。
- 主要功能：
  - 自動列出靜態樣本，點擊即送出推論並在右側顯示疊圖與逐齒結果。
  - 支援單檔上傳，完成後可於頁面上直接檢視 overlay、Grad-CAM 與 CSV 下載提示。
  - 互動式結果表格可切換觀察焦點，若 API 提供 `cam_path` 會同步切換熱力圖視圖。
- API 工具：
  - `fetchDemoSamples` 與 `submitDemoInference` 移除 `rerun` 參數，純粹以 `sampleId` 或檔案輸入觸發推論。
  - 型別定義維持於 `frontend/lib/types.ts`，並擴充 `DemoSampleSummary` 的 optional 欄位以對應自動偵測的 overlay / cam 路徑。

## 5. 雲端部署藍圖

### 5.1 容器建置
1. **後端映像**
   ```Dockerfile
   FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
   WORKDIR /app
   COPY demo_backend/ requirements.txt src/ models/ ./
   RUN pip install --no-cache-dir -r demo_backend/requirements.txt
   CMD ["uvicorn", "demo_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   - 權重檔可直接打包進映像，或在部署時透過 Volume 掛載。
   - 對外開放 `8000` 埠；若目標環境無 GPU，將 `DEMO_DEVICE` 設為 `cpu`。

2. **前端映像**
   ```Dockerfile
   FROM node:18-alpine AS builder
   WORKDIR /app
   COPY frontend/ ./
   RUN npm install && npm run build

   FROM nginx:alpine
   COPY --from=builder /app/out /usr/share/nginx/html
   ```
   - 以 `NEXT_PUBLIC_API_BASE_URL` 指定後端服務網址，例如 `https://demo-api.example.com`。
   - 亦可改採 Vercel、Netlify 等託管方案，以環境變數設定 API 網域。

### 5.2 雲端拓撲選項
- **選項 A：單機 VM（快速驗證）**
  - 建議使用具備 GPU 的 VM（AWS g4dn、GCP a2、Azure NC 系列）以支援即時推論。
  - 安裝 Docker 後以 `--gpus all` 啟動後端容器；前端可由 nginx 或 Next.js SSR 伺服器提供。
  - 使用 nginx 作為反向代理，統一管理 HTTPS（可透過 Let’s Encrypt 自動簽發憑證）。

- **選項 B：受管服務**
  - **後端**：部署至 AWS ECS/Fargate（GPU 型別）或 GKE/AKS，權重可存於 S3 / EFS，啟動時掛載。
  - **前端**：將靜態輸出放置於 S3 + CloudFront、Netlify 或 Vercel，僅需設定 API 網域環境變數。
  - **儲存**：若需保存使用者上傳與推論輸出，可寫入 S3 / Blob Storage 並回傳簽名網址；目前預設寫入本地磁碟以簡化流程。

### 5.3 CI/CD 建議
- 為 `feature/cross-attn-demo`（或其他 Demo 分支）新增建置流程，自動產生並推送後端映像到容器登錄。
- 在 CI 中啟動服務並設定 `DEMO_AUTOLOAD=true`，以確保權重可順利載入並完成一次煙霧測試。
- 前端沿用既有 build pipeline，額外加入 `/demo` 頁面的靜態建置與 API fallback 檢查。

## 6. 營運與安全考量
- **冷啟動**：建議開啟 `DEMO_AUTOLOAD=true`，並於啟動時呼叫 `pipeline.warmup()` 使用內建樣本預熱。
- **流量控制**：可在 FastAPI dependency 中加入簡易 Token Bucket，或於 API Gateway / CloudFront 設定速率限制。
- **監控**：開啟 uvicorn access log、整合 APM/metrics（如 Prometheus）以觀測推論延遲與 GPU 使用率。
- **安全**：公開 Demo 可加入 API Key 驗證或前端簡易驗證碼機制，避免大量惡意請求。
- **資源清理**：排程背景工作定期清除 `demo_backend/outputs/` 舊檔，防止磁碟空間耗盡。

## 7. 後續優化方向
- 將 Grad-CAM 產出改為即時計算，支援所有新上傳影像（目前若需示範可於靜態資料夾預先放置對應檔案）。
- 引入 WebSocket 或長輪詢回報推論進度，改善大型檔案的使用者體驗。
- 擴充靜態樣本的中繼資料格式（例如額外 JSON），加入臨床評語或嚴重度標籤，提升 Demo 的敘事性。

# 口腔 X 光影像分析系統設計提案

## 1. 系統概要

- **核心目標**：建立一個模組化且可擴展的口腔 X 光影像分析平台，支援多模型協作，涵蓋影像上傳、AI 分析、專業人員審閱與報告產出。
- **關鍵特色**：
  - 插件式 AI 模型整合，支援未來不同架構的檢測器。
  - 統一資料結構，確保前後端在顯示多類病徵時的彈性。
  - 前後端分離但緊密協作的系統，兼顧操作人員與 AI 工程團隊需求。

## 2. 需求整理

| 類別 | 需求摘要 |
| --- | --- |
| 功能 | 患者管理、影像上傳與預處理、AI 分析檢視、報告產出、模型管理、權限管理 |
| 非功能 | 高可用、高擴展性、可審計、具備版本管理與回溯能力 |
| 使用者角色 | 臨床醫師、放射師、模型管理員、系統管理員 |
| 合規 | 資料加密、權限控管、系統活動紀錄 (Audit Trail)、可支援 HIPAA/GDPR 的部署策略 |

## 3. 系統架構

```
┌────────────────────────────────────┐
│               前端 (Next.js)          │
│  • Dashboard / 患者 / 上傳 / 分析結果  │
│  • 統一呼叫後端 REST / WebSocket API  │
└────────────────────────────────────┘
                │ HTTPS / WSS
┌────────────────────────────────────┐
│     API Gateway & Backend (Next.js API Routes / FastAPI) │
│  • AuthN/AuthZ、RBAC                                 │
│  • 患者資料 CRUD、影像與報告服務                      │
│  • AI 作業排程、模型管理 API                         │
└────────────────────────────────────┘
                │ Internal gRPC / REST / Queue
┌────────────────────────────────────┐
│        AI Pipeline Orchestrator (Python)           │
│  • 任務排程、模型插件載入、佇列監控                     │
│  • 模型推論容器 (參考 `infer_cross_cam.py`)             │
│  • 訓練工作流 (參考 `train_cross_cam.py`)              │
└────────────────────────────────────┘
                │
┌────────────────────────────────────┐
│   儲存層 │
│  • MySQL：患者 / 影像 / 病徵 / 設定                     │
│  • 物件儲存 (S3/MinIO)：影像原檔與推論產出              │
│  • Redis / RabbitMQ：任務佇列、快取                   │
└────────────────────────────────────┘
```

- **前端**：採用 Next.js App Router，整合型態化 API 客戶端，支援 Server Components 與 Edge Rendering，提高效能。
- **後端**：
  - 患者資料、使用者管理等 CRUD 可以由 Next.js API Routes 或獨立 FastAPI 服務提供。
  - 模型推論、訓練作業建議由 FastAPI (Python) 實作，以便與現有模型程式共享環境。
- **AI Pipeline**：
  - 以 Celery / FastAPI Background Tasks 管理推論任務，並透過 plugin registry 掛載不同模型。
  - 模型容器採 Docker 化，統一輸入/輸出介面。
- **資料儲存**：
  - 影像與報告原檔以物件儲存管理，資料庫只保存索引與中繼資料。
  - Redis 提供 session、快取以及工作佇列支援。

## 4. 前端介面設計

### 4.1 儀表板 (Dashboard)
- **目的**：提供整體系統概況與快速進入點。
- **主要元件**：
  - 頂部導航列：Logo、患者管理、影像上傳、設定、幫助、使用者選單。
  - 左側邊欄：近期患者、待處理影像、報告審閱。
  - 主內容：
    - 快速操作卡片 (上傳影像、搜尋患者)
    - 系統狀態指標 (待處理影像數、新報告數)
    - 統計概覽 (本週處理量、病徵統計)

### 4.2 患者管理
- **患者列表**：可依姓名、ID、出生日期、最後就診日排序/篩選。
- **詳細面板**：顯示基本資訊、影像歷史、分析報告歷史。
- **操作**：新增/編輯患者、查閱影像與報告、產出 PDF。

### 4.3 影像上傳與預處理
- **拖放式上傳**：支援批次拖放與點擊選擇，顯示即時進度。
- **預覽與標記**：上傳後呈現縮圖，允許綁定患者、指定影像類型 (全口、CBCT、單顆等)。
- **預處理參數**：預留影像增強、去噪等選項，並可選擇是否立即觸發 AI 分析。

### 4.4 影像分析結果
- **左側影像區**：
  - 影像檢視工具 (縮放、平移、亮度/對比調整、灰階反轉)。
  - 疊加層可切換不同病徵與模型輸出。
  - 點選病徵列表時自動聚焦到對應區域。
- **右側資訊區**：
  - 患者資訊與整體評估摘要。
  - 病徵列表 (支援 plugin 模型輸出) 顯示類型、位置、嚴重度、置信度、模型版本。
  - 人工確認與備註欄位，支援編輯紀錄。
  - 報告產出、加入治療計畫、下載 DICOM/CSV。

### 4.5 設定頁面
- **用戶偏好**：語言、主題、通知。
- **模型管理**：
  - 模型卡片顯示狀態、版本、支持的病徵類型、最後更新時間。
  - 調整閾值、靈敏度、輸出欄位對應。
  - 啟用/停用模型、滾動更新、回滾。
- **權限管理**：角色、使用者授權、審核流程設定。
- **資料備援**：備份策略、恢復程序、API 金鑰管理。

## 5. 後端服務與 API

### 5.1 API Gateway
- 授權：OAuth 2.0 + JWT，支援單一登入 (SSO) 與多因子。
- 篩選：Rate limiting、防重放攻擊、審計。

### 5.2 REST / GraphQL API 範例

| 功能 | Method & Path | 說明 |
| --- | --- | --- |
| 查詢患者 | `GET /api/patients?search=` | 支援條件搜尋與分頁 |
| 新增患者 | `POST /api/patients` | 驗證欄位後寫入 MySQL |
| 上傳影像 | `POST /api/images` | 透過 pre-signed URL 上傳，成功後寫入中繼資料 |
| 發起分析 | `POST /api/analyses` | 建立推論任務，送入佇列 |
| 查詢分析結果 | `GET /api/analyses/{id}` | 回傳統一格式的病徵列表與影像檢視連結 |
| 管理模型 | `POST /api/models/{model_id}/actions` | 啟用/停用/更新/回滾 |
| 報告生成 | `POST /api/reports` | 依模板輸出 PDF/HTML |

### 5.3 WebSocket / SSE
- 提供即時推播：上傳進度、分析完成通知、報告審核提醒。

### 5.4 插件式模型註冊流程

1. **模型封裝**：每個模型以 Docker 映像提供，內含推論 API，遵循標準 `/predict` 與 `/metadata` 介面。
2. **metadata**：模型回傳病徵類型、輸出欄位、可調參數、版本資訊。
3. **註冊**：管理員透過設定頁面上傳 metadata，系統寫入 `ai_models` 資料表。
4. **啟用**：AI Pipeline 讀取啟用模型清單，載入對應容器並保持健康檢查。
5. **輸出格式**：所有模型輸出 JSON 需符合下列 Schema (見第 6 章)。

## 6. 通用資料結構

### 6.1 資料庫 ERD (節錄)

```
Patients (id, name, dob, gender, contact, medical_history, ...)
Images (id, patient_id, type, status, storage_uri, captured_at, ...)
Analyses (id, image_id, status, requested_by, triggered_at, completed_at, ...)
Findings (id, analysis_id, finding_type, tooth_number, region, severity, confidence, model_version, ...)
ModelConfigs (id, model_key, name, version, status, params_json, ...)
UserNotes (id, finding_id, author_id, note, confirmed, ...)
AuditLogs (id, actor_id, action, target_type, payload, created_at)
```

### 6.2 統一病徵資料 Schema

```json
{
  "finding_id": "uuid",
  "type": "caries | periodontal | periapical | bone_loss | cyst | tumor | ...",
  "tooth_label": "FDI-11",
  "region": {
    "bbox": [x, y, width, height],
    "mask_uri": "s3://.../mask.png"
  },
  "severity": "mild | moderate | severe",
  "confidence": 0.92,
  "model_key": "caries_detector",
  "model_version": "v1.2.0",
  "extra": {
    "distance_to_pulp": 1.4,
    "cbct_slice": null
  }
}
```

- **擴充性**：`type` 與 `extra` 欄位可延伸，新模型只需沿用主結構。
- **座標系統**：採像素座標與 DICOM 參考點，支援多種影像尺度。

## 7. AI 模型工作流

### 7.1 推論流程

1. 使用者上傳影像並觸發分析。
2. Backend 將任務寫入 `analysis_jobs` 佇列。
3. Pipeline Orchestrator 擷取任務，依啟用模型列表建立推論流水線。
4. 針對同一影像，各模型可並行執行。以 `infer_cross_cam.py` 為模板封裝，統一輸入：
   - 影像 URI
   - 模型參數 (閾值、影像增強設定)
5. 模型完成後回傳標準化 JSON，Orchestrator 彙整為 Findings。
6. 結果存入資料庫並推播前端。

### 7.2 訓練流程

1. 管理員透過後端上傳訓練資料與設定。
2. Pipeline 觸發 `train_cross_cam.py` 或其他模型訓練腳本。
3. 訓練過程紀錄於 MLflow / TensorBoard。
4. 完成後將最佳權重上傳至模型倉庫 (S3 + metadata)。
5. 經驗證後可發佈成新版本，透過 Canary 或 A/B 測試逐步釋出。

### 7.3 模型參數與設定
- 每個模型在 `ModelConfigs` 表中維護預設參數 (ex: 信心閾值、NMS 設定)。
- 前端設定頁面提供 UI 調整，後端寫回 JSON 欄位。
- 推論時將設定注入模型容器，確保版本一致性。

## 8. 安全與合規

- **身份驗證**：整合身份供應商 (Azure AD / Keycloak) 與多因素。
- **存取控制**：以角色為基礎 (RBAC)，細緻到影像與報告層級。
- **資料保護**：
  - 傳輸層使用 TLS 1.2 以上。
  - 靜態資料使用資料庫與物件儲存加密。
  - 匿名化功能：可在匯出報告時移除患者識別資訊。
- **審計**：所有重大操作記錄於 AuditLogs，提供變更歷史。

## 9. DevOps 與部署

- **CI/CD**：
  - 使用 GitHub Actions / GitLab CI 進行測試、Lint、建置。
  - 模型映像 (Docker) 透過同一 pipeline 自動打包並發佈。
- **環境劃分**：Dev / Staging / Production，各環境可連接不同模型版本。
- **部署選項**：
  - Kubernetes：部署前後端、FastAPI、模型服務與訊息佇列。
  - Serverless (可選)：影像上傳與報告生成可採用 Lambda / Cloud Functions。
- **監控**：Prometheus + Grafana 監控系統負載；ELK 堆疊做日誌分析；Sentry 監控前端錯誤。

## 10. 未來擴展方向

- **多模態整合**：支援 CBCT、口內相機影像與臨床紀錄，建立跨模態分析。
- **診療支援**：導入治療計畫建議、預約系統整合。
- **AI 治療模擬**：提供 3D 模型與虛擬治療模擬結果。
- **病徵知識庫**：建立診斷依據與病例對照，提高 AI 透明度。


## 11. 實際部署與環境設定指引

1. **資料庫準備**
   - 建議在 PostgreSQL / MySQL / Azure SQL 建立獨立資料庫與具備 DDL 權限的帳號。
   - 於部署主機設定環境變數 `DATABASE_URL`（範例：`postgresql+psycopg://oral_ai:StrongPass@db-host:5432/oral_ai`）。
   - 首次啟動 FastAPI 會自動建立必要資料表；正式環境建議導入 Alembic 以版本化 schema。
2. **後端環境變數**
   - `SECRET_KEY`：JWT 簽章用的長字串，至少 32 bytes，正式環境勿外洩。
   - `ACCESS_TOKEN_EXPIRE_MINUTES`：存取權杖有效時間（預設 60 分鐘）。
   - `SESSION_EXPIRE_DAYS`：伺服器端登入工作階段保存天數（預設 7 天）。
   - 於本地開發可在 `.env` 建立上述值並透過 `uvicorn --env-file .env backend.main:app --reload` 啟動。
3. **檔案儲存**
   - FastAPI 預設將影像存放至專案根目錄 `uploaded_images/`。
   - 正式環境可改為掛載 NFS / S3 互動層：設定 `uploaded_images` 為掛載點或改寫 `_persist_upload`。
4. **帳號管理流程**
   - 透過 `/api/auth/register` 建立新使用者，密碼將以 bcrypt 雜湊存放。
   - `/api/auth/login` 回傳含 `access_token` 的 JWT，並於 `UserSession` 表寫入可撤銷的 session 記錄。
   - `/api/auth/logout` 與 `/api/auth/change-password` 會標記工作階段失效或更新雜湊，若需強制登出所有裝置可直接清除 `user_sessions` 中相同 `user_id` 的紀錄。
5. **前端整合**
   - 新增 `/login`、`/register`、`/account` 頁面，於瀏覽器 `localStorage` 保存存取權杖並自動附加於受保護 API。
   - 未登入的使用者將被導向 `/login`，上傳影像或建立新病患時會自動檢查權杖是否過期。
6. **驗證流程**
   - 啟動後端 `uvicorn backend.main:app --reload` 並確認 `/docs` 可使用。
   - 於前端資料夾執行 `npm install && npm run dev`，登入成功後即可從 `/account` 管理個人資料並測試影像上傳流程。

---

本設計著重於模組化與擴展性，透過統一資料結構與插件式模型管理，確保系統能隨著新病徵與新演算法快速演進，同時兼顧臨床使用者體驗與工程維運效率。

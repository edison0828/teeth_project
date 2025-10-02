export interface SeveritySlice {
  level: string;
  percentage: number;
}

export interface ConditionSummary {
  label: string;
  count: number;
  severity_breakdown: SeveritySlice[];
}

export interface QuickAction {
  id: string;
  label: string;
  description?: string;
  action: string;
}

export interface SystemStatus {
  pending_images: number;
  new_reports: number;
  models_active: number;
  active_model_name?: string | null;
  last_synced: string;
}

export interface StatisticsOverview {
  weekly_volume: number;
  week_over_week_change: number;
  detection_rate: number;
  average_processing_time: number;
  uptime_percentage: number;
}

export interface PatientSummary {
  id: string;
  name: string;
  last_visit?: string | null;
  most_recent_study?: string | null;
}

export interface ImageQueueItem {
  id: string;
  patient_id: string;
  patient_name: string;
  image_type: string;
  submitted_at: string;
  status: string;
}

export interface DashboardOverview {
  quick_actions: QuickAction[];
  system_status: SystemStatus;
  statistics: StatisticsOverview;
  detected_conditions: ConditionSummary[];
  recent_patients: PatientSummary[];
  pending_images: ImageQueueItem[];
}

export interface PatientListResponse {
  items: PatientSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface ImageMetadata {
  id: string;
  patient_id: string;
  type: string;
  captured_at: string;
  status: string;
  storage_uri?: string | null;
  public_url?: string | null;
}

export interface ImageUploadResponse {
  upload_url: string;
  image: ImageMetadata;
  auto_analyze: boolean;
  analysis_id?: string | null;
}

export type ModelType = "cross_attn" | "yolo_caries";

export interface ModelConfig {
  id: string;
  name: string;
  description?: string | null;
  model_type: ModelType;
  detector_path: string;
  classifier_path: string;
  detector_threshold: number;
  classification_threshold: number;
  max_teeth: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelConfigCreateRequest {
  name: string;
  description?: string | null;
  model_type: ModelType;
  detector_path: string;
  classifier_path: string;
  detector_threshold: number;
  classification_threshold: number;
  max_teeth: number;
  is_active?: boolean;
}

export interface ModelConfigUpdateRequest extends Partial<ModelConfigCreateRequest> {
  is_active?: boolean;
}


export interface AnalysisSummary {
  id: string;
  image_id: string;
  requested_by: string;
  status: string;
  triggered_at: string;
  completed_at?: string | null;
  overall_assessment?: string | null;
  preview?: AnalysisPreview | null;
}

export interface PatientDetail {
  id: string;
  name: string;
  dob: string;
  gender: string;
  contact?: string;
  medical_history?: string;
  last_visit?: string | null;
  notes?: string | null;
  recent_images: ImageMetadata[];
  recent_analyses: AnalysisSummary[];
}

export interface AnalysisPreviewFinding {
  finding_id: string;
  tooth_label?: string | null;
  bbox: number[];
  severity: string;
  confidence: number;
  assets?: Record<string, string | null> | null;
  bbox_normalized?: number[] | null;
  centroid?: number[] | null;
  color_bgr?: number[] | null;
}

export interface AnalysisPreview {
  image_uri?: string | null;
  overlay_uri?: string | null;
  image_size?: number[] | null;
  findings: AnalysisPreviewFinding[];
}

export interface AnalysisFinding {
  finding_id: string;
  type: string;
  tooth_label?: string | null;
  region: {
    bbox: number[];
    mask_uri?: string | null;
  };
  severity: string;
  confidence: number;
  model_key: string;
  model_version: string;
  extra: Record<string, unknown>;
  note?: string | null;
  confirmed?: boolean | null;
}

export interface ReportAction {
  label: string;
  description: string;
  href: string;
}

export interface TimelineEvent {
  timestamp: string;
  title: string;
  description: string;
  status: string;
}

export interface AnalysisDetail extends AnalysisSummary {
  patient: PatientSummary;
  image: ImageMetadata;
  findings: AnalysisFinding[];
  detected_conditions: ConditionSummary[];
  report_actions: ReportAction[];
  progress: {
    steps: TimelineEvent[];
    overall_confidence: number;
  };
}

export interface UserProfile {
  id: string;
  email: string;
  full_name?: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserCreateRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface UserUpdateRequest {
  email?: string;
  full_name?: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}


export interface DemoBoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DemoToothFinding {
  fdi: string;
  prob_caries: number;
  thr_used: number;
  pred: boolean;
  bbox: DemoBoundingBox;
  orig_image?: string | null;
  cam_path?: string | null;
  roi_path?: string | null;
}

export interface DemoSampleSummary {
  id: string;
  title: string;
  description: string;
  image_path: string;
  overlay_path?: string | null;
  cam_paths?: Record<string, string>;
}

export interface DemoSampleListResponse {
  items: DemoSampleSummary[];
}

export interface DemoInferenceResult {
  request_id: string;
  overlay_url: string;
  csv_url: string;
  findings: DemoToothFinding[];
  warnings: string[];
}

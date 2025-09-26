import { type AnalysisDetail, type AnalysisSummary, type DashboardOverview, type ModelConfig, type PatientDetail, type PatientListResponse } from "./types";

const nowIso = new Date().toISOString();

export const fallbackDashboard: DashboardOverview = {
  quick_actions: [
    {
      id: "upload",
      label: "Upload New Image",
      description: "Drag DICOM files or panoramic studies",
      action: "/image-upload"
    },
    {
      id: "search",
      label: "Search Patients",
      description: "Look up patients by name or MRN",
      action: "/patients"
    }
  ],
  system_status: {
    pending_images: 2,
    new_reports: 3,
    models_active: 2,
    active_model_name: "Mock Model v1",
    last_synced: nowIso
  },
  statistics: {
    weekly_volume: 38,
    week_over_week_change: 0.08,
    detection_rate: 0.82,
    average_processing_time: 7.1,
    uptime_percentage: 0.992
  },
  detected_conditions: [
    {
      label: "Caries",
      count: 12,
      severity_breakdown: [
        { level: "mild", percentage: 0.4 },
        { level: "moderate", percentage: 0.4 },
        { level: "severe", percentage: 0.2 }
      ]
    },
    {
      label: "Periodontal",
      count: 8,
      severity_breakdown: [
        { level: "mild", percentage: 0.6 },
        { level: "moderate", percentage: 0.3 },
        { level: "severe", percentage: 0.1 }
      ]
    }
  ],
  recent_patients: [
    {
      id: "P12345",
      name: "Jane Doe",
      last_visit: "2023-10-30",
      most_recent_study: "Panoramic"
    },
    {
      id: "P72631",
      name: "John Smith",
      last_visit: "2023-09-14",
      most_recent_study: "Bitewing"
    },
    {
      id: "P84021",
      name: "Amy Chen",
      last_visit: "2023-08-22",
      most_recent_study: "CBCT"
    }
  ],
  pending_images: [
    {
      id: "IMG-902",
      patient_id: "P72631",
      patient_name: "John Smith",
      image_type: "Bitewing",
      submitted_at: nowIso,
      status: "queued"
    },
    {
      id: "IMG-945",
      patient_id: "P84021",
      patient_name: "Amy Chen",
      image_type: "CBCT",
      submitted_at: nowIso,
      status: "processing"
    }
  ]
};

export const fallbackPatients: PatientListResponse = {
  items: fallbackDashboard.recent_patients,
  total: fallbackDashboard.recent_patients.length,
  page: 1,
  page_size: 10
};

export const fallbackPatientDetail: PatientDetail = {
  id: "P12345",
  name: "Jane Doe",
  dob: "1985-03-12",
  gender: "female",
  contact: "jane.doe@example.com",
  medical_history: "Allergy: Penicillin. Previous root canal.",
  last_visit: "2023-10-30",
  notes: "Prefers SMS reminders.",
  recent_images: [
    {
      id: "IMG-001",
      patient_id: "P12345",
      type: "Panoramic",
      captured_at: nowIso,
      status: "analyzed",
      storage_uri: "s3://oral-xray/panoramic/P12345_20231030.png",
      public_url: "/mock-data/panoramic.png"
    }
  ],
  recent_analyses: [
    {
      id: "AN-901",
      image_id: "IMG-001",
      requested_by: "Dr. Lee",
      status: "completed",
      triggered_at: nowIso,
      completed_at: nowIso,
      overall_assessment: "Found 2 caries, 1 periodontal lesion"
    }
  ]
};

export const fallbackAnalysesSummary: AnalysisSummary[] = [
  {
    id: "AN-901",
    image_id: "IMG-001",
    requested_by: "Dr. Lee",
    status: "completed",
    triggered_at: nowIso,
    completed_at: nowIso,
    overall_assessment: "Found 2 caries, 1 periodontal lesion",
    preview: {
      image_uri: "/mock-data/panoramic.png",
      overlay_uri: "/mock-data/panoramic_overlay.png",
      image_size: [2048, 1024],
      findings: [
        {
          finding_id: "FND-PREVIEW-1",
          tooth_label: "FDI-26",
          bbox: [240, 132, 88, 76],
          severity: "moderate",
          confidence: 0.87,
          assets: { gradcam: "/mock-data/gradcam_26.png" }
        }
      ]
    },
  },
  {
    id: "AN-902",
    image_id: "IMG-902",
    requested_by: "Dr. Wu",
    status: "processing",
    triggered_at: nowIso,
    completed_at: null,
    overall_assessment: null
  },
  {
    id: "AN-903",
    image_id: "IMG-945",
    requested_by: "Dr. Chen",
    status: "queued",
    triggered_at: nowIso,
    completed_at: null,
    overall_assessment: null
  }
];

export const fallbackAnalysis: AnalysisDetail = {
  id: "AN-901",
  image_id: "IMG-001",
  requested_by: "Dr. Lee",
  status: "completed",
  triggered_at: nowIso,
  completed_at: nowIso,
  overall_assessment: "Found 2 caries, 1 periodontal lesion.",
  preview: {
    image_uri: "/mock-data/panoramic.png",
    overlay_uri: "/mock-data/panoramic_overlay.png",
    image_size: [2048, 1024],
    findings: [
      {
        finding_id: "FND-1",
        tooth_label: "FDI-26",
        bbox: [240, 132, 88, 76],
        severity: "moderate",
        confidence: 0.87,
        assets: { gradcam: "/mock-data/gradcam_26.png" }
      }
    ]
  },
  patient: fallbackPatientDetail,
  image: fallbackPatientDetail.recent_images[0],
  detected_conditions: fallbackDashboard.detected_conditions,
  findings: [
    {
      finding_id: "FND-1",
      type: "caries",
      tooth_label: "FDI-26",
      region: { bbox: [240, 132, 88, 76] },
      severity: "moderate",
      confidence: 0.87,
      model_key: "caries_detector",
      model_version: "v1.2.0",
      extra: { distance_to_pulp: 1.4 },
      note: "Verify lesion depth",
      confirmed: true
    },
    {
      finding_id: "FND-2",
      type: "periodontal",
      tooth_label: "FDI-16",
      region: { bbox: [120, 220, 64, 54] },
      severity: "mild",
      confidence: 0.74,
      model_key: "perio_segmenter",
      model_version: "v0.9.1",
      extra: {},
      confirmed: false
    }
  ],
  report_actions: [
    { label: "Generate Report", description: "Export PDF clinical report", href: "#" },
    { label: "Download CSV", description: "Download raw findings data", href: "#" }
  ],
  progress: {
    overall_confidence: 0.82,
    steps: [
      {
        timestamp: nowIso,
        title: "Upload received",
        description: "Image queued for preprocessing",
        status: "done"
      },
      {
        timestamp: nowIso,
        title: "AI models",
        description: "Running detectors",
        status: "done"
      },
      {
        timestamp: nowIso,
        title: "Awaiting review",
        description: "Assigned to Dr. Lee for confirmation",
        status: "in-progress"
      }
    ]
  }
};
export const fallbackModels: ModelConfig[] = [
  {
    id: "MODEL-MOCK",
    name: "Mock Model v1",
    description: "Default mock configuration for offline mode",
    model_type: "cross_attn",
    detector_path: "models/fdi_all seg.pt",
    classifier_path: "models/cross_attn_fdi_camAlignA.pth",
    detector_threshold: 0.25,
    classification_threshold: 0.5,
    max_teeth: 64,
    is_active: true,
    created_at: nowIso,
    updated_at: nowIso
  },
  {
    id: "MODEL-MOCK-YOLO",
    name: "YOLO Caries Demo",
    description: "Demo configuration for direct caries detection",
    model_type: "yolo_caries",
    detector_path: "models/fdi_all seg.pt",
    classifier_path: "models/yolo_caries.pt",
    detector_threshold: 0.2,
    classification_threshold: 0.4,
    max_teeth: 64,
    is_active: false,
    created_at: nowIso,
    updated_at: nowIso
  }
];

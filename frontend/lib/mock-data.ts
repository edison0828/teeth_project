import { type AnalysisDetail, type DashboardOverview, type PatientDetail, type PatientListResponse } from "./types";

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
    models_active: 4,
    last_synced: new Date().toISOString()
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
    }
  ],
  pending_images: [
    {
      id: "IMG-902",
      patient_id: "P72631",
      patient_name: "John Smith",
      image_type: "Bitewing",
      submitted_at: new Date().toISOString(),
      status: "queued"
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
      captured_at: new Date().toISOString(),
      status: "analyzed",
      storage_uri: "s3://oral-xray/panoramic/P12345_20231030.png"
    }
  ],
  recent_analyses: [
    {
      id: "AN-901",
      image_id: "IMG-001",
      requested_by: "Dr. Lee",
      status: "completed",
      triggered_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      overall_assessment: "Found 2 caries, 1 periodontal lesion"
    }
  ]
};

export const fallbackAnalysis: AnalysisDetail = {
  id: "AN-901",
  image_id: "IMG-001",
  requested_by: "Dr. Lee",
  status: "completed",
  triggered_at: new Date().toISOString(),
  completed_at: new Date().toISOString(),
  overall_assessment: "Found 2 caries, 1 periodontal lesion.",
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
        timestamp: new Date().toISOString(),
        title: "Upload received",
        description: "Image queued for preprocessing",
        status: "done"
      },
      {
        timestamp: new Date().toISOString(),
        title: "AI models",
        description: "Running detectors",
        status: "done"
      }
    ]
  }
};

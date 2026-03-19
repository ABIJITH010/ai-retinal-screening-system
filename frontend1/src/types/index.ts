// ── Severity Levels ──
export type SeverityLevel = "critical" | "high" | "moderate" | "stable";

// ── Screening / Prediction ──
export interface PredictionResult {
  label: string;
  stage: string;
  confidence: number;
  description: string;
}

export interface ScreeningResult {
  id: string;
  imageUrl: string;
  prediction: PredictionResult;
  heatmapUrl: string;
  createdAt: string;
}

// ── Patient Records ──
export interface PatientRecord {
  id: string;
  name: string;
  initials: string;
  date: string;
  result: string;
  resultType: "healthy" | "warning" | "critical";
  confidence: number;
}

// ── Triage ──
export interface TriageCase {
  id: string;
  name: string;
  patientId: string;
  severity: SeverityLevel;
  statusLabel: string;
  aiInsight: string;
  timeAgo: string;
}

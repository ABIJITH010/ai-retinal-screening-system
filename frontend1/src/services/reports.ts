import api from "./api";

export interface ReportResponse {
  id: number;
  patient_id: number;
  image_url: string;
  heatmap_url: string;
  prediction: string;
  confidence: number; // backend uses 0..1
  created_at: string;
}

export async function createReport(params: { patientId: number; file: File }) {
  const formData = new FormData();
  formData.append("file", params.file);

  const { data } = await api.post<ReportResponse>("/reports", formData, {
    params: { patient_id: params.patientId },
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listReports() {
  const { data } = await api.get<ReportResponse[]>("/reports");
  return data;
}


import api from "./api";

export interface PatientCreateRequest {
  name: string;
  age: number;
  gender: string;
  phone: string;
  address: string;
}

export interface PatientResponse {
  id: number;
  name: string;
  age: number;
  gender: string;
  phone: string;
  address: string;
  created_at: string;
  doctor_id: number;
}

export async function listPatients() {
  const { data } = await api.get<PatientResponse[]>("/patients");
  return data;
}

export async function createPatient(request: PatientCreateRequest) {
  const { data } = await api.post<PatientResponse>("/patients", request);
  return data;
}


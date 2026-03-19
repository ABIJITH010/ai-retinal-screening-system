import axios, { AxiosHeaders, type InternalAxiosRequestConfig } from "axios";
import { API_BASE_URL } from "../utils/constants";
import { getAuthToken } from "./authStorage";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getAuthToken();
  if (!token) return config;

  const headers = AxiosHeaders.from(config.headers);
  headers.set("Authorization", `Bearer ${token}`);
  config.headers = headers;

  return config;
});

/** Upload a retinal image for AI analysis */
export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append("image", file);
  const { data } = await api.post("/screening/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/** Retrieve prediction results */
export async function getPredictions() {
  const { data } = await api.get("/screening/predictions");
  return data;
}

/** Retrieve patient records with optional filters */
export async function getRecords(params?: { search?: string; severity?: string; page?: number }) {
  const { data } = await api.get("/records", { params });
  return data;
}

/** Retrieve triage cases */
export async function getTriageCases() {
  const { data } = await api.get("/triage");
  return data;
}

export default api;

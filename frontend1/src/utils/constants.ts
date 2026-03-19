export const APP_NAME = "Retina Max";

export const NAV_ITEMS = [
  { label: "Dashboard", icon: "grid_view", path: "/" },
  { label: "Screening", icon: "visibility", path: "/screening" },
  { label: "Records", icon: "folder_shared", path: "/records" },
  { label: "Triage", icon: "medical_services", path: "/triage" },
  { label: "Settings", icon: "settings", path: "/settings" },
] as const;

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

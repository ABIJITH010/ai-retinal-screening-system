import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { getAuthToken } from "../../services/authStorage";

export default function RequireAuth({ children }: { children: ReactNode }) {
  const location = useLocation();
  const token = getAuthToken();

  if (!token) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children;
}


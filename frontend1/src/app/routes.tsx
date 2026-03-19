import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import RequireAuth from "../features/auth/RequireAuth";

// Lazy-loaded pages
const Login = lazy(() => import("../features/auth/Login"));
const Register = lazy(() => import("../features/auth/Register"));
const Dashboard = lazy(() => import("../features/dashboard/Dashboard"));
const Screening = lazy(() => import("../features/screening/Screening"));
const Records = lazy(() => import("../features/records/Records"));
const Triage = lazy(() => import("../features/triage/Triage"));
const Settings = lazy(() => import("../features/settings/Settings"));
const Profile = lazy(() => import("../features/profile/Profile"));
const Notifications = lazy(() => import("../features/notifications/Notifications"));

export default function AppRoutes() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center">
          <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        </div>
      }
    >
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        <Route
          path="/"
          element={
            <RequireAuth>
              <Dashboard />
            </RequireAuth>
          }
        />
        <Route
          path="/screening"
          element={
            <RequireAuth>
              <Screening />
            </RequireAuth>
          }
        />
        <Route
          path="/records"
          element={
            <RequireAuth>
              <Records />
            </RequireAuth>
          }
        />
        <Route
          path="/triage"
          element={
            <RequireAuth>
              <Triage />
            </RequireAuth>
          }
        />
        <Route
          path="/settings"
          element={
            <RequireAuth>
              <Settings />
            </RequireAuth>
          }
        />
        <Route
          path="/profile"
          element={
            <RequireAuth>
              <Profile />
            </RequireAuth>
          }
        />
        <Route
          path="/notifications"
          element={
            <RequireAuth>
              <Notifications />
            </RequireAuth>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}

import { BrowserRouter } from "react-router-dom";
import { useLocation } from "react-router-dom";
import Navbar from "../components/ui/Navbar";
import BottomNav from "../components/ui/BottomNav";
import AppRoutes from "./routes";

function AppChrome() {
  const location = useLocation();
  const isAuthRoute =
    location.pathname === "/login" || location.pathname === "/register";

  return (
    <div className="min-h-screen font-body selection:bg-primary/30">
      {!isAuthRoute && <Navbar />}
      <AppRoutes />
      {!isAuthRoute && <BottomNav />}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppChrome />
    </BrowserRouter>
  );
}

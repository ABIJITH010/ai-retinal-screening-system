import { useNavigate } from "react-router-dom";
import { APP_NAME } from "../../utils/constants";
import { clearAuthToken } from "../../services/authStorage";

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <header className="fixed top-0 left-0 right-0 z-[60] bg-transparent font-headline font-semibold tracking-tight">
      <div className="flex justify-between items-center w-full px-8 py-4">
        <div 
          className="text-xl font-bold text-on-surface cursor-pointer"
          onClick={() => navigate('/')}
        >
          {APP_NAME}
        </div>
        <div className="flex items-center gap-6">
          <button
            onClick={() => {
              clearAuthToken();
              navigate("/login", { replace: true });
            }}
            className="text-on-surface-variant hover:text-primary transition-colors scale-95 active:scale-90 transition-transform flex items-center justify-center"
            title="Logout"
            aria-label="Logout"
          >
            <span className="material-symbols-outlined">logout</span>
          </button>
          <button 
            onClick={() => navigate('/notifications')}
            className="text-on-surface-variant hover:text-primary transition-colors scale-95 active:scale-90 transition-transform flex items-center justify-center"
          >
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button 
            onClick={() => navigate('/profile')}
            className="text-on-surface-variant hover:text-primary transition-colors scale-95 active:scale-90 transition-transform flex items-center justify-center"
          >
            <span className="material-symbols-outlined">account_circle</span>
          </button>
        </div>
      </div>
    </header>
  );
}

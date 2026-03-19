import { NavLink, useLocation } from "react-router-dom";
import { NAV_ITEMS } from "../../utils/constants";

export default function BottomNav() {
  const location = useLocation();

  return (
    <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 p-2 mb-6 bg-surface-container-high backdrop-blur-xl rounded-full mx-auto w-fit shadow-[0_0_32px_rgba(148,34,156,0.1)] font-label text-xs font-medium border border-outline-variant/10">
      {NAV_ITEMS.map((item) => {
        const isActive = location.pathname === item.path;
        return (
          <NavLink
            key={item.path}
            to={item.path}
            className={`flex flex-col items-center justify-center rounded-full px-6 py-2 transition-all duration-300 ease-out ${
              isActive
                ? "bg-primary/15 text-primary scale-105"
                : "text-on-surface-variant hover:bg-surface-container-highest hover:text-on-surface"
            }`}
          >
            <span
              className="material-symbols-outlined"
              style={
                isActive
                  ? { fontVariationSettings: "'FILL' 1" }
                  : undefined
              }
            >
              {item.icon}
            </span>
            <span>{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}

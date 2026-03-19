export type ThemeMode = "dark" | "light";

const THEME_STORAGE_KEY = "retinamax_theme_mode";
const LIGHT_CLASS = "theme-light";

export function getThemeMode(): ThemeMode {
  try {
    const raw = localStorage.getItem(THEME_STORAGE_KEY);
    return raw === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function applyThemeMode(mode: ThemeMode) {
  const root = document.documentElement;
  root.classList.toggle(LIGHT_CLASS, mode === "light");
}

export function setThemeMode(mode: ThemeMode) {
  localStorage.setItem(THEME_STORAGE_KEY, mode);
  applyThemeMode(mode);
}

export function initThemeMode() {
  applyThemeMode(getThemeMode());
}


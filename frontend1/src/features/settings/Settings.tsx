import Card from "../../components/ui/Card";
import { useState } from "react";
import { getThemeMode, setThemeMode, type ThemeMode } from "../../services/theme";

export default function Settings() {
  const [themeMode, setThemeModeState] = useState<ThemeMode>(() => getThemeMode());

  return (
    <main className="min-h-screen pt-24 pb-32 px-6 md:px-12 max-w-4xl mx-auto">
      <div className="mb-10">
        <h1 className="text-4xl font-headline font-extrabold text-on-surface tracking-tight">
          Settings
        </h1>
        <p className="text-on-surface-variant text-lg">
          Configure application preferences and AI diagnostics parameters.
        </p>
      </div>

      <div className="space-y-8">
        {/* A. Model Settings */}
        <Card className="p-8">
          <h3 className="text-xl font-headline font-bold mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">smart_toy</span>
            Model Settings
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Primary AI Model</label>
              <select className="w-full bg-surface-container-lowest border border-outline/10 rounded-xl px-4 py-3 text-on-surface outline-none focus:ring-1 focus:ring-primary/40">
                <option>EfficientNet Version 2.4</option>
                <option>ResNet-50 Optimized</option>
                <option>Ensemble (Experimental)</option>
              </select>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Confidence Threshold</label>
                <span className="text-xs font-bold text-primary">85%</span>
              </div>
              <input type="range" className="w-full accent-primary mt-3 h-1.5 bg-surface-container-highest rounded-lg cursor-pointer" />
              <p className="text-[10px] text-on-surface-variant mt-1">Predictions below this value will be flagged for manual review.</p>
            </div>
          </div>
        </Card>

        {/* B. System Preferences */}
        <Card className="p-8">
          <h3 className="text-xl font-headline font-bold mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">display_settings</span>
            System Preferences
          </h3>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-bold text-on-surface">Dark Mode</p>
                <p className="text-xs text-on-surface-variant">Switch between dark and light themes.</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  const next: ThemeMode = themeMode === "dark" ? "light" : "dark";
                  setThemeMode(next);
                  setThemeModeState(next);
                }}
                className={`w-12 h-6 rounded-full relative transition-colors duration-300 ${
                  themeMode === "dark" ? "bg-primary" : "bg-surface-container-highest"
                }`}
                aria-label="Toggle dark mode"
                aria-pressed={themeMode === "dark"}
              >
                <div
                  className={`absolute top-1 w-4 h-4 rounded-full shadow-sm transition-transform duration-300 ${
                    themeMode === "dark"
                      ? "bg-white translate-x-6"
                      : "bg-outline translate-x-1"
                  }`}
                />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-bold text-on-surface">Enable Animations</p>
                <p className="text-xs text-on-surface-variant">Smoother transitions and UI interactions.</p>
              </div>
              <button className="w-12 h-6 rounded-full bg-primary relative transition-colors duration-300">
                <div className="absolute right-1 top-1 w-4 h-4 rounded-full bg-white shadow-sm" />
              </button>
            </div>
          </div>
        </Card>

        {/* C. Notifications Settings */}
        <Card className="p-8">
          <h3 className="text-xl font-headline font-bold mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">notifications</span>
            Notification Preferences
          </h3>
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-bold text-on-surface">High-risk alerts</p>
                <p className="text-xs text-on-surface-variant">Instant alerts when critical pathology is detected.</p>
              </div>
              <button className="w-12 h-6 rounded-full bg-primary relative transition-colors duration-300">
                <div className="absolute right-1 top-1 w-4 h-4 rounded-full bg-white shadow-sm" />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-bold text-on-surface">Daily summary</p>
                <p className="text-xs text-on-surface-variant">Get a digest of daily screening performance.</p>
              </div>
              <button className="w-12 h-6 rounded-full bg-surface-container-highest relative transition-colors duration-300">
                <div className="absolute left-1 top-1 w-4 h-4 rounded-full bg-outline shadow-sm" />
              </button>
            </div>
          </div>
        </Card>

        {/* D. Danger Zone */}
        <Card className="p-8 border-2 border-error/10">
          <h3 className="text-xl font-headline font-bold mb-6 flex items-center gap-2 text-error">
            <span className="material-symbols-outlined text-error">warning</span>
            Danger Zone
          </h3>
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <p className="text-sm text-on-surface-variant max-w-md">
              Reset application state, clear cache, and re-initialize AI models. This action cannot be undone.
            </p>
            <button className="px-6 py-2.5 rounded-lg border border-error/50 text-error font-bold hover:bg-error/10 transition-colors text-sm whitespace-nowrap">
              Reset System
            </button>
          </div>
        </Card>
      </div>
    </main>
  );
}

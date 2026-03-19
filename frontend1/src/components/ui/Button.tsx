import type { ReactNode, ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  icon?: string;
  children: ReactNode;
}

const variantClasses: Record<string, string> = {
  primary:
    "bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-bold shadow-lg shadow-primary/20 hover:scale-105 transition-transform",
  secondary:
    "bg-surface-container-highest text-on-surface font-bold hover:bg-surface-container transition-colors",
  ghost:
    "border border-outline-variant/30 text-on-surface-variant font-bold hover:bg-surface-container-high transition-colors",
};

export default function Button({
  variant = "primary",
  icon,
  children,
  className = "",
  ...props
}: ButtonProps) {
  return (
    <button
      className={`px-6 py-2.5 rounded-lg flex items-center justify-center gap-2 text-sm active:scale-95 transition-transform ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {icon && (
        <span className="material-symbols-outlined text-sm">{icon}</span>
      )}
      {children}
    </button>
  );
}

import { useMousePosition } from "../../hooks/useMousePosition";

/**
 * Mechanical AI Eye — animated eye graphic with pupil that follows the cursor.
 * Fully self-contained component.
 */
export default function MechanicalEye() {
  const { x, y } = useMousePosition();

  const centerX = typeof window !== "undefined" ? window.innerWidth / 2 : 0;
  const centerY = typeof window !== "undefined" ? window.innerHeight / 2 : 0;

  const limit = 40;
  const moveX = Math.min(Math.max((x - centerX) / 40, -limit), limit);
  const moveY = Math.min(Math.max((y - centerY) / 40, -limit), limit);

  return (
    <div className="eye-container">
      {/* Outer ring */}
      <div className="outer-ring" />

      {/* Spinning dashed inner ring */}
      <div className="inner-ring" />

      {/* Pupil core — follows cursor */}
      <div
        className="pupil-core"
        style={{
          transform: `translate(calc(-50% + ${moveX}px), calc(-50% + ${moveY}px))`,
        }}
      >
        <div className="absolute inset-0 bg-white/20 rounded-full blur-sm transform scale-50 -translate-x-2 -translate-y-2" />
      </div>

      {/* Pulsating glow */}
      <div className="absolute inset-0 rounded-full bg-primary/5 animate-pulse blur-3xl -z-10" />
    </div>
  );
}

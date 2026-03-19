import { useEffect, useState, useCallback } from "react";

interface MousePosition {
  x: number;
  y: number;
}

/**
 * Custom hook that tracks mouse position relative to the viewport.
 * Uses requestAnimationFrame for smooth performance.
 */
export function useMousePosition(): MousePosition {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  const handleMouseMove = useCallback((e: MouseEvent) => {
    requestAnimationFrame(() => {
      setPosition({ x: e.clientX, y: e.clientY });
    });
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [handleMouseMove]);

  return position;
}

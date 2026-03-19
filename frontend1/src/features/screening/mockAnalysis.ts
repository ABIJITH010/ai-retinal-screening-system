export type DrStageLabel =
  | "No DR"
  | "Mild"
  | "Moderate"
  | "Severe"
  | "Proliferative DR";

export interface MockReport {
  id: number;
  patient_id: number;
  image_url: string;
  heatmap_url: string;
  prediction: DrStageLabel;
  confidence: number; // 0..1
  created_at: string;
  priority_score: 1 | 2 | 3 | 4 | 5;
}

function randomChoice<T>(values: readonly T[]) {
  return values[Math.floor(Math.random() * values.length)]!;
}

const STAGES: readonly DrStageLabel[] = [
  "No DR",
  "Mild",
  "Moderate",
  "Severe",
  "Proliferative DR",
];

export function severityFromStage(stage: string): 1 | 2 | 3 | 4 | 5 {
  switch (stage) {
    case "No DR":
      return 1;
    case "Mild":
      return 2;
    case "Moderate":
      return 3;
    case "Severe":
      return 4;
    case "Proliferative DR":
      return 5;
    default:
      return 3;
  }
}

export function stageDescription(stage: string) {
  switch (stage) {
    case "No DR":
      return "No visible signs of diabetic retinopathy detected.";
    case "Mild":
      return "Mild non-proliferative changes detected; microaneurysms may be present.";
    case "Moderate":
      return "Moderate non-proliferative changes detected; closer monitoring is recommended.";
    case "Severe":
      return "Severe non-proliferative changes detected; high risk of progression.";
    case "Proliferative DR":
      return "Proliferative DR detected; neovascularization suspected and urgent attention is recommended.";
    default:
      return "Diabetic retinopathy stage predicted.";
  }
}

export function mockHeatmapDataUrl() {
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="720" height="720" viewBox="0 0 720 720">
  <defs>
    <radialGradient id="g1" cx="50%" cy="45%" r="55%">
      <stop offset="0%" stop-color="#ffef5a" stop-opacity="0.95"/>
      <stop offset="35%" stop-color="#ff7a18" stop-opacity="0.75"/>
      <stop offset="65%" stop-color="#ff2a6d" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#7b2cff" stop-opacity="0.0"/>
    </radialGradient>
    <radialGradient id="g2" cx="35%" cy="65%" r="40%">
      <stop offset="0%" stop-color="#ff2a6d" stop-opacity="0.85"/>
      <stop offset="55%" stop-color="#ff7a18" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
    </radialGradient>
    <filter id="blur" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
  </defs>

  <rect width="720" height="720" fill="#0b0b14"/>
  <circle cx="360" cy="360" r="310" fill="#111128"/>
  <circle cx="360" cy="360" r="280" fill="url(#g1)" filter="url(#blur)"/>
  <circle cx="270" cy="460" r="190" fill="url(#g2)" filter="url(#blur)"/>

  <g opacity="0.65">
    <circle cx="430" cy="240" r="22" fill="#ffef5a"/>
    <circle cx="480" cy="320" r="16" fill="#ff7a18"/>
    <circle cx="300" cy="370" r="14" fill="#ff2a6d"/>
    <circle cx="250" cy="520" r="18" fill="#ffef5a"/>
    <circle cx="410" cy="520" r="13" fill="#ff7a18"/>
  </g>

  <circle cx="360" cy="360" r="300" fill="none" stroke="#ffffff" stroke-opacity="0.07" stroke-width="2"/>
</svg>`;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

export function createMockReport(params: {
  patientId: number;
  imageUrl: string;
  heatmapUrl?: string;
}): MockReport {
  const prediction = randomChoice(STAGES);
  const confidence = Number((0.7 + Math.random() * 0.29).toFixed(4));
  const severity = severityFromStage(prediction);

  return {
    id: Date.now(),
    patient_id: params.patientId,
    image_url: params.imageUrl,
    heatmap_url: params.heatmapUrl ?? mockHeatmapDataUrl(),
    prediction,
    confidence,
    created_at: new Date().toISOString(),
    priority_score: severity,
  };
}

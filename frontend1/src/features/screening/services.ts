import { uploadImage, getPredictions } from "../../services/api";
import type { ScreeningResult } from "../../types";

/** Upload a retinal image and return screening result */
export async function submitScreening(file: File): Promise<ScreeningResult> {
  const uploadResponse = await uploadImage(file);
  return uploadResponse as ScreeningResult;
}

/** Fetch latest prediction results */
export async function fetchPredictions() {
  const data = await getPredictions();
  return data;
}

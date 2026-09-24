export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type Approach = "diffusion" | "flow_matching";

export const APPROACH_LABELS: Record<Approach, string> = {
  diffusion: "DDPM (diffusion)",
  flow_matching: "Flow Matching",
};

export interface GenerateResponse {
  image_base64: string;
  approach: Approach;
  prompt: string;
  num_steps: number;
}

export interface StatusResponse {
  diffusion: boolean;
  flow_matching: boolean;
}

export interface TrainingStatusResponse {
  running: boolean;
  latest_step: number | null;
  loss_points: [number, number][];
  latest_preview_base64: string | null;
  latest_preview_step: number | null;
}

import { useEffect, useState, type FormEvent } from "react";
import {
  API_BASE_URL,
  APPROACH_LABELS,
  type Approach,
  type GenerateResponse,
  type StatusResponse,
} from "../api";

const DEFAULT_STEPS: Record<Approach, number> = {
  diffusion: 1000,
  flow_matching: 50,
};

function Generate() {
  const [prompt, setPrompt] = useState("a young woman smiling with bangs");
  const [approach, setApproach] = useState<Approach>("diffusion");
  const [numSteps, setNumSteps] = useState(DEFAULT_STEPS.diffusion);
  const [imageSize, setImageSize] = useState(64);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastGenerationTime, setLastGenerationTime] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/status`)
      .then((res) => res.json())
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  function handleApproachChange(next: Approach) {
    setApproach(next);
    setNumSteps(DEFAULT_STEPS[next]);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    const startedAt = performance.now();

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          approach,
          num_steps: numSteps,
          image_size: imageSize,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Erreur ${response.status}`);
      }

      const data: GenerateResponse = await response.json();
      setImageBase64(data.image_base64);
      setLastGenerationTime((performance.now() - startedAt) / 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
      setImageBase64(null);
    } finally {
      setIsLoading(false);
    }
  }

  const checkpointMissing = status !== null && !status[approach];

  return (
    <>
      <form className="form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Prompt</span>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="a smiling woman with bangs"
            required
          />
        </label>

        <div className="field-row">
          <label className="field">
            <span>Approche</span>
            <select value={approach} onChange={(e) => handleApproachChange(e.target.value as Approach)}>
              <option value="diffusion">{APPROACH_LABELS.diffusion}</option>
              <option value="flow_matching">{APPROACH_LABELS.flow_matching}</option>
            </select>
          </label>

          <label className="field">
            <span>Étapes de sampling</span>
            <input
              type="number"
              min={1}
              max={1000}
              value={numSteps}
              onChange={(e) => setNumSteps(Number(e.target.value))}
            />
          </label>

          <label className="field">
            <span>Taille image</span>
            <select value={imageSize} onChange={(e) => setImageSize(Number(e.target.value))}>
              <option value={64}>64×64</option>
              <option value={128}>128×128</option>
            </select>
          </label>
        </div>

        {checkpointMissing && (
          <p className="warning">
            Aucun checkpoint trouvé pour « {APPROACH_LABELS[approach]} » (checkpoints/{approach}/latest.pt).
            Entraînez d'abord le modèle via la CLI (voir README.md).
          </p>
        )}

        <button type="submit" disabled={isLoading}>
          {isLoading ? "Génération en cours…" : "Générer"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      <section className="result">
        {imageBase64 && (
          <>
            <img src={`data:image/png;base64,${imageBase64}`} alt={prompt} />
            {lastGenerationTime !== null && (
              <p className="meta">Généré en {lastGenerationTime.toFixed(1)}s</p>
            )}
          </>
        )}
      </section>
    </>
  );
}

export default Generate;

import { useEffect, useRef, useState } from "react";
import { API_BASE_URL, APPROACH_LABELS, type Approach, type TrainingStatusResponse } from "../api";
import LossChart from "../components/LossChart";

const POLL_INTERVAL_MS = 4000;

function Monitor() {
  const [approach, setApproach] = useState<Approach>("diffusion");
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function poll() {
      try {
        const response = await fetch(`${API_BASE_URL}/training-status?approach=${approach}`);
        if (!response.ok) throw new Error(`Erreur ${response.status}`);
        const data: TrainingStatusResponse = await response.json();
        setTrainingStatus(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Erreur inconnue");
      }
    }

    poll();
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [approach]);

  return (
    <>
      <div className="field-row">
        <label className="field">
          <span>Approche à surveiller</span>
          <select value={approach} onChange={(e) => setApproach(e.target.value as Approach)}>
            <option value="diffusion">{APPROACH_LABELS.diffusion}</option>
            <option value="flow_matching">{APPROACH_LABELS.flow_matching}</option>
          </select>
        </label>
      </div>

      {error && <p className="error">{error}</p>}

      {trainingStatus && !trainingStatus.running && (
        <p className="warning">
          Aucun entraînement trouvé pour « {APPROACH_LABELS[approach]} ». Lancez-en un via la CLI
          (voir README.md) — cette page se mettra à jour automatiquement toutes les {POLL_INTERVAL_MS / 1000}s.
        </p>
      )}

      {trainingStatus?.running && (
        <div className="monitor">
          <div className="monitor-panel">
            <h2>Loss (step {trainingStatus.latest_step ?? "—"})</h2>
            <LossChart points={trainingStatus.loss_points} />
          </div>

          <div className="monitor-panel">
            <h2>
              Dernière preview
              {trainingStatus.latest_preview_step !== null &&
                ` (step ${trainingStatus.latest_preview_step})`}
            </h2>
            {trainingStatus.latest_preview_base64 ? (
              <img
                className="preview-image"
                src={`data:image/png;base64,${trainingStatus.latest_preview_base64}`}
                alt="Dernière grille d'images générées pendant l'entraînement"
              />
            ) : (
              <p className="meta">Pas encore de preview générée.</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default Monitor;

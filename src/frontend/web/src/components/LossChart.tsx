interface LossChartProps {
  points: [number, number][]; // [step, loss][]
  width?: number;
  height?: number;
}

/**
 * Line chart minimal en SVG pur (pas de dépendance de charting) pour la
 * courbe de loss pendant l'entraînement. Suffisant pour une seule série,
 * mise à jour périodiquement par polling (voir pages/Monitor.tsx).
 */
function LossChart({ points, width = 480, height = 220 }: LossChartProps) {
  if (points.length < 2) {
    return <p className="meta">En attente de plus de données ({points.length} point(s))…</p>;
  }

  const padding = { top: 12, right: 12, bottom: 28, left: 48 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const steps = points.map((p) => p[0]);
  const losses = points.map((p) => p[1]);

  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);

  // Marge verticale pour ne pas coller la courbe aux bords du graphique.
  const lossRange = maxLoss - minLoss || 1;
  const yMin = minLoss - lossRange * 0.1;
  const yMax = maxLoss + lossRange * 0.1;

  function xToPixel(step: number): number {
    if (maxStep === minStep) return padding.left;
    return padding.left + ((step - minStep) / (maxStep - minStep)) * innerWidth;
  }

  function yToPixel(loss: number): number {
    return padding.top + (1 - (loss - yMin) / (yMax - yMin)) * innerHeight;
  }

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xToPixel(p[0]).toFixed(1)} ${yToPixel(p[1]).toFixed(1)}`)
    .join(" ");

  const yTicks = [yMin + (yMax - yMin) * 0.25, yMin + (yMax - yMin) * 0.75];

  return (
    <svg width={width} height={height} role="img" aria-label="Courbe de loss d'entraînement">
      {yTicks.map((tick) => (
        <g key={tick}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={yToPixel(tick)}
            y2={yToPixel(tick)}
            stroke="currentColor"
            strokeOpacity={0.15}
          />
          <text x={4} y={yToPixel(tick) + 4} fontSize={10} fill="currentColor" opacity={0.7}>
            {tick.toFixed(3)}
          </text>
        </g>
      ))}

      <text x={padding.left} y={height - 6} fontSize={10} fill="currentColor" opacity={0.7}>
        step {minStep}
      </text>
      <text x={width - padding.right} y={height - 6} fontSize={10} fill="currentColor" opacity={0.7} textAnchor="end">
        step {maxStep}
      </text>

      <path d={pathD} fill="none" stroke="#4f46e5" strokeWidth={2} />
    </svg>
  );
}

export default LossChart;

import type { PillarKey, RiskScore } from "../../types/report";
import { maxSeverity } from "../../lib/severity";
import { isDrivingPillar } from "../../lib/risk";
import "./styles/AttributionChart.css";

const PILLAR_ORDER: { key: PillarKey; label: string }[] = [
  { key: "statistical", label: "Statistical" },
  { key: "steganographic", label: "Steganographic" },
  { key: "behavioral", label: "Behavioral" },
];

const STATUS_LABEL: Record<string, string> = {
  not_run: "Not assessed",
  ran_clean: "Clean",
  flagged: "Flagged",
};

// Fusion's own MEDIUM+ floor (peekaboo/pipeline/fusion.py:
// evidence_score(q) = 0.4 at q = 0.05). Drawn as a reference line so a
// bar's height reads against the threshold that actually drives risk_level,
// not just against an arbitrary 0-1 axis.
const MEDIUM_THRESHOLD = 0.4;
const TRACK_HEIGHT = 160;

interface AttributionChartProps {
  riskScore: RiskScore;
  onSelectPillar?: (key: PillarKey) => void;
}

// Grouped bars, not stacked: fusion's overall_score is a MAX over pillars,
// not a sum, so a stacked chart would visually imply contributions add up
// to the total, which is false (see PHASE5.md). The pillar equal to
// overall_score is the one "driving" the verdict and gets a ring, not a
// different hue -- this is an *emphasis* chart (one series is the point,
// the rest are context), not a categorical one.
const AttributionChart = ({ riskScore, onSelectPillar }: AttributionChartProps) => {
  return (
    <div className="attribution-chart">
      <div className="attribution-chart-plot">
        <div
          className="attribution-chart-threshold"
          style={{ bottom: `${MEDIUM_THRESHOLD * TRACK_HEIGHT}px` }}
        >
          <span className="attribution-chart-threshold-label">MEDIUM+ ≥ 0.40</span>
        </div>

        {PILLAR_ORDER.map(({ key, label }) => {
          const pillar = riskScore.pillars[key];
          const isNotRun = pillar.status === "not_run";
          const pct = pillar.score === null ? 0 : Math.max(0, Math.min(1, pillar.score));
          const barPx = isNotRun ? 10 : Math.max(2, pct * TRACK_HEIGHT);
          const driving = isDrivingPillar(pillar, riskScore.overall_score);
          const scoreText = pillar.score === null ? "N/A" : pillar.score.toFixed(2);
          let barClass = isNotRun ? "is-not-run" : "is-clean";
          if (!isNotRun && pillar.status === "flagged") {
            const worst = maxSeverity(pillar.flags);
            barClass = `is-flag-${worst === "critical" ? "critical" : worst === "high" ? "high" : "medium"}`;
          }
          const tooltipId = `attribution-tip-${key}`;

          return (
            <button
              key={key}
              type="button"
              className={`attribution-chart-col${driving ? " is-driving" : ""}`}
              style={{ height: `${TRACK_HEIGHT}px` }}
              onClick={() => onSelectPillar?.(key)}
              aria-describedby={tooltipId}
            >
              <span
                className={`attribution-chart-value${isNotRun ? " is-not-run" : ""}`}
                style={{ bottom: `${barPx + 6}px` }}
              >
                {scoreText}
              </span>
              <span className={`attribution-chart-bar ${barClass}`} style={{ height: `${barPx}px` }} />
              <span className="attribution-chart-tooltip" role="tooltip" id={tooltipId}>
                <strong>{label}</strong>
                <span>
                  {STATUS_LABEL[pillar.status]}
                  {pillar.weight === 0 ? " · report-only" : ""}
                </span>
                <span>{pillar.summary}</span>
              </span>
            </button>
          );
        })}
      </div>

      <div className="attribution-chart-labels">
        {PILLAR_ORDER.map(({ key, label }) => (
          <span key={key} className="attribution-chart-label">
            {label}
          </span>
        ))}
      </div>

      <p className="attribution-chart-caption">{riskScore.explanation.text}</p>
    </div>
  );
};

export default AttributionChart;

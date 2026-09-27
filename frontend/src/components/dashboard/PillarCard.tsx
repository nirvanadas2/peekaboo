import type { PillarScore } from "../../types/report";
import { maxSeverity } from "../../lib/severity";
import "./styles/PillarCard.css";

const STATUS_LABEL: Record<PillarScore["status"], string> = {
  not_run: "Not assessed",
  ran_clean: "Clean",
  flagged: "Flagged",
};

interface PillarCardProps {
  label: string;
  pillar: PillarScore;
  /** True when this pillar's score equals the overall risk score -- it's
   * the one driving the current verdict. */
  driving?: boolean;
}

const PillarCard = ({ label, pillar, driving = false }: PillarCardProps) => {
  const statusClass =
    pillar.status === "flagged" ? `st-flag-${maxSeverity(pillar.flags)}` : `st-${pillar.status}`;

  return (
    <div className={`pillar-card${driving ? " is-driving" : ""} ${statusClass}`}>
      <div className="pillar-card-head">
        <h3 className="pillar-card-title">{label}</h3>
        <span className="pillar-card-status">{STATUS_LABEL[pillar.status]}</span>
      </div>

      <div className="pillar-card-score">
        {pillar.score === null ? (
          <span className="pillar-card-score-value pillar-card-score-value--na">—</span>
        ) : (
          <span className="pillar-card-score-value">{pillar.score.toFixed(2)}</span>
        )}
        {pillar.weight === 0 && <span className="pillar-card-report-only">report-only</span>}
      </div>

      <p className="pillar-card-summary">{pillar.summary}</p>

      {pillar.flags.length > 0 && (
        <p className="pillar-card-flag-count">
          {pillar.flags.length} finding{pillar.flags.length === 1 ? "" : "s"}
        </p>
      )}

      {driving && <p className="pillar-card-driving-note">Drives the current verdict</p>}
    </div>
  );
};

export default PillarCard;

import type { BehavioralReport } from "../../types/report";
import "./styles/BehavioralFindingsList.css";

// Mirrors peekaboo/report.py::_evidence -- the same structured-detail
// keys, rendered the same way, so the dashboard's evidence line reads
// like the Markdown/CLI report's.
function evidenceLine(details: Record<string, unknown>): string {
  const parts: string[] = [];
  if (typeof details.fdr_p_value === "number") parts.push(`FDR q=${details.fdr_p_value.toExponential(2)}`);
  if (typeof details.p_value === "number") parts.push(`raw p=${details.p_value.toExponential(2)}`);
  if (Array.isArray(details.reach)) parts.push(`class reach [${details.reach.join(", ")}]`);
  if (
    typeof details.hit_rate === "number" &&
    typeof details.forced_class === "number" &&
    Array.isArray(details.position)
  ) {
    parts.push(
      `${(details.hit_rate * 100).toFixed(0)}% of inputs → class ${details.forced_class} at position [${details.position.join(", ")}]`
    );
  }
  return parts.join(" · ");
}

const MODE_LABEL: Record<BehavioralReport["mode"], string> = {
  probed: "Probed",
  not_runnable: "Not runnable",
};

interface BehavioralFindingsListProps {
  behavioral: BehavioralReport;
}

const BehavioralFindingsList = ({ behavioral }: BehavioralFindingsListProps) => {
  return (
    <div className="behavioral-findings">
      <div className="behavioral-findings-head">
        <h3 className="behavioral-findings-title">Behavioral probing</h3>
        <span className={`behavioral-findings-mode mode-${behavioral.mode}`}>{MODE_LABEL[behavioral.mode]}</span>
      </div>

      <ul className="behavioral-findings-list">
        {behavioral.findings.map((f, i) => {
          const evidence = evidenceLine(f.details);
          return (
            <li key={`${f.check}-${i}`} className={`behavioral-finding sev-${f.severity}`}>
              <div className="behavioral-finding-head">
                <span className={`behavioral-finding-badge sev-${f.severity}`}>{f.severity.toUpperCase()}</span>
                <code className="behavioral-finding-check">{f.check}</code>
              </div>
              <p className="behavioral-finding-message">{f.message}</p>
              {evidence && <p className="behavioral-finding-evidence">{evidence}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default BehavioralFindingsList;

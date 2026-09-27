import type { PeekabooReport, RiskLevel } from "../../types/report";
import "./styles/VerdictBanner.css";

// Mirrors peekaboo/report.py::_VERDICT. Kept in sync by hand since the
// phrase itself isn't part of the JSON (only the risk_level string is).
const VERDICT_COPY: Record<RiskLevel, string> = {
  info: "No evidence of tampering from the checks that ran",
  low: "No evidence of tampering from the checks that ran",
  medium: "SUSPICIOUS — review before deployment",
  high: "HIGH RISK — do not deploy without investigation",
  critical: "HIGH RISK — do not deploy without investigation",
};

interface VerdictBannerProps {
  report: PeekabooReport;
}

const VerdictBanner = ({ report }: VerdictBannerProps) => {
  const { scan } = report;

  if (scan.stopped_at_metadata || !scan.risk_score) {
    const criticalFindings = scan.metadata.findings.filter(
      (f) => !f.passed && f.severity === "critical"
    );
    return (
      <div className="verdict-banner verdict-banner--unsafe">
        <div className="verdict-banner-head">
          <span className="verdict-banner-tag">UNSAFE / NOT ANALYZED</span>
        </div>
        <p className="verdict-banner-copy">
          Stage 1 (metadata integrity) hard-failed, so the file was not loaded and no further
          checks ran.
        </p>
        {criticalFindings.map((f) => (
          <p className="verdict-banner-explanation" key={f.check}>
            <strong>{f.check}:</strong> {f.message}
          </p>
        ))}
        <p className="verdict-banner-meta">
          <code>{scan.metadata.model_path}</code>
        </p>
      </div>
    );
  }

  const rs = scan.risk_score;
  const level = rs.metadata.risk_level;
  const behavioralNotRun = rs.pillars.behavioral.status === "not_run";

  return (
    <div className={`verdict-banner verdict-banner--${level}`}>
      <div className="verdict-banner-head">
        <span className="verdict-banner-tag">
          {level.toUpperCase()}
          {behavioralNotRun && <span className="verdict-banner-caveat"> · BEHAVIOR NOT ASSESSED</span>}
        </span>
        <span className="verdict-banner-score">{rs.overall_score.toFixed(2)}</span>
      </div>
      <p className="verdict-banner-copy">{VERDICT_COPY[level]}</p>
      <p className="verdict-banner-explanation">{rs.explanation.text}</p>
      <p className="verdict-banner-meta">
        <code>{scan.metadata.model_path}</code> · {scan.metadata.detected_format} ·{" "}
        {scan.metadata.file_size.toLocaleString()} bytes
      </p>

      {behavioralNotRun && (
        <div className="verdict-banner-warning" role="alert">
          <span className="verdict-banner-warning-tag">Not assessed</span>
          <p>
            Behavioral probing did not run for this scan. No static check detects backdoors — on
            held-out data, backdoor detection AUC without a runnable model falls to 0.50 (chance).
            This report's {level.toUpperCase()} score does not mean the model is backdoor-free.
          </p>
        </div>
      )}
    </div>
  );
};

export default VerdictBanner;

import type { PeekabooReport, RiskLevel } from "../../types/report";
import type { ReportSource } from "../../hooks/useReport";
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

// The 0.50 AUC figure is measured only on the TinyCNN benchmark
// (PHASE6.md) -- quoting it for an arbitrary uploaded model would
// overclaim precision that doesn't transfer. Only a "demo" source is
// that exact measured benchmark; live scans and uploaded JSON reports
// (which could be from anywhere) get the qualitative claim only.
function notAssessedCopy(level: string, isMeasuredBenchmark: boolean): string {
  const evidence = isMeasuredBenchmark
    ? "on held-out data, backdoor detection AUC without a runnable model falls to 0.50 (chance)"
    : "this is a fundamental limitation of static analysis, not specific to this file";
  return `Behavioral probing did not run for this scan. No static check detects backdoors — ${evidence}. This report's ${level.toUpperCase()} score does not mean the model is backdoor-free.`;
}

interface VerdictBannerProps {
  report: PeekabooReport;
  source: ReportSource | null;
}

const VerdictBanner = ({ report, source }: VerdictBannerProps) => {
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
          <p>{notAssessedCopy(level, source?.kind === "demo")}</p>
        </div>
      )}
    </div>
  );
};

export default VerdictBanner;

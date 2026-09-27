import "./styles/DetectionResults.css";

interface ValidatedResult {
  label: string;
  detail: string;
  detected: string;
  falsePositives: string;
  note?: string;
  isLimitation?: boolean;
}

// Cumulative held-out results (PHASE6.md section 5, "Cumulative held-out
// record"). Fixed project-level facts about the system's validated track
// record across pre-registered test suites -- not derived from the
// currently-loaded report, and doesn't change per-scan.
const RESULTS: ValidatedResult[] = [
  {
    label: "Backdoors",
    detail: "Stage 5, with forward_fn",
    detected: "21 / 26",
    falsePositives: "0 / 78",
  },
  {
    label: "Noise injection",
    detail: "Stage 4",
    detected: "16 / 20",
    falsePositives: "0 / 80",
  },
  {
    label: "Stego payload",
    detail: "Stage 4, the benchmark's own 67-byte payload",
    detected: "0 / 20",
    falsePositives: "—",
    note: "Known limitation: only structured, dense payloads are detectable at all — see PHASE3.md.",
    isLimitation: true,
  },
  {
    label: "Clean models",
    detail: "Fused score",
    detected: "—",
    falsePositives: "0 / 10",
  },
];

const DetectionResults = () => {
  return (
    <div className="detection-results">
      <h3 className="detection-results-title">Detection results</h3>
      <p className="detection-results-lede">
        The system's cumulative, held-out track record across pre-registered test suites — the
        project's validated capability, not this scan's own findings (shown below once a report is
        loaded).
      </p>

      <div className="detection-results-table" role="table">
        <div className="detection-results-head" role="row">
          <span role="columnheader">Detector</span>
          <span role="columnheader">Detected</span>
          <span role="columnheader">False positives</span>
        </div>
        {RESULTS.map((row) => (
          <div
            key={row.label}
            role="row"
            className={`detection-results-row${row.isLimitation ? " is-limitation" : ""}`}
          >
            <span role="cell" className="detection-results-label">
              <span className="detection-results-label-main">{row.label}</span>
              <span className="detection-results-label-detail">{row.detail}</span>
            </span>
            <span role="cell" className="detection-results-value">
              <span className="detection-results-value-label">Detected: </span>
              {row.detected}
            </span>
            <span role="cell" className="detection-results-value">
              <span className="detection-results-value-label">False positives: </span>
              {row.falsePositives}
            </span>
          </div>
        ))}
      </div>

      {RESULTS.filter((r) => r.note).map((r) => (
        <p className="detection-results-note" key={r.label}>
          {r.note}
        </p>
      ))}

      <p className="detection-results-caveat">
        Held-out, pre-registered suites on one synthetic architecture (TinyCNN) and task — evidence
        the mechanisms work, not calibrated real-world rates. Sample sizes are small: at 95%
        confidence, 0/78 bounds the false-positive rate below roughly 4%, and 21/26 is compatible
        with recall from roughly 61% to 93% (PHASE6.md).
      </p>
    </div>
  );
};

export default DetectionResults;

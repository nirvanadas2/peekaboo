import "./styles/CoverageComparison.css";

interface CoverageRow {
  tool: string;
  fileSafety: boolean;
  weightContent: boolean;
  highlight?: boolean;
}

const ROWS: CoverageRow[] = [
  { tool: "picklescan / ModelScan", fileSafety: true, weightContent: false },
  { tool: "Peekaboo", fileSafety: true, weightContent: true, highlight: true },
];

const Cell = ({ label, ok }: { label: string; ok: boolean }) => (
  <span className={`coverage-cell ${ok ? "is-covered" : "is-gap"}`}>
    <span className="coverage-cell-label">{label}: </span>
    {ok ? "Checked" : "Not checked"}
  </span>
);

// Fixed project-level comparison (mirrors README.md's own framing: file
// safety vs. weight content). Originally shown on the landing page,
// removed there when the scroll-story was trimmed (b47a963); reused here
// since the dashboard is meant to stand alone as reference material.
const CoverageComparison = () => {
  return (
    <div className="coverage-comparison">
      <h3 className="coverage-comparison-title">The coverage gap</h3>
      <p className="coverage-comparison-lede">
        File-safety scanners and weight-content analysis check different things. Most pipelines
        only run the first.
      </p>
      <div className="coverage-table" role="table">
        <div className="coverage-table-head" role="row">
          <span role="columnheader">Tool</span>
          <span role="columnheader">File Safety</span>
          <span role="columnheader">Weight Content</span>
        </div>
        {ROWS.map((row) => (
          <div key={row.tool} role="row" className={`coverage-row${row.highlight ? " is-peekaboo" : ""}`}>
            <span role="cell" className="coverage-tool">
              {row.tool}
            </span>
            <span role="cell">
              <Cell label="File Safety" ok={row.fileSafety} />
            </span>
            <span role="cell">
              <Cell label="Weight Content" ok={row.weightContent} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CoverageComparison;

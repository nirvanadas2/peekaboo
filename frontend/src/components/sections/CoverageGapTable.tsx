import "./styles/CoverageGapTable.css";

const ROWS = [
  {
    tool: "picklescan / ModelScan",
    fileSafety: true,
    weightContent: false,
  },
  {
    tool: "Peekaboo",
    fileSafety: true,
    weightContent: true,
    highlight: true,
  },
];

const Cell = ({ label, ok }: { label: string; ok: boolean }) => (
  <span className={`coverage-cell ${ok ? "is-covered" : "is-gap"}`}>
    <span className="coverage-cell-label">{label}: </span>
    {ok ? "Checked" : "Not checked"}
  </span>
);

const CoverageGapTable = () => {
  return (
    <section className="pk-coverage section-shell" id="pk-coverage">
      <div className="section-container">
        <h2 className="title coverage-title">The coverage gap</h2>
        <p className="para coverage-lede">
          File-safety scanners and weight-content analysis check different
          things. Most pipelines only run the first.
        </p>
        <div className="coverage-table" role="table">
          <div className="coverage-table-head" role="row">
            <span role="columnheader">Tool</span>
            <span role="columnheader">File Safety</span>
            <span role="columnheader">Weight Content</span>
          </div>
          {ROWS.map((row) => (
            <div
              key={row.tool}
              role="row"
              className={`coverage-row ${row.highlight ? "is-peekaboo" : ""}`}
            >
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
    </section>
  );
};

export default CoverageGapTable;

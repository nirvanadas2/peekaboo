import "./styles/ArchitectureDiagram.css";

const STAGES = [
  "Model Upload",
  "Metadata & Parser",
  "Analyzer Bank",
  "Anomaly Fusion",
  "Explainability",
  "Risk Dashboard",
];

const ArchitectureDiagram = () => {
  return (
    <section className="pk-architecture section-shell" id="pk-architecture">
      <div className="section-container">
        <h2 className="title arch-title">How a model moves through Peekaboo</h2>
        <p className="para arch-lede">
          Six stages, each one gating the next. A model only reaches the
          dashboard after every analyzer has had its say.
        </p>
        <div className="arch-diagram">
          {STAGES.map((stage, index) => (
            <div className="arch-step" key={stage}>
              <div className="arch-node">
                <span className="arch-node-index">{String(index + 1).padStart(2, "0")}</span>
                <span className="arch-node-label">{stage}</span>
              </div>
              {index < STAGES.length - 1 && <div className="arch-edge" />}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ArchitectureDiagram;

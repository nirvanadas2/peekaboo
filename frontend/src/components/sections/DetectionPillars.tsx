import "./styles/DetectionPillars.css";

const PILLARS = [
  {
    id: "01",
    name: "Statistical Analysis",
    description:
      "Profiles weight distributions layer by layer, flagging entropy and variance patterns that don't match a cleanly trained model.",
  },
  {
    id: "02",
    name: "Bit-Level Steganalysis",
    description:
      "Inspects the low-order bits of float tensors for the structured noise that LSB and similar embedding techniques leave behind.",
  },
  {
    id: "03",
    name: "Behavioral Probing",
    description:
      "Runs targeted inputs through the model and watches for activation and output behavior consistent with a hidden trigger or payload.",
  },
];

const DetectionPillars = () => {
  return (
    <section className="pk-pillars section-shell" id="pk-pillars">
      <div className="section-container">
        <h2 className="title pillars-title">Three detection pillars</h2>
        <p className="para pillars-lede">
          No single check catches every hiding technique. Peekaboo runs all
          three and fuses the results.
        </p>
        <div className="pillars-grid">
          {PILLARS.map((pillar) => (
            <div className="pillar-card" key={pillar.id}>
              <span className="pillar-index">{pillar.id}</span>
              <h3 className="pillar-name">{pillar.name}</h3>
              <p className="pillar-description">{pillar.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default DetectionPillars;

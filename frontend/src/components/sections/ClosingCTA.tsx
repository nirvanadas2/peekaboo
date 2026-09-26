import { Link } from "react-router-dom";
import "./styles/ClosingCTA.css";

const ClosingCTA = () => {
  return (
    <section className="pk-cta section-shell" id="pk-cta">
      <div className="section-container cta-container">
        <h2 className="title cta-title">Stop trusting the weights blindly.</h2>
        <p className="para cta-copy">
          Run a model through the full pipeline and see exactly what it
          flags, and why.
        </p>
        <Link to="/dashboard" className="cta-button">
          Open the Scanner
        </Link>
      </div>
    </section>
  );
};

export default ClosingCTA;

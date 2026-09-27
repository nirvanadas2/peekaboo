import "./styles/LimitationsFooter.css";

interface LimitationsFooterProps {
  limitations: string[];
}

// Renders report.limitations verbatim -- this is peekaboo/report.py's own
// carefully-worded scope/limitations text (PHASE6.md's design principle:
// say what wasn't checked as loudly as what was found). Not paraphrased.
const LimitationsFooter = ({ limitations }: LimitationsFooterProps) => {
  return (
    <div className="limitations-footer">
      <h3 className="limitations-footer-title">Scope and limitations</h3>
      <ul className="limitations-footer-list">
        {limitations.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </div>
  );
};

export default LimitationsFooter;

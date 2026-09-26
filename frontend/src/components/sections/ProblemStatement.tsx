import "./styles/ProblemStatement.css";

const ProblemStatement = () => {
  return (
    <section className="pk-problem section-shell" id="pk-problem">
      <div className="section-container problem-grid">
        <div className="problem-text">
          <h2 className="title problem-title">
            Models are shared as trusted artifacts. That trust doesn't cover
            what's hidden in the weights.
          </h2>
          <p className="para problem-copy">
            Every day, checkpoints move through Hugging Face and the ONNX
            Model Zoo with no inspection of what's actually encoded in their
            parameters. A model can carry a payload &mdash; exfiltrated data,
            a trigger, an embedded message &mdash; smuggled inside the
            statistical noise of its own weights, and still load, run, and
            produce normal-looking outputs.
          </p>
          <p className="para problem-copy">
            Tools like <code>picklescan</code> and <code>ModelScan</code> exist
            for a reason: they catch unsafe deserialization, malicious pickle
            opcodes, and known exploit patterns in the file itself. That's{" "}
            <em>file safety</em>. It says nothing about whether the tensors
            inside a clean, safely-loadable file have been tampered with to
            hide something. That's the gap Peekaboo is built to close.
          </p>
        </div>
        <div className="problem-character-space" aria-hidden="true" />
      </div>
    </section>
  );
};

export default ProblemStatement;

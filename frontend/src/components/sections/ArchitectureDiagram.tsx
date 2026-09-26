import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import "./styles/ArchitectureDiagram.css";

// Mirrors peekaboo/pipeline/gate.py: Stage 1 is the only hard gate, the
// loader feeds Stage 2, Stages 3-5 run as parallel pillars, Stage 6 fuses
// them and Stage 7 renders the report + CI exit code.

type NodeId =
  | "input"
  | "s1"
  | "unsafe"
  | "loader"
  | "s2"
  | "s3"
  | "s4"
  | "fwd"
  | "s5"
  | "s6"
  | "s7"
  | "verdict"
  | "dash";

type NodeKind = "io" | "stage" | "fail" | "optional" | "planned";
type Status = "pass" | "flag" | "na" | "fail" | "miss";

interface NodeInfo {
  tag: string;
  title: string;
  short: string;
  sub: string;
  kind: NodeKind;
  summary: string;
  points: string[];
  source: string;
}

const NODES: Record<NodeId, NodeInfo> = {
  input: {
    tag: "INPUT",
    title: "Model file",
    short: "Model file",
    sub: "safetensors·pt·onnx",
    kind: "io",
    summary: "A single weight file comes in through the CLI or the Python API.",
    points: [
      "python -m peekaboo scan model.safetensors",
      "or run_pre_checks(path) from Python",
      "Formats: SafeTensors, PyTorch pickle (.pt / .pth), ONNX",
    ],
    source: "peekaboo/__main__.py",
  },
  s1: {
    tag: "STAGE 1",
    title: "Metadata gate",
    short: "Metadata gate",
    sub: "integrity · safety",
    kind: "stage",
    summary:
      "Checks the file itself before any tensor is trusted. This is the only stage that can stop the pipeline.",
    points: [
      "Exists, non-empty, SHA-256 fingerprint, extension matches content",
      "SafeTensors: header, offsets and extra keys validated",
      "Pickle: restricted unpickler, weights-only or refuse",
      "ONNX: parses, passes onnx.checker, sane opset",
    ],
    source: "peekaboo/pipeline/metadata_check.py",
  },
  unsafe: {
    tag: "EXIT 3",
    title: "Unsafe file",
    short: "Unsafe",
    sub: "nothing loaded",
    kind: "fail",
    summary:
      "Hard fail. Nothing is loaded, no later stage runs, and no risk score is produced.",
    points: ["Verdict: UNSAFE / NOT ANALYZED", "The CLI exits with code 3"],
    source: "peekaboo/pipeline/gate.py",
  },
  loader: {
    tag: "LOADER",
    title: "Tensor loader",
    short: "Tensor loader",
    sub: "LoadedModel",
    kind: "io",
    summary:
      "Turns every supported format into one common representation, so each later stage is format-agnostic.",
    points: [
      "Layer names, shapes, dtypes and raw tensors",
      "Dispatches on the detected format",
      "Read from disk once, shared by every analyzer",
    ],
    source: "peekaboo/loaders/dispatch.py",
  },
  s2: {
    tag: "STAGE 2",
    title: "Structure",
    short: "Structure",
    sub: "shapes · dtypes",
    kind: "stage",
    summary:
      "Sanity-checks the architecture the tensors describe. Findings are labels only; they never stop the scan.",
    points: [
      "Degenerate shapes and dtype uniformity",
      "Consecutive layer dimensions chain correctly",
      "Weight/bias shapes agree, parameter-count profile, naming outliers",
      "Optional exact diff against a known ArchitectureSpec",
    ],
    source: "peekaboo/pipeline/structural_check.py",
  },
  s3: {
    tag: "STAGE 3",
    title: "Weight statistics",
    short: "Statistics",
    sub: "report-only",
    kind: "stage",
    summary:
      "Profiles each layer's weight distribution and looks for layers that don't fit the rest of the model.",
    points: [
      "Mean, std, skewness, kurtosis and entropy per layer",
      "Outliers relative to the model's own layers (robust z-scores)",
      "Report-only: shown in the report, weight 0 in the score, because it flagged clean models and caught no tampering",
    ],
    source: "peekaboo/pipeline/statistical_check.py",
  },
  s4: {
    tag: "STAGE 4",
    title: "Bit steganalysis",
    short: "Stego bits",
    sub: "low bits · BH-FDR",
    kind: "stage",
    summary:
      "Runs hypothesis tests on the lowest mantissa bits, where steganographic payloads hide.",
    points: [
      "Bit-balance χ², block homogeneity χ², Wald-Wolfowitz runs, autocorrelation at lags 1·2·4·8",
      "Benjamini-Hochberg FDR correction across every test in the report",
      "Catches structured, dense tampering. Encrypted payloads remain undetectable.",
    ],
    source: "peekaboo/pipeline/stego_check.py",
  },
  fwd: {
    tag: "OPTIONAL",
    title: "forward_fn",
    short: "forward_fn",
    sub: "runnable model",
    kind: "optional",
    summary:
      "Weight files carry no computation graph, so behavior can only be probed if the caller supplies a way to run the model.",
    points: [
      "A function: batch of inputs → logits",
      "Plus input_shape and num_classes",
      "--tinycnn supplies one for the bundled benchmark only",
    ],
    source: "peekaboo/benchmark/runnable.py",
  },
  s5: {
    tag: "STAGE 5",
    title: "Behavioral probe",
    short: "Behavior",
    sub: "island + asymmetry",
    kind: "stage",
    summary:
      "Runs the model on patched probe inputs, looking for the signature of a backdoor: a small patch that forces one class.",
    points: [
      "Island test: a patch location that flips predictions when its neighbours don't",
      "Class-reach asymmetry: one class is reachable far more easily than the others",
      "Names the suspected target class",
      "No forward_fn means \"not assessed\", never \"clean\"",
    ],
    source: "peekaboo/pipeline/behavioral_probe.py",
  },
  s6: {
    tag: "STAGE 6",
    title: "Anomaly fusion",
    short: "Anomaly fusion",
    sub: "max of pillars",
    kind: "stage",
    summary:
      "Combines the pillars into one Model Risk Score. There's no learned model here, only evidence fusion.",
    points: [
      "Each pillar's score comes from its strongest finding's q-value: 0.4 at q = 0.05, up to 1.0 at q ≤ 1e-10",
      "Overall score = max over the scored pillars that ran (stego, behavior)",
      "A pillar that didn't run is excluded, not counted as clean",
      "Exact attribution: the driving pillar and its top findings are named",
    ],
    source: "peekaboo/pipeline/fusion.py",
  },
  s7: {
    tag: "STAGE 7",
    title: "Risk report",
    short: "Risk report",
    sub: "Markdown · JSON",
    kind: "stage",
    summary: "Explains the verdict in plain language, with evidence per layer.",
    points: [
      "What was checked, per stage, with status and score",
      "The findings that drive the score, per layer",
      "Documented limitations included in every report",
      "--md report.md and/or --json report.json",
    ],
    source: "peekaboo/report.py",
  },
  verdict: {
    tag: "CI GATE",
    title: "Verdict",
    short: "Verdict",
    sub: "exit 0 · 1 · 2 · 3",
    kind: "io",
    summary:
      "A CI-friendly exit code, so a pipeline can block a deployment automatically.",
    points: [
      "0: no scored evidence (info / low)",
      "1: MEDIUM",
      "2: HIGH / CRITICAL",
      "3: unsafe file, not analyzed",
    ],
    source: "peekaboo/report.py · exit_code()",
  },
  dash: {
    tag: "PLANNED",
    title: "Risk dashboard",
    short: "Dashboard",
    sub: "/dashboard",
    kind: "planned",
    summary: "The /dashboard page will visualize scan results. For now it's a placeholder.",
    points: ["Planned: a per-layer risk view built from the JSON report"],
    source: "frontend/src/pages/DashboardPlaceholder.tsx",
  },
};

interface Edge {
  from: NodeId;
  to: NodeId;
  label?: string;
  kind?: "fail" | "optional" | "planned";
  toFrac?: number;
}

const EDGES: Edge[] = [
  { from: "input", to: "s1" },
  { from: "s1", to: "unsafe", label: "hard fail", kind: "fail" },
  { from: "s1", to: "loader" },
  { from: "loader", to: "s2" },
  { from: "s2", to: "s3" },
  { from: "s2", to: "s4" },
  { from: "s2", to: "s5", toFrac: 0.3 },
  { from: "fwd", to: "s5", kind: "optional", toFrac: 0.72 },
  { from: "s3", to: "s6" },
  { from: "s4", to: "s6" },
  { from: "s5", to: "s6" },
  { from: "s6", to: "s7" },
  { from: "s7", to: "verdict" },
  { from: "s7", to: "dash", kind: "planned" },
];

interface Rect {
  x: number;
  y: number;
  w: number;
}

interface Layout {
  width: number;
  height: number;
  nodeH: number;
  rects: Record<NodeId, Rect>;
}

const WIDE: Layout = {
  width: 1280,
  height: 570,
  nodeH: 86,
  rects: {
    input: { x: 10, y: 227, w: 150 },
    s1: { x: 190, y: 227, w: 150 },
    unsafe: { x: 190, y: 400, w: 150 },
    loader: { x: 370, y: 227, w: 150 },
    s2: { x: 550, y: 227, w: 150 },
    s3: { x: 740, y: 70, w: 160 },
    s4: { x: 740, y: 227, w: 160 },
    s5: { x: 740, y: 384, w: 160 },
    fwd: { x: 550, y: 470, w: 150 },
    s6: { x: 940, y: 227, w: 150 },
    s7: { x: 1120, y: 227, w: 150 },
    verdict: { x: 1120, y: 400, w: 150 },
    dash: { x: 1120, y: 70, w: 150 },
  },
};

const TALL: Layout = {
  width: 360,
  height: 1030,
  nodeH: 80,
  rects: {
    input: { x: 80, y: 10, w: 200 },
    s1: { x: 10, y: 140, w: 200 },
    unsafe: { x: 240, y: 140, w: 110 },
    loader: { x: 80, y: 270, w: 200 },
    s2: { x: 10, y: 400, w: 200 },
    fwd: { x: 240, y: 400, w: 110 },
    s3: { x: 10, y: 540, w: 108 },
    s4: { x: 126, y: 540, w: 108 },
    s5: { x: 242, y: 540, w: 108 },
    s6: { x: 80, y: 680, w: 200 },
    s7: { x: 80, y: 810, w: 200 },
    verdict: { x: 10, y: 940, w: 200 },
    dash: { x: 240, y: 940, w: 110 },
  },
};

interface Scenario {
  id: string;
  label: string;
  steps: NodeId[][];
  status: Partial<Record<NodeId, Status>>;
  verdict: string;
  exit: string;
  summary: string;
  notes: Partial<Record<NodeId, string>>;
}

const ANALYZE_TAIL: NodeId[][] = [["s6"], ["s7"], ["verdict"]];
const HEAD: NodeId[][] = [["input"], ["s1"], ["loader"], ["s2"]];

const SCENARIOS: Scenario[] = [
  {
    id: "clean",
    label: "Clean model",
    steps: [...HEAD, ["fwd", "s3", "s4", "s5"], ...ANALYZE_TAIL],
    status: { s1: "pass", s2: "pass", s3: "pass", s4: "pass", s5: "pass" },
    verdict: "LOW RISK",
    exit: "exit 0",
    summary:
      "Every pillar that ran came back clean. On the final held-out suite, 0 of 10 clean models were flagged.",
    notes: {
      s5: "Probed with a forward_fn. No trigger island was found.",
      s6: "No MEDIUM+ finding in a scored pillar, so the overall score is 0.00.",
    },
  },
  {
    id: "backdoor",
    label: "Backdoored",
    steps: [...HEAD, ["fwd", "s3", "s4", "s5"], ...ANALYZE_TAIL],
    status: { s1: "pass", s2: "pass", s3: "pass", s4: "pass", s5: "flag" },
    verdict: "HIGH RISK",
    exit: "exit 2",
    summary:
      "Stage 5 finds a small patch that forces one class, and fusion takes the max, so behavior alone drives the verdict. The bundled backdoored.safetensors scores 0.98.",
    notes: {
      s3: "Static statistics see nothing. The backdoor lives in behavior.",
      s4: "The low bits look normal. A backdoor isn't a bit pattern.",
      s5: "Trigger island found, with the target class named.",
      s6: "The behavioral pillar is the argmax, so it drives the overall score.",
    },
  },
  {
    id: "nofwd",
    label: "Backdoored, no forward_fn",
    steps: [...HEAD, ["s3", "s4", "s5"], ...ANALYZE_TAIL],
    status: { s1: "pass", s2: "pass", s3: "pass", s4: "pass", s5: "na" },
    verdict: "LOW · behavior NOT ASSESSED",
    exit: "exit 0",
    summary:
      "Without a runnable model, Stage 5 can't probe. The report says \"not assessed\", never \"clean\", but the static stages can't see a backdoor, so the exit code alone won't catch it.",
    notes: {
      s5: "mode = not_runnable: zero forward passes, one visible finding explaining why.",
      s6: "The behavioral pillar is NOT_RUN, so it's excluded from the max rather than scored 0.",
    },
  },
  {
    id: "noisy",
    label: "Noisy low bits",
    steps: [...HEAD, ["fwd", "s3", "s4", "s5"], ...ANALYZE_TAIL],
    status: { s1: "pass", s2: "pass", s3: "pass", s4: "flag", s5: "pass" },
    verdict: "MEDIUM–HIGH",
    exit: "exit 1 or 2",
    summary:
      "Tampered low mantissa bits fail the bit-level tests after FDR correction. Across held-out suites: 16 of 20 noisy models caught, 0 of 80 clean ones flagged.",
    notes: {
      s4: "The bit tests reject the null of independent, unbiased bits even after BH-FDR.",
      s6: "The stego pillar drives the score. Its strength follows the q-value.",
    },
  },
  {
    id: "stego",
    label: "Encrypted payload",
    steps: [...HEAD, ["fwd", "s3", "s4", "s5"], ...ANALYZE_TAIL],
    status: { s1: "pass", s2: "pass", s3: "pass", s4: "miss", s5: "pass" },
    verdict: "LOW (missed)",
    exit: "exit 0",
    summary:
      "Known gap: an encrypted payload in the low bits looks exactly like rounding noise, so no test fires at any density. Peekaboo documents this instead of claiming it.",
    notes: {
      s4: "Random-looking bits are statistically identical to natural rounding noise.",
    },
  },
  {
    id: "unsafe",
    label: "Malicious pickle",
    steps: [["input"], ["s1"], ["unsafe"]],
    status: { s1: "fail" },
    verdict: "UNSAFE / NOT ANALYZED",
    exit: "exit 3",
    summary:
      "A pickle that would run code on load is refused by the restricted unpickler. The pipeline stops before any tensor is loaded, and no risk score is computed.",
    notes: {
      s1: "pickle_weights_only_safe hard-fails.",
      unsafe: "Short-circuit: every later stage is skipped.",
    },
  },
];

const BADGE: Record<Status, string> = {
  pass: "OK",
  flag: "FLAG",
  na: "N/A",
  fail: "FAIL",
  miss: "MISS",
};

const STEP_MS = 520;

function edgePath(layout: Layout, edge: Edge) {
  const a = layout.rects[edge.from];
  const b = layout.rects[edge.to];
  const h = layout.nodeH;
  const frac = edge.toFrac ?? 0.5;
  if (b.x >= a.x + a.w - 1) {
    const x1 = a.x + a.w;
    const y1 = a.y + h / 2;
    const x2 = b.x;
    const y2 = b.y + h * frac;
    const mx = (x1 + x2) / 2;
    return { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, mid: [mx, (y1 + y2) / 2] };
  }
  const down = b.y >= a.y + h;
  const x1 = a.x + a.w / 2;
  const y1 = down ? a.y + h : a.y;
  const x2 = b.x + b.w / 2;
  const y2 = down ? b.y : b.y + h;
  const my = (y1 + y2) / 2;
  return { d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`, mid: [(x1 + x2) / 2, my] };
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches
  );
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

const ArchitectureDiagram = () => {
  const isNarrow = useMediaQuery("(max-width: 760px)");
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const layout = isNarrow ? TALL : WIDE;

  const [selected, setSelected] = useState<NodeId>("input");
  const [hovered, setHovered] = useState<NodeId | null>(null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [step, setStep] = useState(-1);
  const [runKey, setRunKey] = useState(0);

  const scenario = SCENARIOS.find((s) => s.id === scenarioId) ?? null;

  useEffect(() => {
    if (!scenario || reducedMotion) return;
    const last = scenario.steps.length - 1;
    let current = 0;
    const timer = setInterval(() => {
      current += 1;
      setStep(current);
      if (current >= last) clearInterval(timer);
    }, STEP_MS);
    return () => clearInterval(timer);
  }, [scenario, runKey, reducedMotion]);

  const pathNodes = useMemo(() => new Set(scenario?.steps.flat() ?? []), [scenario]);
  const litNodes = useMemo(
    () => new Set(scenario?.steps.slice(0, step + 1).flat() ?? []),
    [scenario, step]
  );
  const done = scenario !== null && step >= scenario.steps.length - 1;
  const focus = hovered ?? selected;

  const runScenario = (id: string) => {
    const next = SCENARIOS.find((s) => s.id === id);
    setScenarioId(id);
    setStep(reducedMotion && next ? next.steps.length - 1 : 0);
    setRunKey((k) => k + 1);
  };

  const clearScenario = () => {
    setScenarioId(null);
    setStep(-1);
  };

  const onNodeKey = (event: KeyboardEvent, id: NodeId) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setSelected(id);
    }
  };

  const nodeState = (id: NodeId) => {
    if (!scenario) return "";
    if (litNodes.has(id)) return "is-lit";
    if (pathNodes.has(id)) return "is-pending";
    return "is-off";
  };

  const info = NODES[selected];
  const scenarioNote = scenario?.notes[selected];
  const selectedStatus = scenario && litNodes.has(selected) ? scenario.status[selected] : undefined;

  return (
    <section className="pk-architecture section-shell" id="pk-architecture">
      <div className="section-container">
        <h2 className="title arch-title">How a model moves through Peekaboo</h2>
        <p className="para arch-lede">
          Seven stages, one hard gate, three independent detection pillars.
          Click any stage to see what it checks, or trace a model through the
          pipeline.
        </p>

        <div className="arch-controls" role="group" aria-label="Trace a scenario">
          <span className="arch-controls-label">Trace a scenario</span>
          <div className="arch-scenarios">
            {SCENARIOS.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`arch-chip${s.id === scenarioId ? " is-active" : ""}`}
                aria-pressed={s.id === scenarioId}
                onClick={() => runScenario(s.id)}
              >
                {s.label}
              </button>
            ))}
            {scenario && (
              <button type="button" className="arch-chip arch-chip-ghost" onClick={clearScenario}>
                Reset
              </button>
            )}
          </div>
        </div>

        <div className="arch-canvas">
          <svg
            className="arch-svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            role="img"
            aria-label="Peekaboo pipeline architecture"
          >
            <g className="arch-edges">
              {EDGES.map((edge) => {
                const { d, mid } = edgePath(layout, edge);
                const related = edge.from === focus || edge.to === focus;
                const lit = scenario !== null && litNodes.has(edge.from) && litNodes.has(edge.to);
                const off =
                  scenario !== null && !(pathNodes.has(edge.from) && pathNodes.has(edge.to));
                const cls = [
                  "arch-edge-group",
                  edge.kind ? `is-${edge.kind}` : "",
                  related ? "is-related" : "",
                  lit ? "is-lit" : "",
                  off ? "is-off" : "",
                ].join(" ");
                return (
                  <g className={cls} key={`${edge.from}-${edge.to}`}>
                    <path className="arch-edge" d={d} pathLength={1} />
                    {lit && !reducedMotion && <path className="arch-flow" d={d} />}
                    {edge.label && !isNarrow && (
                      <text className="arch-edge-label" x={mid[0] + 8} y={mid[1] + 4}>
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>

            {(Object.keys(NODES) as NodeId[]).map((id) => {
              const node = NODES[id];
              const rect = layout.rects[id];
              const narrow = rect.w < 150;
              const status = scenario && litNodes.has(id) ? scenario.status[id] : undefined;
              const cls = [
                "arch-box",
                `is-${node.kind}`,
                nodeState(id),
                id === selected ? "is-selected" : "",
                id === hovered ? "is-hovered" : "",
                status ? `st-${status}` : "",
              ].join(" ");
              return (
                <g className="arch-node" key={id}>
                  <g
                    className={cls}
                    transform={`translate(${rect.x},${rect.y})`}
                    role="button"
                    tabIndex={0}
                    aria-label={`${node.tag}: ${node.title}`}
                    aria-pressed={id === selected}
                    onClick={() => setSelected(id)}
                    onKeyDown={(e) => onNodeKey(e, id)}
                    onMouseEnter={() => setHovered(id)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <rect className="arch-box-bg" width={rect.w} height={layout.nodeH} rx={10} />
                    <text className="arch-box-tag" x={12} y={24}>
                      {node.tag}
                    </text>
                    <text className="arch-box-title" x={12} y={narrow ? 52 : 50}>
                      {narrow ? node.short : node.title}
                    </text>
                    {!narrow && (
                      <text className="arch-box-sub" x={12} y={70}>
                        {node.sub}
                      </text>
                    )}
                    {status && (
                      <g
                        className="arch-badge"
                        transform={`translate(${rect.w - 46},${narrow ? layout.nodeH - 26 : 10})`}
                      >
                        <rect width={38} height={18} rx={9} />
                        <text x={19} y={13}>
                          {BADGE[status]}
                        </text>
                      </g>
                    )}
                  </g>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="arch-panels">
          <div className="arch-detail" aria-live="polite">
            <div className="arch-detail-head">
              <span className="arch-detail-tag">{info.tag}</span>
              {selectedStatus && (
                <span className={`arch-detail-status st-${selectedStatus}`}>
                  {BADGE[selectedStatus]}
                </span>
              )}
            </div>
            <h3 className="arch-detail-title">{info.title}</h3>
            <p className="arch-detail-summary">{info.summary}</p>
            <ul className="arch-detail-points">
              {info.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
            {scenarioNote && litNodes.has(selected) && (
              <p className="arch-detail-note">
                <span>In this scenario</span>
                {scenarioNote}
              </p>
            )}
            <code className="arch-detail-source">{info.source}</code>
          </div>

          <div className={`arch-result${scenario ? " is-active" : ""}`} aria-live="polite">
            {!scenario && (
              <>
                <span className="arch-result-label">Outcome</span>
                <p className="arch-result-hint">
                  Pick a scenario above to watch a model travel the pipeline,
                  stage by stage, to its verdict.
                </p>
                <ul className="arch-legend">
                  <li><i className="lg-solid" />data flow</li>
                  <li><i className="lg-dashed" />failure, optional or planned path</li>
                </ul>
              </>
            )}
            {scenario && (
              <>
                <span className="arch-result-label">{scenario.label}</span>
                {done ? (
                  <>
                    <p className="arch-result-verdict">{scenario.verdict}</p>
                    <p className="arch-result-exit">{scenario.exit}</p>
                    <p className="arch-result-summary">{scenario.summary}</p>
                  </>
                ) : (
                  <p className="arch-result-running">
                    Tracing… step {step + 1} of {scenario.steps.length}
                  </p>
                )}
                <button
                  type="button"
                  className="arch-chip arch-chip-ghost"
                  onClick={() => runScenario(scenario.id)}
                >
                  Replay
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ArchitectureDiagram;

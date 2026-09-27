// Shared pipeline-stage metadata and status vocabulary. Extracted from
// ArchitectureDiagram.tsx so the dashboard's PipelineStatusStrip can use
// the same stage tags/titles/sources and the same OK/FLAG/N-A/FAIL/MISS
// badge language as the landing page's architecture diagram, without the
// two staying in sync by hand.
//
// Mirrors peekaboo/pipeline/gate.py: Stage 1 is the only hard gate, the
// loader feeds Stage 2, Stages 3-5 run as parallel pillars, Stage 6 fuses
// them and Stage 7 renders the report + CI exit code.

export type NodeId =
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

export type NodeKind = "io" | "stage" | "fail" | "optional" | "planned";
export type Status = "pass" | "flag" | "na" | "fail" | "miss";

export interface NodeInfo {
  tag: string;
  title: string;
  short: string;
  sub: string;
  kind: NodeKind;
  summary: string;
  points: string[];
  source: string;
}

export const NODES: Record<NodeId, NodeInfo> = {
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
    tag: "LIVE",
    title: "Risk dashboard",
    short: "Dashboard",
    sub: "/dashboard",
    kind: "io",
    summary: "Upload a scan's JSON report (or pick a demo scenario) to see its verdict, pillar breakdown and per-layer findings.",
    points: [
      "Verdict, overall score and the pillar driving it",
      "Per-pillar detail: stego findings by layer, behavioral probes, per-layer risk",
      "Every report's own scope and limitations, shown, not hidden",
    ],
    source: "frontend/src/pages/Dashboard.tsx",
  },
};

export const BADGE: Record<Status, string> = {
  pass: "OK",
  flag: "FLAG",
  na: "N/A",
  fail: "FAIL",
  miss: "MISS",
};

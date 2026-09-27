// Mirrors the JSON produced by peekaboo/report.py::render_json, which is
// itself peekaboo.pipeline.gate.PreCheckResult.to_dict() wrapped with a
// verdict/limitations envelope. Field names and casing match the Python
// dataclasses' to_dict() output exactly (see peekaboo/schema/reports.py
// and peekaboo/schema/model_risk_score.py) -- no camelCasing.

export type Severity = "info" | "low" | "medium" | "high" | "critical";

export interface Finding {
  check: string;
  severity: Severity;
  passed: boolean;
  message: string;
  details: Record<string, unknown>;
}

export interface MetadataReport {
  model_path: string;
  declared_format: string;
  detected_format: string;
  file_size: number;
  file_hash: string;
  passed: boolean;
  hard_fail: boolean;
  findings: Finding[];
  metadata: Record<string, unknown>;
}

export interface StructuralReport {
  model_path: string;
  mode: "self_consistency" | "spec_diff";
  passed: boolean;
  findings: Finding[];
  metadata: Record<string, unknown>;
}

export interface StatisticalReport {
  model_path: string;
  mode: "relative_outlier" | "absolute_fallback";
  passed: boolean;
  findings: Finding[];
  metadata: {
    num_layers_analyzed?: number;
    weight_bearing_population?: string[];
    per_layer_stats?: Record<string, unknown>;
    [key: string]: unknown;
  };
}

export interface StegoReport {
  model_path: string;
  passed: boolean;
  max_severity: Severity;
  findings: Finding[];
  metadata: {
    n_bits?: number;
    layers_analyzed?: number;
    layers_skipped?: number;
    fdr_correction?: string;
    n_tests_corrected?: number;
    init_lattice_aware?: boolean;
    n_layers_lattice_aware?: number;
    [key: string]: unknown;
  };
}

export interface BehavioralReport {
  model_path: string;
  mode: "not_runnable" | "probed";
  passed: boolean;
  max_severity: Severity;
  findings: Finding[];
  metadata: {
    input_shape?: number[];
    num_classes?: number;
    n_carriers?: number;
    n_candidates_tested?: number;
    n_asymmetry_tests?: number;
    n_forward_batches?: number;
    patch_colors_sigma?: number[];
    fdr?: string;
    seed?: number;
    [key: string]: unknown;
  };
}

export type PillarStatus = "not_run" | "ran_clean" | "flagged";

// Findings/flags from different stages populate different keys of
// `details`; there is no discriminant field beyond `flag_type`/`check`,
// so this stays a loose union of the known keys rather than a tagged type.
export interface LayerFlagDetails {
  // Stego-check (Stage 4) bit-level tests
  p_value?: number;
  fdr_p_value?: number;
  // Behavioral (Stage 5) findings
  position?: [number, number];
  forced_class?: number;
  hit_rate?: number;
  reach?: number;
  color?: number;
  size?: number;
  n_neighbors?: number;
  // Statistical (Stage 3, report-only) findings
  modified_z_score?: number;
  [key: string]: unknown;
}

export interface LayerFlag {
  layer_name: string;
  flag_type: string;
  severity: Severity;
  score: number;
  message: string;
  details: LayerFlagDetails;
}

export interface PillarScore {
  name: string;
  status: PillarStatus;
  // null exactly when status is "not_run" (score/status are kept in sync
  // by PillarScore.__post_init__ on the Python side).
  score: number | null;
  weight: number;
  summary: string;
  flags: LayerFlag[];
}

export interface Explanation {
  text: string;
  feature_attributions: Record<string, number>;
}

export type RiskLevel = "info" | "low" | "medium" | "high" | "critical";

export interface RiskScoreMetadata {
  fusion: string;
  risk_level: RiskLevel;
  pillar_weights: Record<string, number>;
  report_only_pillars: string[];
  pillars_run: string[];
  pillars_not_run: string[];
  layer_risk: Record<string, number>;
}

export type PillarKey = "statistical" | "steganographic" | "behavioral";

export interface RiskScore {
  model_path: string;
  overall_score: number;
  pillars: Record<PillarKey, PillarScore>;
  layer_flags: LayerFlag[];
  explanation: Explanation;
  metadata: RiskScoreMetadata;
}

export interface ScanResult {
  stopped_at_metadata: boolean;
  metadata: MetadataReport;
  // All null together exactly when stopped_at_metadata is true (Stage 1
  // hard-failed before anything else ran).
  structural: StructuralReport | null;
  statistical: StatisticalReport | null;
  stego: StegoReport | null;
  behavioral: BehavioralReport | null;
  risk_score: RiskScore | null;
}

export type Verdict = RiskLevel | "unsafe_not_analyzed";

export interface PeekabooReport {
  report_version: string;
  verdict: Verdict;
  limitations: string[];
  scan: ScanResult;
}

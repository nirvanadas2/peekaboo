import { useMemo, useState } from "react";
import type { Finding, StegoReport, Severity } from "../../types/report";
import { maxSeverity } from "../../lib/severity";
import "./styles/StegoFindingsTable.css";

interface LayerGroup {
  layerName: string;
  findings: Finding[];
  maxSeverity: Severity;
  bestQ: number | null;
}

function qOf(f: Finding): number {
  const q = f.details.fdr_p_value;
  return typeof q === "number" ? q : Infinity;
}

function groupByLayer(findings: Finding[]): LayerGroup[] {
  const byLayer = new Map<string, Finding[]>();
  for (const f of findings) {
    const layerName = typeof f.details.layer_name === "string" ? f.details.layer_name : "(unknown layer)";
    const list = byLayer.get(layerName) ?? [];
    list.push(f);
    byLayer.set(layerName, list);
  }

  const groups: LayerGroup[] = [];
  for (const [layerName, layerFindings] of byLayer) {
    const qValues = layerFindings.map((f) => f.details.fdr_p_value).filter((q): q is number => typeof q === "number");
    groups.push({
      layerName,
      findings: [...layerFindings].sort((a, b) => qOf(a) - qOf(b)),
      maxSeverity: maxSeverity(layerFindings),
      bestQ: qValues.length ? Math.min(...qValues) : null,
    });
  }

  // Sorted by the FDR-corrected q, not raw p, so the correction is what
  // drives which layers surface first -- a layer's most significant
  // (lowest-q) finding determines its rank; layers with no q-bearing test
  // (e.g. "insufficient data") sort last.
  return groups.sort((a, b) => {
    if (a.bestQ === null && b.bestQ === null) return a.layerName.localeCompare(b.layerName);
    if (a.bestQ === null) return 1;
    if (b.bestQ === null) return -1;
    return a.bestQ - b.bestQ;
  });
}

interface StegoFindingsTableProps {
  stego: StegoReport;
}

const StegoFindingsTable = ({ stego }: StegoFindingsTableProps) => {
  const [onlyNotable, setOnlyNotable] = useState(true);
  const groups = useMemo(() => groupByLayer(stego.findings), [stego.findings]);
  const visibleGroups = onlyNotable ? groups.filter((g) => g.maxSeverity !== "info") : groups;

  return (
    <div className="stego-findings">
      <div className="stego-findings-head">
        <h3 className="stego-findings-title">Steganographic detection — bit-level tests</h3>
        <p className="stego-findings-meta">
          {stego.metadata.layers_analyzed ?? "?"} layers analyzed · {stego.metadata.n_tests_corrected ?? "?"} tests
          {stego.metadata.fdr_correction ? `, ${stego.metadata.fdr_correction.replace(/_/g, "-")}-corrected` : ""}.
          Grouped by layer, sorted by FDR-corrected q (most significant first).
        </p>
        <label className="stego-findings-filter">
          <input type="checkbox" checked={onlyNotable} onChange={(e) => setOnlyNotable(e.target.checked)} />
          Only show layers with a finding above info severity
        </label>
      </div>

      {visibleGroups.length === 0 ? (
        <p className="stego-findings-empty">
          No layers to show{onlyNotable ? " — every test on every layer came back info-level." : "."}
        </p>
      ) : (
        <div className="stego-findings-list">
          {visibleGroups.map((group) => (
            <details key={group.layerName} className="stego-layer-group">
              <summary className="stego-layer-summary">
                <code className="stego-layer-name">{group.layerName}</code>
                <span className="stego-layer-count">
                  {group.findings.length} test{group.findings.length === 1 ? "" : "s"}
                </span>
                <span className={`stego-layer-badge sev-${group.maxSeverity}`}>
                  {group.maxSeverity.toUpperCase()}
                </span>
                <span className="stego-layer-q">{group.bestQ === null ? "no q" : `q=${group.bestQ.toExponential(2)}`}</span>
              </summary>
              <table className="stego-layer-table">
                <thead>
                  <tr>
                    <th>Check</th>
                    <th>Severity</th>
                    <th>FDR q</th>
                    <th>Raw p</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {group.findings.map((f, i) => (
                    <tr key={`${f.check}-${i}`}>
                      <td>
                        <code>{f.check}</code>
                      </td>
                      <td>
                        <span className={`stego-layer-severity-label sev-${f.severity}`}>{f.severity}</span>
                      </td>
                      <td>{typeof f.details.fdr_p_value === "number" ? f.details.fdr_p_value.toExponential(2) : "—"}</td>
                      <td>{typeof f.details.p_value === "number" ? f.details.p_value.toExponential(2) : "—"}</td>
                      <td className="stego-layer-message">{f.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          ))}
        </div>
      )}
    </div>
  );
};

export default StegoFindingsTable;

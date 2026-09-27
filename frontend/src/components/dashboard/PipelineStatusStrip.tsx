import { BADGE, NODES, type NodeId, type Status } from "../../data/pipelineNodes";
import type { PeekabooReport } from "../../types/report";
import "./styles/PipelineStatusStrip.css";

const STRIP_STAGES: NodeId[] = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"];

function pillarStatus(status: "not_run" | "ran_clean" | "flagged"): Status {
  if (status === "not_run") return "na";
  if (status === "flagged") return "flag";
  return "pass";
}

// Real per-stage outcome for the loaded report, using the same
// pass/flag/na/fail vocabulary ArchitectureDiagram's canned scenarios use
// -- but derived from actual data, not a scripted trace.
function deriveStatuses(report: PeekabooReport): Partial<Record<NodeId, Status>> {
  const { scan } = report;
  if (scan.stopped_at_metadata) {
    return { s1: "fail" };
  }

  const statuses: Partial<Record<NodeId, Status>> = {
    s1: scan.metadata.passed ? "pass" : "flag",
    s2: scan.structural?.passed ? "pass" : "flag",
  };

  const rs = scan.risk_score;
  if (rs) {
    statuses.s3 = pillarStatus(rs.pillars.statistical.status);
    statuses.s4 = pillarStatus(rs.pillars.steganographic.status);
    statuses.s5 = pillarStatus(rs.pillars.behavioral.status);
    statuses.s6 = "pass";
    statuses.s7 = "pass";
  }

  return statuses;
}

interface PipelineStatusStripProps {
  report: PeekabooReport;
  onSelect?: (id: NodeId) => void;
}

const PipelineStatusStrip = ({ report, onSelect }: PipelineStatusStripProps) => {
  const statuses = deriveStatuses(report);
  const reached: Set<NodeId> = report.scan.stopped_at_metadata
    ? new Set(["s1"])
    : new Set(STRIP_STAGES);

  return (
    <div className="pipeline-strip" role="list" aria-label="Pipeline status for this scan">
      {STRIP_STAGES.map((id) => {
        const node = NODES[id];
        const status = statuses[id];
        const isReached = reached.has(id);
        return (
          <button
            key={id}
            type="button"
            className={`pipeline-strip-chip${status ? ` st-${status}` : ""}${
              isReached ? "" : " is-unreached"
            }`}
            onClick={() => onSelect?.(id)}
            disabled={!isReached}
          >
            <span className="pipeline-strip-tag">{node.tag}</span>
            <span className="pipeline-strip-title">{node.short}</span>
            {status && <span className="pipeline-strip-badge">{BADGE[status]}</span>}
          </button>
        );
      })}
    </div>
  );
};

export default PipelineStatusStrip;

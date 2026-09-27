import { useCallback, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { useReport } from "../hooks/useReport";
import ReportSourcePanel from "../components/dashboard/ReportSourcePanel";
import VerdictBanner from "../components/dashboard/VerdictBanner";
import PipelineStatusStrip from "../components/dashboard/PipelineStatusStrip";
import PillarCard from "../components/dashboard/PillarCard";
import AttributionChart from "../components/dashboard/AttributionChart";
import StegoFindingsTable from "../components/dashboard/StegoFindingsTable";
import BehavioralFindingsList from "../components/dashboard/BehavioralFindingsList";
import LayerRiskTable from "../components/dashboard/LayerRiskTable";
import LimitationsFooter from "../components/dashboard/LimitationsFooter";
import { isDrivingPillar } from "../lib/risk";
import type { NodeId } from "../data/pipelineNodes";
import type { PeekabooReport, PillarKey } from "../types/report";
import "./styles/Dashboard.css";

const PILLAR_LABELS: Record<PillarKey, string> = {
  statistical: "Statistical",
  steganographic: "Steganographic",
  behavioral: "Behavioral",
};

const STAGE_TO_PILLAR: Partial<Record<NodeId, PillarKey>> = {
  s3: "statistical",
  s4: "steganographic",
  s5: "behavioral",
};

const ReportView = ({ report }: { report: PeekabooReport }) => {
  const pillarRefs = useRef<Partial<Record<PillarKey, HTMLDivElement | null>>>({});

  const scrollToPillar = useCallback((key: PillarKey) => {
    pillarRefs.current[key]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const handleStageSelect = useCallback(
    (id: NodeId) => {
      const key = STAGE_TO_PILLAR[id];
      if (key) scrollToPillar(key);
    },
    [scrollToPillar]
  );

  const rs = report.scan.risk_score;

  return (
    <div className="dashboard-report">
      <VerdictBanner report={report} />
      <PipelineStatusStrip report={report} onSelect={handleStageSelect} />

      {rs && (
        <>
          <div className="dashboard-pillars">
            {(Object.keys(PILLAR_LABELS) as PillarKey[]).map((key) => (
              <div
                key={key}
                ref={(el) => {
                  pillarRefs.current[key] = el;
                }}
              >
                <PillarCard
                  label={PILLAR_LABELS[key]}
                  pillar={rs.pillars[key]}
                  driving={isDrivingPillar(rs.pillars[key], rs.overall_score)}
                />
              </div>
            ))}
          </div>

          <AttributionChart riskScore={rs} onSelectPillar={scrollToPillar} />

          {rs.pillars.statistical.flags.length > 0 && (
            <div className="dashboard-statistical">
              <h3 className="dashboard-statistical-title">Report-only observations (Stage 3)</h3>
              <p className="dashboard-statistical-note">
                Statistical findings are shown but do not affect the score — see the limitations below.
              </p>
              <ul className="dashboard-statistical-list">
                {rs.pillars.statistical.flags.map((f, i) => (
                  <li key={`${f.flag_type}-${i}`}>
                    <code>{f.layer_name}</code> — {f.message}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.scan.stego && <StegoFindingsTable stego={report.scan.stego} />}
          {report.scan.behavioral && <BehavioralFindingsList behavioral={report.scan.behavioral} />}
          <LayerRiskTable layerRisk={rs.metadata.layer_risk} />
        </>
      )}

      <LimitationsFooter limitations={report.limitations} />
    </div>
  );
};

const Dashboard = () => {
  const { report, source, loading, error, loadFromFile, loadDemo, clear } = useReport();

  // The landing page's scroll-story keeps body { overflow: hidden } (see
  // index.css) and switches it to auto itself once ready (initialFX.ts).
  // The dashboard needs ordinary page scroll instead -- take it over on
  // mount, restore whatever was there on unmount so navigating back to "/"
  // doesn't leave the landing page's own scroll setup broken.
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "auto";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <Link to="/" className="dashboard-back">
          ← Peekaboo
        </Link>
        <span className="dashboard-header-tag">Risk Dashboard</span>
      </header>

      <div className="dashboard-body">
        <ReportSourcePanel
          source={source}
          loading={loading}
          error={error}
          onFileSelected={loadFromFile}
          onDemoSelected={loadDemo}
          onClear={clear}
        />

        {!report && (
          <p className="dashboard-empty">
            Upload a scan report or pick a demo scenario above to see the results.
          </p>
        )}

        {report && <ReportView report={report} />}
      </div>
    </div>
  );
};

export default Dashboard;

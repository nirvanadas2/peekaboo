import "./styles/LayerRiskTable.css";

function band(score: number): "critical" | "high" | "medium" | "low" {
  if (score >= 0.8) return "critical";
  if (score >= 0.6) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

interface LayerRiskTableProps {
  layerRisk: Record<string, number>;
}

const LayerRiskTable = ({ layerRisk }: LayerRiskTableProps) => {
  const entries = Object.entries(layerRisk).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;

  return (
    <div className="layer-risk">
      <h3 className="layer-risk-title">Per-layer risk</h3>
      <table className="layer-risk-table">
        <tbody>
          {entries.map(([name, score]) => (
            <tr key={name}>
              <td className="layer-risk-name">
                <code>{name}</code>
              </td>
              <td className="layer-risk-meter-cell">
                <div className="layer-risk-meter">
                  <div
                    className={`layer-risk-meter-fill band-${band(score)}`}
                    style={{ width: `${Math.min(1, Math.max(0, score)) * 100}%` }}
                  />
                </div>
              </td>
              <td className="layer-risk-value">{score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default LayerRiskTable;

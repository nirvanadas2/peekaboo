import type { Finding, LayerFlag, Severity } from "../types/report";

// Mirrors peekaboo/schema/reports.py::compute_max_severity's ordering.
export const SEVERITY_ORDER: Severity[] = ["info", "low", "medium", "high", "critical"];

export function maxSeverity(items: Array<Finding | LayerFlag>): Severity {
  return items.reduce<Severity>(
    (worst, item) =>
      SEVERITY_ORDER.indexOf(item.severity) > SEVERITY_ORDER.indexOf(worst) ? item.severity : worst,
    "info"
  );
}

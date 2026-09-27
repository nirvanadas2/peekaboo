import { useCallback, useState } from "react";
import type { PeekabooReport } from "../types/report";

export interface DemoReport {
  id: string;
  label: string;
  file: string;
}

// Generated via `python -m peekaboo scan <fixture> --tinycnn --json <out>`
// against tests/fixtures/benchmark/*.safetensors, committed as static
// JSON under public/demo-reports/ so the dropdown works without a backend.
export const DEMO_REPORTS: DemoReport[] = [
  { id: "clean", label: "Clean model", file: "clean.json" },
  { id: "noisy", label: "Noisy low bits", file: "noisy.json" },
  { id: "steganographic", label: "Encrypted payload", file: "steganographic.json" },
  { id: "backdoored", label: "Backdoored", file: "backdoored.json" },
  { id: "combined", label: "Backdoored + noisy", file: "combined.json" },
  {
    id: "backdoored-no-forward-fn",
    label: "Backdoored, no forward_fn",
    file: "backdoored-no-forward-fn.json",
  },
];

export type ReportSource = { kind: "upload"; fileName: string } | { kind: "demo"; id: string };

function isPeekabooReport(value: unknown): value is PeekabooReport {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.report_version === "string" &&
    typeof v.verdict === "string" &&
    Array.isArray(v.limitations) &&
    typeof v.scan === "object" &&
    v.scan !== null
  );
}

interface UseReportState {
  report: PeekabooReport | null;
  source: ReportSource | null;
  loading: boolean;
  error: string | null;
}

const INITIAL_STATE: UseReportState = { report: null, source: null, loading: false, error: null };

export function useReport() {
  const [state, setState] = useState<UseReportState>(INITIAL_STATE);

  const loadFromFile = useCallback(async (file: File) => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const text = await file.text();
      const parsed: unknown = JSON.parse(text);
      if (!isPeekabooReport(parsed)) {
        throw new Error("This file doesn't look like a Peekaboo scan report.");
      }
      setState({ report: parsed, source: { kind: "upload", fileName: file.name }, loading: false, error: null });
    } catch (err) {
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : "Failed to read that file.",
      }));
    }
  }, []);

  const loadDemo = useCallback(async (id: string) => {
    const demo = DEMO_REPORTS.find((d) => d.id === id);
    if (!demo) return;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch(`/demo-reports/${demo.file}`);
      if (!res.ok) throw new Error(`Failed to load demo report (HTTP ${res.status}).`);
      const parsed: unknown = await res.json();
      if (!isPeekabooReport(parsed)) {
        throw new Error("Demo report failed validation.");
      }
      setState({ report: parsed, source: { kind: "demo", id }, loading: false, error: null });
    } catch (err) {
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : "Failed to load demo report.",
      }));
    }
  }, []);

  const clear = useCallback(() => setState(INITIAL_STATE), []);

  return { ...state, loadFromFile, loadDemo, clear };
}

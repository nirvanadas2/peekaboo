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

export type ReportSource =
  | { kind: "upload"; fileName: string }
  | { kind: "demo"; id: string }
  | { kind: "scan"; fileName: string };

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

  const loadFromScan = useCallback(async (file: File) => {
    setState((s) => ({ ...s, loading: true, error: null }));
    const formData = new FormData();
    formData.append("file", file);

    // peekaboo/api.py, proxied by vite.config.ts's /api -> :8000 rewrite
    // (PHASE7.md). Never sends a forward_fn -- an uploaded model's
    // behavioral pillar always comes back not_run; that's the honest
    // answer, not a bug in this call.
    let res: Response;
    try {
      res = await fetch("/api/scan", { method: "POST", body: formData });
    } catch {
      // fetch() itself throwing (not an HTTP error response) means the
      // server likely isn't running -- the realistic mistake in the
      // two-terminal dev workflow (PHASE7.md), so name it specifically
      // rather than a generic "failed to fetch".
      setState((s) => ({
        ...s,
        loading: false,
        error: "Couldn't reach the scan server. Is it running? (uvicorn peekaboo.api:app --reload)",
      }));
      return;
    }

    try {
      if (!res.ok) {
        // Vite's dev proxy answers with a plain-text 502 (not a thrown
        // fetch error, confirmed by testing) when it can't reach the
        // Python server at all -- the same "forgot to start uvicorn"
        // case the catch block above is watching for, just surfaced as
        // an HTTP status here instead of an exception.
        if (res.status === 502) {
          throw new Error("Couldn't reach the scan server. Is it running? (uvicorn peekaboo.api:app --reload)");
        }
        const body: unknown = await res.json().catch(() => null);
        const detail =
          body && typeof body === "object" && typeof (body as { detail?: unknown }).detail === "string"
            ? (body as { detail: string }).detail
            : `Scan failed (HTTP ${res.status}).`;
        throw new Error(detail);
      }
      const parsed: unknown = await res.json();
      if (!isPeekabooReport(parsed)) {
        throw new Error("The scan server returned something that doesn't look like a Peekaboo report.");
      }
      setState({ report: parsed, source: { kind: "scan", fileName: file.name }, loading: false, error: null });
    } catch (err) {
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : "Failed to scan that file.",
      }));
    }
  }, []);

  const clear = useCallback(() => setState(INITIAL_STATE), []);

  return { ...state, loadFromFile, loadDemo, loadFromScan, clear };
}

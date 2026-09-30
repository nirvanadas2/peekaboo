import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { DEMO_REPORTS, type ReportSource, type ScanArchitecture } from "../../hooks/useReport";
import "./styles/ReportSourcePanel.css";

const MODEL_EXTENSIONS = [".safetensors", ".pt", ".pth", ".onnx"];

const ARCHITECTURE_OPTIONS: { value: ScanArchitecture; label: string }[] = [
  { value: "unknown", label: "Unknown / generic" },
  { value: "tinycnn", label: "TinyCNN (peekaboo benchmark)" },
];

interface ReportSourcePanelProps {
  source: ReportSource | null;
  loading: boolean;
  error: string | null;
  onFileSelected: (file: File) => void;
  onScanSelected: (file: File, architecture: ScanArchitecture) => void;
  onDemoSelected: (id: string) => void;
  onClear: () => void;
}

const ReportSourcePanel = ({
  source,
  loading,
  error,
  onFileSelected,
  onScanSelected,
  onDemoSelected,
  onClear,
}: ReportSourcePanelProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [architecture, setArchitecture] = useState<ScanArchitecture>("unknown");
  // The model file behind the current scan, so changing the architecture
  // afterwards can re-scan it (see handleArchitectureChange).
  const lastScannedFileRef = useRef<File | null>(null);

  // A .json is a pre-generated report (existing path); a model file goes
  // to the live-scan API instead (peekaboo/api.py, PHASE7.md). Anything
  // else is rejected here rather than handed to onFileSelected, which
  // would otherwise report it with a misleading "doesn't look like a
  // Peekaboo report" (true, but not the actual problem).
  const handleSelectedFile = useCallback(
    (file: File) => {
      const lower = file.name.toLowerCase();
      if (lower.endsWith(".json")) {
        setLocalError(null);
        onFileSelected(file);
        return;
      }
      if (MODEL_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
        setLocalError(null);
        lastScannedFileRef.current = file;
        onScanSelected(file, architecture);
        return;
      }
      setLocalError(
        `"${file.name}" isn't a supported file. Drop a .json report, or a .safetensors/.pt/.pth/.onnx model to scan live.`
      );
    },
    [onFileSelected, onScanSelected, architecture]
  );

  const handleFileInput = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) handleSelectedFile(file);
      event.target.value = "";
    },
    [handleSelectedFile]
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files?.[0];
      if (file) handleSelectedFile(file);
    },
    [handleSelectedFile]
  );

  // The selector used to feed only the NEXT upload: picking TinyCNN after
  // a file was already scanned left the selector reading "TinyCNN" over a
  // report that was scanned as unknown (Stage 5 not assessed), with no
  // new request sent. Re-scan the same file so the report always matches
  // the architecture shown.
  const handleArchitectureChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const next = event.target.value as ScanArchitecture;
      setArchitecture(next);
      const file = lastScannedFileRef.current;
      if (source?.kind === "scan" && file && file.name === source.fileName && next !== source.architecture) {
        onScanSelected(file, next);
      }
    },
    [source, onScanSelected]
  );

  const handleDemoChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      if (event.target.value) onDemoSelected(event.target.value);
    },
    [onDemoSelected]
  );

  const currentDemoLabel =
    source?.kind === "demo" ? DEMO_REPORTS.find((d) => d.id === source.id)?.label ?? source.id : null;

  const displayError = error ?? localError;

  return (
    <div className="report-source">
      <div
        className={`report-source-drop${dragOver ? " is-dragover" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <span className="report-source-drop-label">
          {loading ? "Loading…" : "Drop a model to scan, or a .json report"}
        </span>
        <span className="report-source-drop-hint">.safetensors · .pt · .pth · .onnx · .json</span>
        <input
          ref={inputRef}
          type="file"
          accept=".json,application/json,.safetensors,.pt,.pth,.onnx"
          className="report-source-input"
          onChange={handleFileInput}
        />
      </div>

      <div className="report-source-demo">
        <label htmlFor="scan-architecture-select" className="report-source-demo-label">
          Architecture
        </label>
        <select
          id="scan-architecture-select"
          className="report-source-demo-select"
          value={architecture}
          onChange={handleArchitectureChange}
        >
          {ARCHITECTURE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <p className="report-source-scan-note">
        Live model scans can't test for backdoors (Stage 5) — that needs a forward function for the
        model's exact architecture, which Peekaboo can't infer from an uploaded file alone. The one
        exception is this repo's own TinyCNN: declare it above and its weights are loaded into the
        real class and probed. Static checks (steganography, weight statistics) always run in full.
      </p>

      <div className="report-source-demo">
        <label htmlFor="demo-report-select" className="report-source-demo-label">
          Or try a demo report
        </label>
        <select
          id="demo-report-select"
          className="report-source-demo-select"
          value={source?.kind === "demo" ? source.id : ""}
          onChange={handleDemoChange}
        >
          <option value="" disabled>
            Select a scenario…
          </option>
          {DEMO_REPORTS.map((demo) => (
            <option key={demo.id} value={demo.id}>
              {demo.label}
            </option>
          ))}
        </select>
      </div>

      {source && (
        <div className="report-source-current">
          <span>
            {source.kind === "upload"
              ? `Loaded: ${source.fileName}`
              : source.kind === "scan"
                ? `Scanned: ${source.fileName}${source.forwardFn === "tinycnn" ? " · probed as TinyCNN" : ""}`
                : `Demo: ${currentDemoLabel}`}
          </span>
          <button type="button" className="report-source-clear" onClick={onClear}>
            Clear
          </button>
        </div>
      )}

      {source?.kind === "scan" && source.forwardFn === "tinycnn-load-failed" && (
        <p className="report-source-error" role="alert">
          Declared TinyCNN, but this file's tensors don't load into TinyCNN — behavior was not
          assessed.
        </p>
      )}

      {displayError && (
        <p className="report-source-error" role="alert">
          {displayError}
        </p>
      )}
    </div>
  );
};

export default ReportSourcePanel;

import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { DEMO_REPORTS, type ReportSource } from "../../hooks/useReport";
import "./styles/ReportSourcePanel.css";

const MODEL_EXTENSIONS = [".safetensors", ".pt", ".pth", ".onnx"];

interface ReportSourcePanelProps {
  source: ReportSource | null;
  loading: boolean;
  error: string | null;
  onFileSelected: (file: File) => void;
  onScanSelected: (file: File) => void;
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
        onScanSelected(file);
        return;
      }
      setLocalError(
        `"${file.name}" isn't a supported file. Drop a .json report, or a .safetensors/.pt/.pth/.onnx model to scan live.`
      );
    },
    [onFileSelected, onScanSelected]
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

      <p className="report-source-scan-note">
        Live model scans can't test for backdoors (Stage 5) — that needs a caller-supplied forward
        function for the model's exact architecture, which Peekaboo can't infer from an uploaded
        file alone. Static checks (steganography, weight statistics) still run in full.
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
                ? `Scanned: ${source.fileName}`
                : `Demo: ${currentDemoLabel}`}
          </span>
          <button type="button" className="report-source-clear" onClick={onClear}>
            Clear
          </button>
        </div>
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

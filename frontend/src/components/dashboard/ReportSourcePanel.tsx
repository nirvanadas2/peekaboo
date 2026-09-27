import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { DEMO_REPORTS, type ReportSource } from "../../hooks/useReport";
import "./styles/ReportSourcePanel.css";

interface ReportSourcePanelProps {
  source: ReportSource | null;
  loading: boolean;
  error: string | null;
  onFileSelected: (file: File) => void;
  onDemoSelected: (id: string) => void;
  onClear: () => void;
}

const ReportSourcePanel = ({
  source,
  loading,
  error,
  onFileSelected,
  onDemoSelected,
  onClear,
}: ReportSourcePanelProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileInput = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) onFileSelected(file);
      event.target.value = "";
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleDemoChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      if (event.target.value) onDemoSelected(event.target.value);
    },
    [onDemoSelected]
  );

  const currentDemoLabel =
    source?.kind === "demo" ? DEMO_REPORTS.find((d) => d.id === source.id)?.label ?? source.id : null;

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
          {loading ? "Loading…" : "Drop a scan report, or click to choose a file"}
        </span>
        <span className="report-source-drop-hint">.json from `python -m peekaboo scan`</span>
        <input
          ref={inputRef}
          type="file"
          accept=".json,application/json"
          className="report-source-input"
          onChange={handleFileInput}
        />
      </div>

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
            {source.kind === "upload" ? `Loaded: ${source.fileName}` : `Demo: ${currentDemoLabel}`}
          </span>
          <button type="button" className="report-source-clear" onClick={onClear}>
            Clear
          </button>
        </div>
      )}

      {error && (
        <p className="report-source-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export default ReportSourcePanel;

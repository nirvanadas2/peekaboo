import { Link } from "react-router-dom";

const DashboardPlaceholder = () => {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "16px",
        textAlign: "center",
        padding: "0 20px",
        overflow: "auto",
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          color: "var(--accent)",
          fontSize: "13px",
          letterSpacing: "2px",
          textTransform: "uppercase",
        }}
      >
        Risk Dashboard
      </span>
      <h1 style={{ margin: 0, fontSize: "32px" }}>Dashboard coming soon</h1>
      <p style={{ color: "#a7abb5", maxWidth: "480px" }}>
        The scan results dashboard is being built. Check back soon to see
        Peekaboo's full pipeline output.
      </p>
      <Link
        to="/"
        style={{
          marginTop: "12px",
          color: "var(--accent)",
          fontFamily: "var(--font-mono)",
          fontSize: "14px",
        }}
      >
        ← Back to the landing page
      </Link>
    </div>
  );
};

export default DashboardPlaceholder;

function HealthStatus({ health, loading, error, onRefresh }) {
  const status = health?.status || "unknown";

  return (
    <section className="panel health-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Health</p>
          <h2>Backend Status</h2>
        </div>
        <button className="icon-button" type="button" onClick={onRefresh} disabled={loading} title="Refresh health">
          Refresh
        </button>
      </div>
      <div className={`status-row ${status === "ok" ? "status-ok" : "status-warn"}`}>
        <span className="status-dot" />
        <strong>{loading ? "Checking" : status}</strong>
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="metric-grid">
        <div>
          <span>Version</span>
          <strong>{health?.version || "-"}</strong>
        </div>
        <div>
          <span>Uptime</span>
          <strong>{health?.uptime_s ? `${health.uptime_s}s` : "-"}</strong>
        </div>
        <div>
          <span>Sessions</span>
          <strong>{health?.active_sessions ?? "-"}</strong>
        </div>
        <div>
          <span>Tasks</span>
          <strong>{health?.task_counts?.total ?? "-"}</strong>
        </div>
      </div>
    </section>
  );
}

export default HealthStatus;

const actions = [
  { type: "run_tests", label: "Run Tests" },
  { type: "get_hint", label: "Get Hint" },
  { type: "submit_fix", label: "Submit Fix" },
  { type: "view_code", label: "View Code" },
];

function ActionPanel({ sessionId, busy, observation, onAction }) {
  return (
    <section className="panel action-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Actions</p>
          <h2>Environment Controls</h2>
        </div>
      </div>
      <div className="action-grid">
        {actions.map((action) => (
          <button
            key={action.type}
            type="button"
            onClick={() => onAction(action.type)}
            disabled={!sessionId || busy || (observation?.done && action.type !== "get_hint")}
          >
            {action.label}
          </button>
        ))}
      </div>
      <div className="action-summary">
        <span>Hints used: {observation?.hints_used ?? 0}</span>
        <span>Remaining: {observation?.steps_remaining ?? "-"}</span>
        <span>{observation?.done ? "Done" : "Active"}</span>
      </div>
    </section>
  );
}

export default ActionPanel;

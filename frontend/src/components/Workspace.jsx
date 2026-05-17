function Workspace({
  task,
  observation,
  codeDraft,
  editLineNumber,
  editLineContent,
  seed,
  busy,
  sessionId,
  onCodeDraftChange,
  onEditLineNumberChange,
  onEditLineContentChange,
  onSeedChange,
  onStartSession,
  onLineAction,
}) {
  const code = observation?.current_code || task?.buggy_code || "";
  const lines = code.split("\n");
  const progress =
    observation?.tests_total > 0 ? `${observation.tests_passed}/${observation.tests_total}` : `0/${task?.num_tests || 0}`;

  return (
    <section className="workspace">
      <div className="panel detail-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Current Task</p>
            <h2>{task?.title || "Choose a task"}</h2>
          </div>
          <div className="badge-row">
            {task?.difficulty ? <span className="badge">{task.difficulty}</span> : null}
            {task?.bug_category ? <span className="badge muted">{task.bug_category}</span> : null}
          </div>
        </div>
        <p className="task-description">
          {task?.description || "Select a task from the library to inspect the prompt and start a session."}
        </p>
        <div className="task-stats">
          <span>Tests: {progress}</span>
          <span>Steps: {observation?.step_count ?? 0}/{observation?.max_steps ?? task?.max_steps ?? "-"}</span>
          <span>Score: {Number(observation?.score_so_far || 0).toFixed(2)}</span>
        </div>
        <div className="session-controls">
          <label>
            Seed
            <input
              type="number"
              value={seed}
              onChange={(event) => onSeedChange(event.target.value)}
              placeholder="optional"
            />
          </label>
          <button type="button" className="primary-button" onClick={onStartSession} disabled={!task || busy}>
            {sessionId ? "Reset Session" : "Start Session"}
          </button>
        </div>
      </div>

      <div className="panel code-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Code</p>
            <h2>Viewer and Editor</h2>
          </div>
          <span className="code-count">{lines.length} lines</span>
        </div>
        <textarea
          className="code-editor"
          value={codeDraft}
          onChange={(event) => onCodeDraftChange(event.target.value)}
          spellCheck="false"
          aria-label="Current code"
          placeholder="Start a session to load editable code."
        />
        <div className="line-editor">
          <label>
            Line
            <input
              type="number"
              min="1"
              value={editLineNumber}
              onChange={(event) => onEditLineNumberChange(event.target.value)}
            />
          </label>
          <label className="line-content-field">
            New content
            <input
              value={editLineContent}
              onChange={(event) => onEditLineContentChange(event.target.value)}
              placeholder="Exact replacement or inserted line"
            />
          </label>
          <div className="line-actions">
            <button type="button" onClick={() => onLineAction("edit_line")} disabled={!sessionId || busy}>
              Edit
            </button>
            <button type="button" onClick={() => onLineAction("insert_line")} disabled={!sessionId || busy}>
              Insert
            </button>
            <button type="button" onClick={() => onLineAction("delete_line")} disabled={!sessionId || busy}>
              Delete
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Workspace;

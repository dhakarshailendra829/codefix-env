const difficulties = [
  { value: "", label: "All" },
  { value: "easy", label: "Easy" },
  { value: "medium", label: "Medium" },
  { value: "hard", label: "Hard" },
];

function TaskList({
  tasks,
  selectedTaskId,
  difficulty,
  loading,
  error,
  onDifficultyChange,
  onSelectTask,
  onRefresh,
}) {
  return (
    <section className="panel task-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Tasks</p>
          <h2>Task Library</h2>
        </div>
        <button className="icon-button" type="button" onClick={onRefresh} disabled={loading} title="Refresh tasks">
          Reload
        </button>
      </div>

      <div className="segmented-control" aria-label="Filter tasks by difficulty">
        {difficulties.map((item) => (
          <button
            key={item.value}
            type="button"
            className={difficulty === item.value ? "active" : ""}
            onClick={() => onDifficultyChange(item.value)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      <div className="task-list" aria-busy={loading}>
        {tasks.map((task) => (
          <button
            key={task.id}
            type="button"
            className={`task-item ${selectedTaskId === task.id ? "selected" : ""}`}
            onClick={() => onSelectTask(task.id)}
          >
            <span className="task-title">{task.title}</span>
            <span className="task-meta">
              {task.difficulty} · {task.num_tests} tests
            </span>
          </button>
        ))}
        {!loading && tasks.length === 0 ? <p className="empty-state">No tasks found.</p> : null}
      </div>
    </section>
  );
}

export default TaskList;

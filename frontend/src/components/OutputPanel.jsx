function OutputPanel({ observation, lastResult, error, busy }) {
  const testResults = observation?.test_results || [];
  const diff = observation?.diff || "";
  const output = observation?.test_output || observation?.feedback || "No output yet.";

  return (
    <section className="panel output-panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">Results</p>
          <h2>Output Panel</h2>
        </div>
        {busy ? <span className="busy-pill">Working</span> : null}
      </div>

      {error ? <p className="error-text">{error}</p> : null}
      {observation?.feedback ? <p className="feedback-text">{observation.feedback}</p> : null}

      <div className="result-strip">
        <span>Reward: {Number(lastResult?.reward ?? observation?.shaped_reward ?? 0).toFixed(3)}</span>
        <span>Passed: {observation?.tests_passed ?? 0}/{observation?.tests_total ?? 0}</span>
        <span>Done: {observation?.done ? "yes" : "no"}</span>
      </div>

      <pre className="terminal-output">{output}</pre>

      {testResults.length > 0 ? (
        <div className="test-results">
          {testResults.map((result) => (
            <div key={result.name} className={`test-row ${result.passed ? "pass" : "fail"}`}>
              <strong>{result.name}</strong>
              <span>{result.passed ? "pass" : "fail"}</span>
            </div>
          ))}
        </div>
      ) : null}

      <details className="diff-block" open={Boolean(diff)}>
        <summary>Diff</summary>
        <pre>{diff || "No code changes yet."}</pre>
      </details>
    </section>
  );
}

export default OutputPanel;

import { getConfiguredApiBaseUrl } from "../api/codefixApi";

function Header({ sessionId }) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">CodeFix-Env</p>
        <h1>Debugging Workspace</h1>
      </div>
      <div className="header-meta" title={sessionId || "Start a session to receive an ID"}>
        <span>API: {getConfiguredApiBaseUrl()}</span>
        <strong>{sessionId ? `Session ${sessionId.slice(0, 8)}` : "No session"}</strong>
      </div>
    </header>
  );
}

export default Header;

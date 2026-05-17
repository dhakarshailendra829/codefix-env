import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getHealth,
  getTask,
  getTasks,
  resetSession,
  stepSession,
} from "./api/codefixApi";
import Header from "./components/Header";
import HealthStatus from "./components/HealthStatus";
import TaskList from "./components/TaskList";
import Workspace from "./components/Workspace";
import ActionPanel from "./components/ActionPanel";
import OutputPanel from "./components/OutputPanel";

function App() {
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState("");
  const [tasks, setTasks] = useState([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [tasksError, setTasksError] = useState("");
  const [difficulty, setDifficulty] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [selectedTask, setSelectedTask] = useState(null);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("codefix-session-id") || "");
  const [observation, setObservation] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [codeDraft, setCodeDraft] = useState("");
  const [editLineNumber, setEditLineNumber] = useState("1");
  const [editLineContent, setEditLineContent] = useState("");
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [workspaceError, setWorkspaceError] = useState("");

  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError("");
    try {
      setHealth(await getHealth());
    } catch (error) {
      setHealthError(error.message);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  const loadTasks = useCallback(async (nextDifficulty = "") => {
    setTasksLoading(true);
    setTasksError("");
    try {
      const data = await getTasks(nextDifficulty);
      setTasks(data.tasks || []);
    } catch (error) {
      setTasksError(error.message);
    } finally {
      setTasksLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    loadTasks("");
  }, [loadHealth, loadTasks]);

  useEffect(() => {
    const firstTask = tasks[0];
    if (!selectedTaskId && firstTask) {
      setSelectedTaskId(firstTask.id);
    }
  }, [tasks, selectedTaskId]);

  useEffect(() => {
    if (!selectedTaskId) return;

    let cancelled = false;
    setWorkspaceError("");
    getTask(selectedTaskId)
      .then((task) => {
        if (cancelled) return;
        setSelectedTask(task);
        if (!observation || observation.task_id !== task.id) {
          setCodeDraft(task.buggy_code || "");
        }
      })
      .catch((error) => {
        if (!cancelled) setWorkspaceError(error.message);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTaskId]);

  const currentObservation = useMemo(() => observation || null, [observation]);

  async function handleDifficultyChange(nextDifficulty) {
    setDifficulty(nextDifficulty);
    setSelectedTaskId("");
    setSelectedTask(null);
    setObservation(null);
    setLastResult(null);
    await loadTasks(nextDifficulty);
  }

  async function handleStartSession() {
    if (!selectedTask) return;

    setBusy(true);
    setWorkspaceError("");
    try {
      const { observation: nextObservation, sessionId: nextSessionId } = await resetSession({
        taskId: selectedTask.id,
        seed,
      });

      if (!nextSessionId) {
        throw new Error("The reset response did not expose X-Session-ID.");
      }

      setSessionId(nextSessionId);
      localStorage.setItem("codefix-session-id", nextSessionId);
      setObservation(nextObservation);
      setLastResult(null);
      setCodeDraft(nextObservation.current_code || "");
      setEditLineNumber("1");
      setEditLineContent("");
      await loadHealth();
    } catch (error) {
      setWorkspaceError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function runStep(action) {
    if (!sessionId) {
      setWorkspaceError("Start a session first.");
      return;
    }

    setBusy(true);
    setWorkspaceError("");
    try {
      const result = await stepSession(sessionId, action);
      setLastResult(result);
      setObservation(result.observation);
      setCodeDraft(result.observation?.current_code || codeDraft);
    } catch (error) {
      setWorkspaceError(error.message);
    } finally {
      setBusy(false);
    }
  }

  function handleEnvironmentAction(actionType) {
    runStep({ action_type: actionType });
  }

  function handleLineAction(actionType) {
    const line = Number(editLineNumber);
    if (!line || line < 1) {
      setWorkspaceError("Choose a valid 1-indexed line number.");
      return;
    }

    const action = {
      action_type: actionType,
      line_number: line,
    };

    if (actionType !== "delete_line") {
      action.new_content = editLineContent;
    }

    runStep(action);
  }

  function handleTaskSelect(taskId) {
    setSelectedTaskId(taskId);
    setObservation(null);
    setLastResult(null);
    setSessionId("");
    localStorage.removeItem("codefix-session-id");
  }

  return (
    <div className="app-shell">
      <Header sessionId={sessionId} />
      <main className="app-layout">
        <aside className="sidebar">
          <HealthStatus
            health={health}
            loading={healthLoading}
            error={healthError}
            onRefresh={loadHealth}
          />
          <TaskList
            tasks={tasks}
            selectedTaskId={selectedTaskId}
            difficulty={difficulty}
            loading={tasksLoading}
            error={tasksError}
            onDifficultyChange={handleDifficultyChange}
            onSelectTask={handleTaskSelect}
            onRefresh={() => loadTasks(difficulty)}
          />
        </aside>

        <div className="main-column">
          <Workspace
            task={selectedTask}
            observation={currentObservation}
            codeDraft={codeDraft}
            editLineNumber={editLineNumber}
            editLineContent={editLineContent}
            seed={seed}
            busy={busy}
            sessionId={sessionId}
            onCodeDraftChange={setCodeDraft}
            onEditLineNumberChange={setEditLineNumber}
            onEditLineContentChange={setEditLineContent}
            onSeedChange={setSeed}
            onStartSession={handleStartSession}
            onLineAction={handleLineAction}
          />
          <div className="bottom-grid">
            <ActionPanel
              sessionId={sessionId}
              busy={busy}
              observation={currentObservation}
              onAction={handleEnvironmentAction}
            />
            <OutputPanel
              observation={currentObservation}
              lastResult={lastResult}
              error={workspaceError}
              busy={busy}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const API_BASE_URL = import.meta.env.DEV ? "/api" : configuredBaseUrl.replace(/\/$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = data?.detail || data?.error || data?.message || `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return { data, headers: response.headers };
}

export function getConfiguredApiBaseUrl() {
  return configuredBaseUrl;
}

export async function getHealth() {
  const { data } = await request("/health");
  return data;
}

export async function getTasks(difficulty = "") {
  const query = difficulty ? `?difficulty=${encodeURIComponent(difficulty)}` : "";
  const { data } = await request(`/tasks${query}`);
  return data;
}

export async function getTask(taskId) {
  const { data } = await request(`/tasks/${encodeURIComponent(taskId)}`);
  return data;
}

export async function resetSession({ taskId, difficulty, seed } = {}) {
  const body = {};
  if (taskId) body.task_id = taskId;
  if (difficulty) body.difficulty = difficulty;
  if (seed !== "" && seed !== null && seed !== undefined) body.seed = Number(seed);

  const { data, headers } = await request("/reset", {
    method: "POST",
    body: JSON.stringify(body),
  });

  const sessionId = headers.get("X-Session-ID") || headers.get("x-session-id") || data?.session_id || "";
  return { observation: data, sessionId };
}

export async function stepSession(sessionId, action) {
  const { data } = await request("/step", {
    method: "POST",
    headers: {
      "X-Session-ID": sessionId,
    },
    body: JSON.stringify(action),
  });

  return data;
}

export async function getState(sessionId) {
  const { data } = await request("/state", {
    headers: {
      "X-Session-ID": sessionId,
    },
  });

  return data;
}

"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { Send, Loader2, Plus } from "lucide-react";
import type { ChatMessage } from "@/types";
import { agentsApi, dataApi } from "@/lib/api";
import type { Dataset } from "@/types";

/** Legacy global keys (migrated once into per-dataset keys). */
const LEGACY_SESSION = "chat_session_id";
const LEGACY_MSGS = "chat_messages";
/** Remember last dataset so remounting /chat does not hydrate bucket "_none" and wipe UI. */
const LAST_DATASET_KEY = "chat_last_dataset_id";

function storageKeys(datasetKey: string) {
  return {
    session: `chat_session_id:${datasetKey}`,
    msgs: `chat_messages:${datasetKey}`,
  };
}

/** Dataset id or "_none" when none selected. */
function datasetStorageKey(selectedDataset: string) {
  return selectedDataset || "_none";
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  /** Do not run chat hydrate until /data/list finished — avoids one pass with "" → _none that clears messages. */
  const [listReady, setListReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  /** True if current storage bucket had a session id before React state caught up — blocks restoreLatestSession race. */
  const skipAutoRestoreLatestRef = useRef(false);
  const legacyMigratedRef = useRef(false);
  const syncGenerationRef = useRef(0);

  const readFromStorage = useCallback((datasetKey: string) => {
    const k = storageKeys(datasetKey);
    let sid = localStorage.getItem(k.session);
    let rawMsgs = localStorage.getItem(k.msgs);

    if (!sid && !rawMsgs && !legacyMigratedRef.current) {
      const legacyS = localStorage.getItem(LEGACY_SESSION);
      const legacyM = localStorage.getItem(LEGACY_MSGS);
      if (legacyS || legacyM) {
        legacyMigratedRef.current = true;
        if (legacyS) {
          sid = legacyS;
          localStorage.setItem(k.session, legacyS);
        }
        if (legacyM) {
          rawMsgs = legacyM;
          localStorage.setItem(k.msgs, legacyM);
        }
        localStorage.removeItem(LEGACY_SESSION);
        localStorage.removeItem(LEGACY_MSGS);
      }
    }
    // Auto-select loads dataset after first paint; copy chat from "_none" bucket.
    if (!sid && !rawMsgs && datasetKey !== "_none") {
      const noneK = storageKeys("_none");
      const noneSid = localStorage.getItem(noneK.session);
      const noneRaw = localStorage.getItem(noneK.msgs);
      if (noneSid) {
        sid = noneSid;
        localStorage.setItem(k.session, noneSid);
      }
      if (noneRaw) {
        rawMsgs = noneRaw;
        localStorage.setItem(k.msgs, noneRaw);
      }
    }

    let parsedMsgs: ChatMessage[] = [];
    if (rawMsgs) {
      try {
        const parsed = JSON.parse(rawMsgs);
        if (Array.isArray(parsed) && parsed.length > 0) {
          parsedMsgs = parsed;
        }
      } catch {
        /* ignore */
      }
    }
    return { sid, parsedMsgs, keys: k };
  }, []);

  const syncSessionMessages = useCallback(async (id: string, keys: { session: string; msgs: string }) => {
    const gen = ++syncGenerationRef.current;
    try {
      const res = await agentsApi.getMessages(id);
      if (gen !== syncGenerationRef.current) return;
      const msgs: ChatMessage[] = res.data ?? [];
      if (msgs.length > 0) {
        setMessages(msgs);
        localStorage.setItem(keys.msgs, JSON.stringify(msgs));
      }
      // If API returns [] but we still have local backup, keep UI (e.g. replication lag); do not wipe.
    } catch (err) {
      if (gen !== syncGenerationRef.current) return;
      if ((err as { response?: { status?: number } })?.response?.status === 404) {
        localStorage.removeItem(keys.session);
        localStorage.removeItem(keys.msgs);
        setSessionId(null);
        setMessages([]);
        skipAutoRestoreLatestRef.current = false;
      }
    }
  }, []);

  const restoreLatestSession = useCallback(
    async (datasetId: string, keys: { session: string; msgs: string }) => {
      try {
        const res = await agentsApi.listSessions(datasetId);
        const sessions = res.data ?? [];
        if (sessions.length > 0) {
          const latest = sessions[0];
          setSessionId(latest.id);
          localStorage.setItem(keys.session, latest.id);
          skipAutoRestoreLatestRef.current = true;
          await syncSessionMessages(latest.id, keys);
        }
      } catch {
        // Ignore restore errors and keep any localStorage state we already have.
      }
    },
    [syncSessionMessages]
  );

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load datasets; pick last-used dataset from localStorage (valid id) or first configured.
  useEffect(() => {
    dataApi
      .list()
      .then((res) => {
        const all: Dataset[] = res.data.datasets ?? [];
        setDatasets(all);
        const ids = new Set(all.map((d) => d.id));
        const last =
          typeof window !== "undefined" ? localStorage.getItem(LAST_DATASET_KEY) : null;
        if (last && ids.has(last)) {
          setSelectedDataset(last);
        } else {
          const configured = all.find(
            (d) => d.dataset_type !== "future_variables" && d.datetime_column && d.target_column
          );
          if (configured) {
            setSelectedDataset(configured.id);
            localStorage.setItem(LAST_DATASET_KEY, configured.id);
          }
        }
      })
      .catch(() => {})
      .finally(() => setListReady(true));
  }, []);

  // Hydrate chat from localStorage + API whenever the storage bucket (dataset) changes.
  // skipAutoRestoreLatestRef prevents restoreLatestSession from overwriting a session id
  // that exists in localStorage but is not yet in React state (same commit as first paint).
  useEffect(() => {
    if (!listReady) return;

    const dKey = datasetStorageKey(selectedDataset);
    const { sid, parsedMsgs, keys } = readFromStorage(dKey);

    skipAutoRestoreLatestRef.current = !!sid;

    if (parsedMsgs.length > 0) {
      setMessages(parsedMsgs);
    } else {
      // Clear stale UI when switching dataset or bucket; API sync will refill if sid exists.
      setMessages([]);
    }

    if (sid) {
      setSessionId(sid);
      syncSessionMessages(sid, keys);
    } else {
      setSessionId(null);
      if (selectedDataset && !skipAutoRestoreLatestRef.current) {
        void restoreLatestSession(selectedDataset, keys);
      }
    }
  }, [listReady, selectedDataset, readFromStorage, syncSessionMessages, restoreLatestSession]);

  // Re-sync when the tab becomes active again.
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === "visible" && sessionId) {
        const k = storageKeys(datasetStorageKey(selectedDataset));
        syncSessionMessages(sessionId, k);
      }
    };

    window.addEventListener("focus", handleVisibility);
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      window.removeEventListener("focus", handleVisibility);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [sessionId, selectedDataset, syncSessionMessages]);

  const persistMessages = useCallback((msgs: ChatMessage[]) => {
    const k = storageKeys(datasetStorageKey(selectedDataset));
    localStorage.setItem(k.msgs, JSON.stringify(msgs));
  }, [selectedDataset]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg: ChatMessage = { role: "user", content: input };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    persistMessages(newMsgs);
    setInput("");
    setLoading(true);

    const keys = storageKeys(datasetStorageKey(selectedDataset));

    try {
      const res = await agentsApi.sendMessage(
        input,
        sessionId || undefined,
        selectedDataset || undefined,
      );
      const data = res.data;

      const newSessionId = data.session_id;
      if (!sessionId) {
        setSessionId(newSessionId);
        localStorage.setItem(keys.session, newSessionId);
        skipAutoRestoreLatestRef.current = true;
      }

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: data.reply,
        tool_calls: data.tool_calls_made?.length ? data.tool_calls_made : undefined,
      };
      const updated = [...newMsgs, assistantMsg];
      setMessages(updated);
      persistMessages(updated);

      if (data.executed_operations?.length) {
        window.dispatchEvent(
          new CustomEvent("datasets-updated", {
            detail: {
              operations: data.executed_operations,
              datasetId: selectedDataset || null,
            },
          })
        );
      }
    } catch {
      const errMsg: ChatMessage = {
        role: "assistant",
        content: "Error sending message — please try again.",
      };
      const updated = [...newMsgs, errMsg];
      setMessages(updated);
      persistMessages(updated);
    } finally {
      setLoading(false);
    }
  };

  const newSession = () => {
    if (messages.length > 0 && !confirm("Start a new conversation? Current history will be cleared.")) return;
    syncGenerationRef.current += 1;
    const k = storageKeys(datasetStorageKey(selectedDataset));
    localStorage.removeItem(k.session);
    localStorage.removeItem(k.msgs);
    skipAutoRestoreLatestRef.current = false;
    setSessionId(null);
    setMessages([]);
  };

  return (
    <AppShell>
      <div className="flex h-[calc(100vh-7rem)] flex-col gap-3">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Chat</h1>
            <p className="text-sm text-gray-500">
              Talk to the AI agent — analyze data, configure models, run forecasts
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Dataset context selector */}
            <div className="flex flex-col items-end gap-0.5">
              <select
                value={selectedDataset}
                onChange={(e) => {
                  const v = e.target.value;
                  setSelectedDataset(v);
                  if (v) localStorage.setItem(LAST_DATASET_KEY, v);
                  else localStorage.removeItem(LAST_DATASET_KEY);
                }}
                className={`rounded-xl border px-3 py-2 text-sm dark:bg-gray-900 dark:border-gray-700 ${
                  !selectedDataset
                    ? "border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20"
                    : "bg-white"
                }`}
              >
                <option value="">⚠ No dataset — select one</option>
                {datasets
                  .filter((d) => d.dataset_type !== "future_variables")
                  .map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                      {d.target_column ? ` · target: ${d.target_column}` : " ⚠ unconfigured"}
                    </option>
                  ))}
              </select>
              {!selectedDataset && (
                <p className="text-xs text-yellow-600 dark:text-yellow-400">
                  Select dataset for operations to take effect
                </p>
              )}
            </div>
            <button
              onClick={newSession}
              title="New conversation"
              className="flex items-center gap-1 rounded-xl border px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-800"
            >
              <Plus className="h-4 w-4" />
              New
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto rounded-xl border bg-white p-4 dark:bg-gray-900 dark:border-gray-800">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-gray-400 gap-3">
              <p className="text-sm">Start a conversation</p>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 max-w-lg">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="rounded-xl border px-4 py-2 text-left text-xs text-gray-500 hover:bg-gray-50 hover:text-gray-700 dark:border-gray-700 dark:hover:bg-gray-800"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages
                .filter((m) => m.role === "user" || m.role === "assistant")
                .map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                        msg.role === "user"
                          ? "bg-brand-600 text-white"
                          : "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                      {msg.tool_calls && msg.tool_calls.length > 0 && (
                        <div className="mt-2 border-t border-white/20 pt-2 dark:border-gray-700">
                          <p className="text-xs opacity-60">
                            Tools: {msg.tool_calls.map((t) => (t as Record<string, string>).name ?? "tool").join(", ")}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl bg-gray-100 px-4 py-3 text-sm text-gray-500 dark:bg-gray-800 dark:text-gray-300">
                    <div className="flex items-center gap-2">
                      <span>Assistant is typing</span>
                      <span className="flex items-center gap-1">
                        <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:0ms]" />
                        <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:120ms]" />
                        <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:240ms]" />
                      </span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
            placeholder="Type a message… (Enter to send)"
            className="flex-1 rounded-xl border bg-white px-4 py-3 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:bg-gray-900 dark:border-gray-700"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-brand-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-brand-700 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>

        {sessionId && (
          <p className="text-center text-xs text-gray-400">
            Session {sessionId.slice(0, 8)}… •{" "}
            <button onClick={newSession} className="underline hover:text-gray-600">
              clear history
            </button>
          </p>
        )}
      </div>
    </AppShell>
  );
}

const SUGGESTIONS = [
  "Analyze the uploaded dataset",
  "What's the best model for my data?",
  "Run a 12-period forecast",
  "What are the most important features?",
];

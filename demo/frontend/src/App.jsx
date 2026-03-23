import { useRef, useState } from "react";

import { askAgent, streamAgent } from "./api";
import ChatInterface from "./components/ChatInterface";
import ReasoningPanel from "./components/ReasoningPanel";
import ToolTrace from "./components/ToolTrace";

const EXAMPLES = [
  "What was the percentage change in total revenue from 2018 to 2019?",
  "Calculate the CAGR from 100 to 150 over 3 years.",
  "Use SQL to inspect the relevant financial table schema before answering.",
];

const INITIAL_STATE = {
  answer: "",
  toolTrace: [],
  reasoningTurns: [],
  observations: [],
  fullText: "",
  isLoading: false,
};

function applyStreamEvent(previous, eventType, payload) {
  if (eventType === "assistant") {
    return {
      ...previous,
      reasoningTurns: [...previous.reasoningTurns, payload],
      isLoading: true,
    };
  }

  if (eventType === "tool") {
    return {
      ...previous,
      toolTrace: [...previous.toolTrace, payload],
      isLoading: true,
    };
  }

  if (eventType === "observation") {
    return {
      ...previous,
      observations: [...previous.observations, payload],
      isLoading: true,
    };
  }

  if (eventType === "done") {
    return {
      ...previous,
      answer: payload.answer,
      toolTrace: payload.tool_trace ?? previous.toolTrace,
      fullText: payload.full_text,
      isLoading: false,
    };
  }

  if (eventType === "error") {
    return {
      ...previous,
      isLoading: false,
    };
  }

  return previous;
}

export default function App() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [runState, setRunState] = useState(INITIAL_STATE);
  const [error, setError] = useState("");
  const abortRef = useRef(null);

  async function handleSubmit(event) {
    event.preventDefault();
    const nextQuestion = question.trim();
    if (!nextQuestion) {
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError("");
    setRunState({
      ...INITIAL_STATE,
      isLoading: true,
    });

    try {
      await streamAgent(nextQuestion, {
        signal: controller.signal,
        onEvent: (eventType, payload) => {
          setRunState((previous) => applyStreamEvent(previous, eventType, payload));
        },
      });
    } catch (requestError) {
      if (controller.signal.aborted) {
        return;
      }

      try {
        const fallback = await askAgent(nextQuestion);
        setRunState({
          answer: fallback.answer,
          toolTrace: fallback.tool_trace,
          reasoningTurns: [],
          observations: [],
          fullText: fallback.full_text,
          isLoading: false,
        });
        setError(`Streaming fallback used: ${requestError.message}`);
      } catch (fallbackError) {
        setRunState((previous) => ({
          ...previous,
          isLoading: false,
        }));
        setError(fallbackError.message);
      }
    } finally {
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunState((previous) => ({
      ...previous,
      isLoading: false,
    }));
  }

  return (
    <main className="app-shell">
      <div className="background-orbit background-orbit-a" />
      <div className="background-orbit background-orbit-b" />

      <div className="layout">
        <ChatInterface
          question={question}
          setQuestion={setQuestion}
          isLoading={runState.isLoading}
          error={error}
          answer={runState.answer}
          onSubmit={handleSubmit}
          onStop={handleStop}
          examples={EXAMPLES}
        />
        <ToolTrace toolTrace={runState.toolTrace} />
        <ReasoningPanel
          reasoningTurns={runState.reasoningTurns}
          observations={runState.observations}
          fullText={runState.fullText}
        />
      </div>
    </main>
  );
}

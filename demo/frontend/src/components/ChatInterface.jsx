export default function ChatInterface({
  question,
  setQuestion,
  isLoading,
  error,
  answer,
  onSubmit,
  onStop,
  examples,
}) {
  return (
    <section className="panel panel-hero">
      <div className="panel-header">
        <p className="eyebrow">Multi-Turn Agent Console</p>
        <h1>FinAgent-R1</h1>
        <p className="muted">
          Stream the agent&apos;s reasoning, watch tool calls land, and inspect the final
          answer without leaving the browser.
        </p>
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <label className="field-label" htmlFor="question">
          Financial question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={5}
          placeholder="Ask a multi-step finance question that might need search, SQL, and calculation."
        />
        <div className="composer-actions">
          <button className="primary-button" type="submit" disabled={isLoading || !question.trim()}>
            {isLoading ? "Streaming..." : "Run Agent"}
          </button>
          <button className="ghost-button" type="button" onClick={onStop} disabled={!isLoading}>
            Stop
          </button>
        </div>
      </form>

      <div className="example-grid">
        {examples.map((example) => (
          <button
            key={example}
            className="example-chip"
            type="button"
            onClick={() => setQuestion(example)}
          >
            {example}
          </button>
        ))}
      </div>

      <div className="answer-card">
        <div className="answer-label">Final answer</div>
        <div className="answer-content">{answer || "The final answer will appear here."}</div>
        {error ? <div className="error-banner">{error}</div> : null}
      </div>
    </section>
  );
}

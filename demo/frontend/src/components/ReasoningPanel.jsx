export default function ReasoningPanel({ reasoningTurns, observations, fullText }) {
  return (
    <section className="panel">
      <div className="panel-header compact">
        <p className="eyebrow">Reasoning Stream</p>
        <h2>Turn-by-turn output</h2>
      </div>

      <div className="reasoning-grid">
        <div className="reasoning-column">
          <h3>Assistant turns</h3>
          {reasoningTurns.length ? (
            reasoningTurns.map((turn) => (
              <article key={`assistant-${turn.turn}`} className="reasoning-card">
                <div className="reasoning-title">Turn {turn.turn}</div>
                <pre>{turn.content}</pre>
              </article>
            ))
          ) : (
            <div className="empty-state">No streamed reasoning yet.</div>
          )}
        </div>

        <div className="reasoning-column">
          <h3>Observations</h3>
          {observations.length ? (
            observations.map((item) => (
              <article key={`observation-${item.turn}`} className="observation-card">
                <div className="reasoning-title">Turn {item.turn}</div>
                <pre>{item.content}</pre>
              </article>
            ))
          ) : (
            <div className="empty-state">Tool observations will appear here.</div>
          )}
        </div>
      </div>

      <div className="full-trace">
        <div className="reasoning-title">Full trace</div>
        <pre>{fullText || "The complete transcript will appear here once the run finishes."}</pre>
      </div>
    </section>
  );
}

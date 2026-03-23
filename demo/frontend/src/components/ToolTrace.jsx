export default function ToolTrace({ toolTrace }) {
  return (
    <section className="panel">
      <div className="panel-header compact">
        <p className="eyebrow">Tool Trace</p>
        <h2>Actions</h2>
      </div>

      <div className="tool-list">
        {toolTrace.length ? (
          toolTrace.map((entry) => (
            <article key={`${entry.turn}-${entry.tool}-${entry.query}`} className="tool-card">
              <div className="tool-meta">
                <span>Turn {entry.turn}</span>
                <span className={`tool-badge tool-${entry.tool}`}>{entry.tool}</span>
              </div>
              <div className="tool-query">{entry.query}</div>
              <pre>{entry.result}</pre>
            </article>
          ))
        ) : (
          <div className="empty-state">Tool calls will show up here as they happen.</div>
        )}
      </div>
    </section>
  );
}

/**
 * PipelineInspector
 * Collapsible panel showing RAG pipeline internals:
 * rewritten question, retrieved docs summary, embeddings distance.
 */
import { useState } from "react";

export default function PipelineInspector({ meta }) {
    const [open, setOpen] = useState(false);

    if (!meta) return null;

    return (
        <div className="inspector">
            <button
                className="inspector-toggle"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
            >
                <span className="inspector-icon">🔍</span>
                <span>Pipeline Inspector</span>
                <span className="inspector-chevron">{open ? "▾" : "▸"}</span>
            </button>

            {open && (
                <div className="inspector-body">
                    {/* Rewrite info */}
                    <div className="inspector-section">
                        <div className="inspector-label">Query</div>
                        <div className="inspector-value">
                            {meta.was_rewritten ? (
                                <>
                                    <span className="inspector-badge rewrite">Rewritten</span>
                                    {meta.rewritten_question}
                                </>
                            ) : (
                                <>
                                    <span className="inspector-badge standalone">Standalone</span>
                                    {meta.rewritten_question}
                                </>
                            )}
                        </div>
                    </div>

                    {/* History info */}
                    <div className="inspector-section">
                        <div className="inspector-label">History</div>
                        <div className="inspector-value">
                            {meta.history_turns} prior turn{meta.history_turns !== 1 ? "s" : ""} in context
                        </div>
                    </div>

                    {/* Retrieval filters */}
                    <div className="inspector-section">
                        <div className="inspector-label">Filters Applied</div>
                        <div className="inspector-value">
                            {meta.retrieval_filters ? (
                                <span style={{ fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-saffron)" }}>
                                    {JSON.stringify(meta.retrieval_filters, null, 0)}
                                </span>
                            ) : (
                                <span style={{ color: "var(--text-dim)", fontSize: "11px" }}>
                                    None — pure semantic search
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Aggregate Stats */}
                    {meta.aggregate_stats && (
                        <>
                            <div className="inspector-section mt-2">
                                <div className="inspector-label">Stat Extraction Setup</div>
                                <div className="inspector-value" style={{ fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-saffron)" }}>
                                    group_by: "{meta.group_by}" | metric: "{meta.metric}"
                                </div>
                            </div>
                            <div className="inspector-section mt-2">
                                <div className="inspector-label">System Calculated Stats</div>
                                <div className="inspector-value" style={{ whiteSpace: "pre-wrap", fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-emerald)" }}>
                                    {meta.aggregate_stats}
                                </div>
                            </div>
                        </>
                    )}

                    {/* Extracted Doc Table Component */}
                    {meta.initial_top_docs && meta.initial_top_docs.length > 0 && (
                        <div className="inspector-section mt-2">
                            <div className="inspector-label">
                                Initial Retrieval ({meta.initial_num_docs} deliveries)
                            </div>
                            <DocTable docs={meta.initial_top_docs} showScore={false} />
                        </div>
                    )}

                    {/* Retrieved docs */}
                    <div className="inspector-section mt-2">
                        <div className="inspector-label">
                            After Reranking ({meta.num_docs} deliveries)
                        </div>
                        {meta.top_docs && meta.top_docs.length > 0 && (
                            <DocTable docs={meta.top_docs} showScore={true} />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

function DocTable({ docs, showScore }) {
    return (
        <table className="inspector-table">
            <thead>
                <tr>
                    <th>Inn</th>
                    <th>Over</th>
                    <th>Batter</th>
                    <th>Bowler</th>
                    <th>Event</th>
                    <th>Runs</th>
                    <th>Dist</th>
                    {showScore && <th>Score</th>}
                </tr>
            </thead>
            <tbody>
                {docs.map((doc, i) => (
                    <tr key={i}>
                        <td>{doc.innings}</td>
                        <td>{doc.over}</td>
                        <td>{doc.batter}</td>
                        <td>{doc.bowler}</td>
                        <td>
                            <span className={`inspector-event ev-${doc.event}`}>
                                {doc.event}
                            </span>
                        </td>
                        <td>{doc.runs}</td>
                        <td>{doc.distance}</td>
                        {showScore && <td>{typeof doc.score === 'number' ? doc.score.toFixed(3) : '-'}</td>}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

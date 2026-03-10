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

                            {meta.aggregate_stats.includes("=== EXACT STATS FOR TOP CONTENDERS ===") ? (
                                <>
                                    <div className="inspector-section mt-2">
                                        <div className="inspector-label">Stat Leaderboard</div>
                                        <div className="inspector-value" style={{ whiteSpace: "pre-wrap", fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-emerald)" }}>
                                            {meta.aggregate_stats.split("=== EXACT STATS FOR TOP CONTENDERS ===")[0].trim()}
                                        </div>
                                    </div>
                                    <div className="inspector-section mt-2">
                                        <div className="inspector-label">Injected Exact Stats (Anti-Hallucination)</div>
                                        <div className="inspector-value" style={{ whiteSpace: "pre-wrap", fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-blue)" }}>
                                            {"=== EXACT STATS FOR TOP CONTENDERS ===\n" + meta.aggregate_stats.split("=== EXACT STATS FOR TOP CONTENDERS ===")[1].trim()}
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <div className="inspector-section mt-2">
                                    <div className="inspector-label">System Calculated Stats</div>
                                    <div className="inspector-value" style={{ whiteSpace: "pre-wrap", fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: "11px", color: "var(--accent-emerald)" }}>
                                        {meta.aggregate_stats}
                                    </div>
                                </div>
                            )}
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

                    {/* LLM Call Traces */}
                    {meta.llm_traces && meta.llm_traces.length > 0 && (
                        <div className="inspector-section mt-2" style={{ borderTop: "1px solid var(--border)", paddingTop: "12px" }}>
                            <div className="inspector-label" style={{ color: "var(--accent-purple)" }}>
                                ⚡ LLM Call Traces ({meta.llm_traces.length})
                            </div>
                            <div className="llm-traces-container" style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "8px" }}>
                                {meta.llm_traces.map((trace, i) => (
                                    <details key={i} className="llm-trace-box" style={{ background: "rgba(0,0,0,0.2)", border: "1px solid var(--border)", borderRadius: "6px", overflow: "hidden" }}>
                                        <summary style={{ padding: "8px 12px", cursor: "pointer", fontSize: "12px", fontWeight: "600", color: "var(--text)", background: "rgba(255,255,255,0.05)", userSelect: "none" }}>
                                            Step {i + 1}: <span style={{ fontFamily: "monospace", color: "var(--accent-blue)" }}>{trace.node}</span>
                                        </summary>
                                        <div style={{ padding: "12px", borderTop: "1px solid var(--border)", fontSize: "11px", fontFamily: "'SF Mono', 'Fira Code', monospace", whiteSpace: "pre-wrap" }}>
                                            <div style={{ color: "var(--text-dim)", marginBottom: "4px" }}>// PROMPT:</div>
                                            <div style={{ color: "var(--text)", background: "rgba(0,0,0,0.3)", padding: "8px", borderRadius: "4px", marginBottom: "12px" }}>
                                                {trace.prompt}
                                            </div>
                                            <div style={{ color: "var(--text-dim)", marginBottom: "4px" }}>// LLM RESPONSE:</div>
                                            <div style={{ color: "var(--accent-emerald)", background: "rgba(0,0,0,0.3)", padding: "8px", borderRadius: "4px" }}>
                                                {trace.response}
                                            </div>
                                        </div>
                                    </details>
                                ))}
                            </div>
                        </div>
                    )}
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

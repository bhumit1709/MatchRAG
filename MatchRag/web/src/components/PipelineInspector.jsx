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

                    {/* Retrieved docs */}
                    <div className="inspector-section">
                        <div className="inspector-label">
                            Retrieved {meta.num_docs} deliveries
                        </div>
                        {meta.top_docs && meta.top_docs.length > 0 && (
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
                                    </tr>
                                </thead>
                                <tbody>
                                    {meta.top_docs.map((doc, i) => (
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
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

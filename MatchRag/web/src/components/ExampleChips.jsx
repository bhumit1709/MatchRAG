/**
 * ExampleChips
 * Clickable starter question chips shown on the welcome screen.
 */
const EXAMPLES = [
    "Who dismissed Shimron Hetmyer?",
    "What happened in the last over?",
    "Who hit the most sixes?",
    "Show all wickets taken by Bumrah.",
    "How did India win the match?",
    "Which over had the most runs?",
];

export default function ExampleChips({ onSelect }) {
    return (
        <div className="welcome">
            <div className="welcome-icon">🏏</div>
            <div>
                <h1 className="welcome-title">Cricket Match RAG</h1>
                <p className="welcome-subtitle">
                    Ask anything about the India vs West Indies T20 World Cup match.
                    Powered by local AI with ball-by-ball commentary context.
                </p>
            </div>
            <p className="chips-label">Try asking</p>
            <div className="chips-grid">
                {EXAMPLES.map((q) => (
                    <button key={q} className="chip" onClick={() => onSelect(q)}>
                        {q}
                    </button>
                ))}
            </div>
        </div>
    );
}

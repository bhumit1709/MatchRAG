/**
 * ChatMessage
 * Renders a single message bubble — either bot or user.
 */
export default function ChatMessage({ role, text, elapsed }) {
    const isUser = role === "user";

    return (
        <div className={`message-row ${isUser ? "user" : ""}`}>
            <div className={`avatar ${isUser ? "avatar-user" : "avatar-bot"}`}>
                {isUser ? "👤" : "🏏"}
            </div>

            <div className="message-body">
                <div className={`bubble ${isUser ? "bubble-user" : "bubble-bot"}`}>
                    {text}
                </div>

                {!isUser && elapsed !== undefined && (
                    <div className="message-meta">
                        <span className="elapsed-badge">⚡ {elapsed}s</span>
                    </div>
                )}
            </div>
        </div>
    );
}

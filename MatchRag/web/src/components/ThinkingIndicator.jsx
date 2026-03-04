/**
 * ThinkingIndicator
 * Animated bouncing-dots bubble shown while awaiting the bot response.
 */
export default function ThinkingIndicator() {
    return (
        <div className="message-row">
            <div className="avatar avatar-bot">🏏</div>
            <div className="message-body">
                <div className="bubble bubble-bot thinking-bubble">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                </div>
            </div>
        </div>
    );
}

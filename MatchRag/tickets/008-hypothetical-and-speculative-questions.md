# Ticket 008: Hypothetical Or Speculative Questions

## Issue

The chatbot should not answer speculative or counterfactual questions, but it does not yet have a dedicated unsupported-intent path for them.

Examples:
- "What if Samson got out early?"
- "Why did New Zealand choke?"
- "Would India still have won without Bumrah?"

Current gap:
- The assistant is meant to stay grounded in match data.
- There is no explicit intent detection and fallback flow for speculative questions.
- Unsupported speculative prompts could still trigger an unhelpful or vague answer.

## Suggested Fix

- Add unsupported-intent detection for speculative and counterfactual questions.
- Return a clear fallback message explaining that only factual match-data questions are supported.
- Add tests that verify the bot declines speculative questions safely and consistently.

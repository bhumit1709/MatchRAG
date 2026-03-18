# Ticket 005: Extras And Wicket-Mode Analytics

## Issue

The chatbot does not support exact analytics for extras and dismissal modes.

Examples:
- "How many wides were bowled?"
- "How many leg-byes were there?"
- "What was the most common wicket type?"

Current gap:
- Extras and wicket-kind fields are not exposed as user-facing analytic routes.
- Fast-path aggregation does not support these event types.
- These questions may only work inconsistently through commentary retrieval.

## Suggested Fix

- Extend the event taxonomy to include wides, no-balls, byes, leg-byes, and wicket kinds.
- Add exact aggregate helpers for extras and dismissal-type counts.
- Add planner support for these analytics questions.
- Add tests for extras and wicket-kind queries.

# Ticket 002: Partnership Analytics

## Issue

The chatbot does not support partnership-based questions for the India vs New Zealand final match.

Examples:
- "What was the highest partnership?"
- "Who had the best partnership for India?"
- "How many runs did Samson and Pandya add together?"

Current gap:
- Partnership spans are not computed from wickets and scoring sequences.
- The system has no partnership aggregation helpers.
- There are no answer templates for partnership summaries.

## Suggested Fix

- Derive partnership windows from wicket boundaries and consecutive batting segments.
- Add exact partnership aggregates such as highest partnership and named-pair partnership runs.
- Add planner support for partnership intents.
- Add tests for highest-partnership and named-pair partnership queries.

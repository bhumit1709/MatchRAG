# Ticket 007: Cross-Match Or Tournament Questions

## Issue

The chatbot cannot answer questions that span beyond the currently indexed India vs New Zealand final match.

Examples:
- "How did India perform across the tournament?"
- "Compare this final to the West Indies match."
- "Which match was Samson's best?"

Current gap:
- The app is scoped to a single indexed match.
- Retrieval is not namespaced across multiple matches in one query flow.
- The planner has no support for cross-match comparison intents.

## Suggested Fix

- Add multi-match indexing with per-match namespaces or collections.
- Support cross-match retrieval and comparison in the planner.
- Expose match selection and match metadata to the query pipeline.
- Add tests for multi-match and tournament-level questions.

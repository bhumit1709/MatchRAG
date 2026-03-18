# Ticket 001: Phase-Based Exact Analytics

## Issue

The chatbot cannot reliably answer exact phase-based questions for the India vs New Zealand final match.

Examples:
- "How many runs did India score in the powerplay?"
- "Who was best in the death overs?"
- "How many wickets fell in the middle overs?"

Current gap:
- The retrieval plan does not model phases like `powerplay`, `middle_overs`, or `death_overs`.
- The metadata filter layer cannot slice deliveries by phase.
- There are no exact aggregate helpers for phase-level stats.

## Suggested Fix

- Extend the retrieval-plan schema with a phase field.
- Add deterministic phase filters in the retrieval layer.
- Build exact aggregate helpers for runs, wickets, strike rate, and economy by phase.
- Add tests for powerplay, middle-over, and death-over questions.

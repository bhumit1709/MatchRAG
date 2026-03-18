# Ticket 004: Exact Score Progression And Scorecards

## Issue

The chatbot cannot produce exact score snapshots or structured scorecards for the match.

Examples:
- "What was India's score after 10 overs?"
- "What was New Zealand's score at 15.3?"
- "Give me the batting scorecard."
- "Show the bowling figures."

Current gap:
- Running score progression is not computed by ball or by over.
- The context builder does not generate batting or bowling scorecards.
- Score-state questions are not routed to exact structured helpers.

## Suggested Fix

- Compute cumulative score progression by ball and by over.
- Add exact helpers for score-at-over and score-at-ball questions.
- Generate batting and bowling scorecards from structured delivery data.
- Add tests for score snapshots and full scorecard outputs.

# Ticket 003: Batter-vs-Bowler Matchup Stats

## Issue

The chatbot cannot answer exact matchup questions between a batter and a bowler.

Examples:
- "How many runs did Samson score off Henry?"
- "How did Bumrah bowl to Seifert?"
- "Which bowler troubled Samson the most?"

Current gap:
- The planner does not detect batter-vs-bowler matchup intent.
- The aggregation layer does not support grouping by batter-bowler pairs.
- Responses would currently depend on best-effort retrieval instead of exact stats.

## Suggested Fix

- Add matchup-intent detection in the retrieval planner.
- Implement exact aggregations over batter and bowler combinations.
- Add response templates for matchup summaries and leader-style matchup questions.
- Add tests for batter-vs-bowler stats and pairwise comparisons.

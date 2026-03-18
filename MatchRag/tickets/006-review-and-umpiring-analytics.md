# Ticket 006: Review And Umpiring Analytics

## Issue

The chatbot cannot answer exact DRS and umpiring-analysis questions.

Examples:
- "How many DRS reviews were taken?"
- "Which reviews were successful?"
- "How many umpire's calls were there?"

Current gap:
- Review events are only present implicitly in commentary text.
- The ingestion pipeline does not extract review outcomes into structured metadata.
- There are no exact counters or answer templates for review analytics.

## Suggested Fix

- Extract review events and outcomes from commentary into structured metadata.
- Add exact counters for reviews, successful reviews, and umpire's call decisions.
- Add answer templates for review-analysis questions.
- Add tests for DRS-related queries.

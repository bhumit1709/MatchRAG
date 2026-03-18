# Ticket 009: Visual Analytics Requests

## Issue

The chatbot cannot fulfill requests for charts or visual match analytics.

Examples:
- "Show a worm chart."
- "Plot a wagon wheel for Samson."
- "Give me a run-rate graph."

Current gap:
- The app does not generate chart-ready structured outputs.
- The backend has no visual-analytics data builders.
- The frontend has no flow for rendering requested charts from match data.

## Suggested Fix

- Add structured data builders for run-rate, score progression, and shot-summary visualizations.
- Decide whether charts should be rendered in the frontend or returned as API payloads.
- Add tests for chart-data generation and output shape.

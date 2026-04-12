---
name: Engine implementation task
about: Track a specific engine or module implementation from TASKS.md
labels: enhancement
---

## Task Reference

- TASKS.md task: <!-- e.g., B1, C3, E2 -->
- File to implement: <!-- e.g., engines/macro/fred_client.py -->

## Acceptance Criteria

- [ ] All public functions have type hints
- [ ] All API calls wrapped with `@retry`
- [ ] Unit tests in `tests/unit/` cover the main logic
- [ ] Module can be run standalone: `python -m engines.<module>`

## Notes

<!-- Any blockers, API key requirements, or design decisions -->

## Summary

<!-- What does this PR do? Which tasks in TASKS.md does it complete? -->

- Completes task(s): <!-- e.g., A2, A3, B1 -->
- Engine/module affected: <!-- e.g., engines/macro, utils -->

## Changes

<!-- List specific changes made -->

- 

## Test Plan

- [ ] Unit tests added/updated for new functions
- [ ] `pytest tests/unit/ -v` passes locally
- [ ] `ruff check .` passes locally
- [ ] Tested manually with: <!-- describe how you tested it -->

## Checklist

- [ ] All timestamps go through `utils/time_utils.to_utc()`
- [ ] All API calls are wrapped with `@retry` from `utils/retry.py`
- [ ] No API keys hardcoded — using `from config.settings import settings`
- [ ] No raw SQLAlchemy sessions in engine code (use repositories)
- [ ] `structlog.get_logger()` used for logging (no bare `print()`)

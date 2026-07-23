---
name: feedback_tmp_scripts
description: Always use scripts/tmp/ for throwaway/one-off scripts, never inline python3 -c
type: feedback
---

Always write throwaway or generator scripts to `scripts/tmp/` (top-level, not inside a project). Never use inline `python3 -c "..."` commands in Bash.

**Why:** User has explicitly corrected this twice. Inline scripts are hard to review and re-run.

**How to apply:** Any time you would use `python3 -c` or `uv run python -c`, instead write the script to `scripts/tmp/something.py` and run that file.

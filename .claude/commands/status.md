Check the current status of the macrocast project.

1. List all Python files under macrocast/ and count lines of code per module
2. List all test files under tests/ and count tests
3. Check if pyproject.toml dependencies are synced: `uv sync --dry-run`
4. Run `uv run ruff check macrocast/ --statistics` for lint summary
5. Check git status: `git log --oneline -5` and `git status --short`

Present a concise summary table:
- Layer 1 (Data): which modules exist, test coverage
- Layer 2 (Pipeline): which modules exist, test coverage
- Layer 3 (Evaluation): which modules exist, test coverage
- Overall: total files, total lines, total tests, lint issues

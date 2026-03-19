Run the test suite for macrocast.

1. Run `uv run pytest tests/ -v --tb=short` for full suite
2. If $ARGUMENTS is specified, run only matching tests: `uv run pytest tests/ -v --tb=short -k "$ARGUMENTS"`
3. Report any failures with the specific assertion that failed
4. If all pass, confirm with count of tests passed

Download and validate FRED data to verify the data layer works correctly.

1. Run the data download for the specified database: $ARGUMENTS (fred_md, fred_qd, or fred_sd)
2. Check that the downloaded file parses correctly
3. Verify transformation codes are present and valid (tcodes 1-7)
4. Check variable count matches expected (FRED-MD: ~130, FRED-QD: ~250)
5. Report date range, missing value summary, and any anomalies
6. If fred_sd, verify xlsx parsing and state coverage

Use `uv run python -c "..."` for quick validation scripts.

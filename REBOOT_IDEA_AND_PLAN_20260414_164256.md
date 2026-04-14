# Macrocast reboot plan

Date: 2026-04-14 16:42:56 KST

Status:
- Previous working tree archived under `archive/reboot_snapshot_20260414_164256`.
- Full project surface except `.git`, `.gitignore`, and `archive/` has been moved into the reboot snapshot.
- Rebuild will start from a near-empty repo root.

Rules:
- Treat prior implementation as reference only.
- Do not restore archived files into root without explicit reason.
- Rebuild package architecture from first principles.

First next action:
- Define the new package contract before recreating code/docs layout.

Archive snapshot:
- `archive/reboot_snapshot_20260414_164256`

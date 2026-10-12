# GitHub Actions cost controls

The user requires avoiding unnecessary GitHub Actions minutes and charges. These instructions take precedence over general instructions to push, run CI, or use cloud verification.

- Workflows in this repository are intentionally manual-only. Preserve this policy; do not add automatic push, pull_request, schedule, or workflow_run triggers without an explicit user request for automatic Actions execution.
- Run relevant lint, tests, type checks, and builds locally first. If local verification is unavailable, report that gap; do not start hosted jobs as an automatic fallback.
- Do not dispatch or rerun GitHub Actions unless the user explicitly requests a GitHub Actions run. A request to implement, test, commit, push, or deploy does not by itself authorize hosted CI.
- Before an authorized Actions run, check available minutes or the account budget and explain the expected runner use. Never raise or remove spending limits.
- Native Android and iOS builds require an explicit choice. Keep native build inputs false by default and run native jobs only after source checks pass. Use macOS runners only for explicitly requested Apple platform verification or release builds.
- Inspect workflow triggers before each push or PR update. Batch completed, locally verified changes into one final push whenever practical; avoid repeated pushes just to test small edits in CI.
- Keep workflow concurrency cancellation, bounded job timeouts, and short artifact retention. Reuse existing valid results and caches rather than rerunning unchanged work.
- Do not bypass failed tests or required branch checks to save minutes, and do not claim verification that was not performed. Report local results and any remaining CI or native verification gap.

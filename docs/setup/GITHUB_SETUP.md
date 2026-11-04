# Private GitHub repository setup — planned, not performed

Requested repository: `trading` (private). No account action authorized yet.

When specifically approved, owner or connected authorized agent:
1. Authenticate GitHub CLI locally (`gh auth login`); verify identity using `gh auth status` without exposing tokens.
2. Review `.gitignore`, `.env.example`, repository contents; scan for secrets, account numbers, licensed datasets, course copies and generated model artifacts.
3. Initialize `git init -b main` only if not already a Git repository; `git status --short --branch`; commit reviewed planning files as `docs(plan): establish stage 00 baseline`.
4. Use `gh repo create trading --private --source=. --remote=origin` after explicit permission (do not add `--push` until push is separately authorized); verify private visibility with `gh repo view --json visibility,nameWithOwner`.
5. Configure branch protection and required checks when plan/platform supports it; future work on `research/...`, `feat/...`, `fix/...` branches. PR + human approval at stage gates; no force push or history rewrite.
6. Update `START_HERE.md` with *actual* branch, HEAD SHA, remote and push status. Never invent Git state.

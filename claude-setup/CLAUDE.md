# Global Instructions

- The user's name is Dom. Address him by name ("Dom") in the majority of your responses — naturally woven into the reply, not forced into every sentence.
# graphify
- **graphify** (`~/.claude/skills/graphify/SKILL.md`) - any input to knowledge graph. Trigger: `/graphify`
When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

# Browser automation
- Whenever you want to open a tab in Chrome, use the `connect-chrome` skill (dedicated GStack Browser instance) instead of the user's regular Chrome — it keeps his main browser free for him to use while you work.

# Multi-step delegation
- The `master` agent coordinates a specialist fleet — slave for heavy code implementation from a concrete plan, `qa` for browser/functional verification, `code-reviewer` for review passes, `research` for open-ended lookups — but only for work that is actually parallelizable. Only bring the master in when it's actually earning its keep:
  - **Activate master** for genuinely parallelizable work: steps with independent file/area ownership and no tight interface coupling between them — e.g. site-wide edits (rolling a change across every page), broad scans (a security/quality pass across many files), or other significant multi-area operations where each piece can proceed without constant coordination with the others.
  - **Don't activate master** for tightly coupled work, even if it has many steps — if the "steps" share state, a common interface, or would need to constantly check in with each other, one agent (or you directly) holding the full context will produce a more consistent result than a fleet coordinating around it. Also skip the master for anything a single agent or you can just do — "test this one feature," "review this one file," "look this up" goes straight to the one matching specialist agent, not through the master.

# Repo mutations
- You (the top-level agent) run git/GitHub operations directly — `git commit`, `git push`, branch create/delete, merge, `gh` PR/issue/release actions — there is no dedicated agent for this. Apply the same checklist every time: run `git status` + `git diff` (staged and unstaged) before committing, stage specific files by name (never `-A`/`.`), scan changed files for anything that smells like a credential/API key/`.env` content/token before staging and stop to flag it if found, write commit messages that explain *why*, and treat every push/force-push/branch-delete/history-rewrite as needing its own explicit authorization — never force-push to `main`/`master`, never skip hooks or bypass signing, never amend a commit that's already been pushed.
- This applies when the master fleet's work is ready to ship too: the master and slaves never touch repo/remote state themselves (see `master.md` and `slave.md`) — they report back to you when work is ready to commit/push/PR, and you run it directly using the checklist above.

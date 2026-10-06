# Claude Code setup

Commands to rebuild my Claude Code environment on a new machine.

## 1. Global instructions

```bash
mkdir -p ~/.claude
curl -fsSL https://raw.githubusercontent.com/dominictedrow/journey/main/claude-setup/CLAUDE.md -o ~/.claude/CLAUDE.md
```

`CLAUDE.md` refers to custom agents (`master`, `slave`, `qa`, `code-reviewer`, `research`) that live in `~/.claude/agents/`. Copy those over separately.

## 2. Plugin marketplaces

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin marketplace add haider-nawaz/liquid-glass-skill
claude plugin marketplace add DietrichGebert/ponytail
claude plugin marketplace add pbakaus/impeccable
```

## 3. Plugins

```bash
claude plugin install figma@claude-plugins-official
claude plugin install liquid-glass@liquid-glass-skills
claude plugin install clangd-lsp@claude-plugins-official
claude plugin disable clangd-lsp@claude-plugins-official   # installed but kept off
```

## 4. Standalone skills

```bash
# gstack (browse, connect-chrome, qa-only, office-hours, plan-ceo-review, ios-clean, ...); needs bun
git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack && ./setup && cd -

# graphify (needs uv)
uv tool install graphifyy
graphify install --platform claude

# find-skills
npx skills add vercel-labs/skills --skill find-skills -g
```

Most gstack skills are switched off with `skillOverrides` in `~/.claude/settings.json`. Only `browse`, `connect-chrome`, `ios-clean`, `office-hours`, `plan-ceo-review` and `qa-only` are left on.

The standalone `liquid-glass` skill (the CSS/SVG one, not the plugin) has no recorded source. Copy `~/.claude/skills/liquid-glass/` from the old machine.

## 5. Connectors (not installed from the CLI)

- Gmail, Google Calendar, Google Drive, Claude Docs: connect them at claude.ai → Settings → Connectors. They follow your account.
- Claude in Chrome: install the Chrome extension and sign in.

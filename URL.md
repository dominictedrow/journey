# Auto-Start on Login (localhost only)

Runs the app at `http://localhost:5001` automatically every time you log into macOS — no terminal needed.

---

## Status

| Step | Status |
|---|---|
| logs/ directory | ✅ Done |
| com.newsbot.flask.plist | ✅ Written to project root |
| Flask LaunchAgent | ⬜ Copy + load (two commands below) |

---

## Step 1 — Copy the LaunchAgent into place

The plist is pre-written at the project root (heredoc pasting in Terminal is unreliable — indentation causes the terminator to be treated as content). Just copy it:

```bash
cp /Users/dominictedrow/projects/news_bot/com.newsbot.flask.plist ~/Library/LaunchAgents/com.newsbot.flask.plist
```

---

## Step 2 — Load it

```bash
launchctl load ~/Library/LaunchAgents/com.newsbot.flask.plist
```

The app starts immediately and will restart automatically on every login.

---

## Step 3 — Bookmark it

Open `http://localhost:5001` in your browser and bookmark it (or set as homepage).

---

## Managing the service

| Action | Command |
|---|---|
| Stop | `launchctl unload ~/Library/LaunchAgents/com.newsbot.flask.plist` |
| Start | `launchctl load ~/Library/LaunchAgents/com.newsbot.flask.plist` |
| Restart | `launchctl kickstart -k gui/$(id -u)/com.newsbot.flask` |
| View logs | `tail -f /Users/dominictedrow/projects/news_bot/logs/flask.log` |
| View errors | `tail -f /Users/dominictedrow/projects/news_bot/logs/flask_error.log` |

---

## Troubleshooting

**Site not loading after login:** Flask takes ~10s to start. Wait and refresh.

**Port 5001 already in use:** macOS AirPlay Receiver uses 5000 — 5001 is usually free. If something else has 5001, change `PORT` to `5002` in the plist and in `config/settings.py`.

**App crashed:** Check `logs/flask_error.log`. `KeepAlive` will restart it automatically.

**To update the plist:** Edit `com.newsbot.flask.plist` in the project root, then re-run Steps 1 and 2 (unload first: `launchctl unload ~/Library/LaunchAgents/com.newsbot.flask.plist`).

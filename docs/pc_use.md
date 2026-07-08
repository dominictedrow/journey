# Running the Stack on a Separate Windows PC (24/7 host)

Step-by-step process for turning a **separate physical Windows computer** (not a VM, not WSL on
the Mac) into a standalone 24/7 backend host — a drop-in replacement for the Beelink mini PC
described in [`deploy.md`](deploy.md), until that hardware is in hand.

This is LAN-only: reachable from any device on the home network, not from the outside internet.
Reaching it from outside the LAN (a VPN into the home network is the safer option over router
port-forwarding, given `WALLET_PRIVATE_KEY` is involved) is a separate, not-yet-covered step.

## A. Prep the physical machine

1. Windows 10 (2004+) or Windows 11, any edition that supports WSL2 (Home works).
2. Plug it into Ethernet if possible (more reliable than Wi-Fi for a 24/7 box).
3. Power settings → set Sleep/Hibernate to "Never" for both "on battery" and "plugged in"
   (Settings → System → Power & battery).
4. Enable auto sign-in to a Windows account (`netplwiz` → uncheck "users must enter a password" →
   set the account) so Docker can restart itself after a reboot without anyone physically logging
   in.

## B. Install prerequisites

5. Install WSL2: open PowerShell as Administrator → `wsl --install` → reboot when prompted.
6. Install **Docker Desktop for Windows** (docker.com/products/docker-desktop), using the WSL2
   backend (default on modern installers). In Docker Desktop → Settings → General, enable
   **"Start Docker Desktop when you sign in."**
7. Install **Git for Windows** (git-scm.com).

## C. Get the code

8. Since the repo is private on GitHub, authenticate first: install the `gh` CLI and run
   `gh auth login`, or generate a Personal Access Token to use as the HTTPS password.
9. Clone it:
   ```powershell
   git clone https://github.com/dominictedrow/polymarket_copy.git
   cd polymarket_copy
   ```

## D. Configure secrets

10. `copy docker\.env.example .env`
11. Edit `.env` and fill in `POLYGON_RPC_URLS` and the CLOB credentials by typing/pasting them
    directly on this machine — never round-trip them through Google Drive or similar, since
    `WALLET_PRIVATE_KEY` is a live signing key. Leave `EXECUTION_MODE=paper` and
    `LIVE_TRADING_ARMED=false` unless you're deliberately arming live trading.

## E. Start and verify

12. ```powershell
    docker compose -f docker/docker-compose.yml up -d --build
    ```
13. Check it's healthy: `curl http://localhost:8000/health` and open `http://localhost:3000` in a
    browser on that machine.

## F. Make it survive reboots

14. All five services already declare `restart: unless-stopped` in `docker-compose.yml`, so once
    Docker's daemon is running they come back up on their own after a crash or reboot — the only
    thing that has to happen on boot is Docker Desktop itself launching, which step 6's "start on
    sign-in" setting plus step 4's auto-login covers.
15. Windows Update can still force a reboot on its own schedule — check Settings → Windows Update
    → "Active hours" and set them to cover when you'd notice if the box went down, so forced
    restarts don't happen at an inconvenient time.

## G. Network access from your Mac / phone / other devices

16. Find its LAN IP: `ipconfig` (look for IPv4 Address).
17. Set a DHCP reservation for that IP on your router, so it doesn't change later and break your
    bookmarks.
18. Confirm Windows Firewall allows inbound on 8000 and 3000 — Docker Desktop usually adds these
    automatically when it publishes the ports; if a browser on another device can't reach it, add
    inbound rules manually (Windows Defender Firewall → Advanced Settings → Inbound Rules → New
    Rule → Port → TCP 8000,3000 → Allow).
19. From any other device on the same LAN: `http://<windows-ip>:3000` for the dashboard, or point
    the macOS desktop app's Settings → backend URL at `http://<windows-ip>:8000`.

## H. Ongoing

20. To pull code updates later: `git pull` then
    `docker compose -f docker/docker-compose.yml up -d --build`.
21. To check on it remotely: `docker compose -f docker/docker-compose.yml logs -f api worker beat`.

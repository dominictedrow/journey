"""One-time, human-run Gmail OAuth consent for the `cold_email` skill.

Run directly (not as a module):

    python scripts/auth_gmail.py

Requires a Google Cloud OAuth client (Desktop app type) with the Gmail API
enabled, downloaded as a JSON file and placed at `.auth/gmail_client_secret.json`
(gitignored — never committed). Create one at
https://console.cloud.google.com/apis/credentials if you don't have one yet.

Requests only the `gmail.compose` scope — enough to create drafts, nothing
that can read Dom's inbox. Opens a local browser consent flow; on success,
stores the resulting token in the macOS Keychain via `skills._lib.secrets`
under the same key (`gmail_token`) that `skills.cold_email` reads.

Requires the optional 'gmail' extra: pip install -e ".[gmail]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills._lib import secrets

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
CLIENT_SECRET_PATH = Path(__file__).resolve().parent.parent / ".auth" / "gmail_client_secret.json"


def main():
    if not CLIENT_SECRET_PATH.is_file():
        print(
            f"No OAuth client found at {CLIENT_SECRET_PATH}.\n"
            "Create a Desktop-app OAuth client with the Gmail API enabled at\n"
            "https://console.cloud.google.com/apis/credentials, download its\n"
            "JSON, and save it at that path, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "Missing the optional 'gmail' extra. Run: pip install -e \".[gmail]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0)

    secrets.set("gmail_token", creds.to_json())
    print("Gmail token saved to Keychain. `cold_email --mode draft` will pick it up.", file=sys.stderr)


if __name__ == "__main__":
    main()

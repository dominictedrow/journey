"""Evening DB backup (D-017). Run with the paper profile so failures push to Dom:

QT_PROFILE=paper uv run python scripts/db_backup.py
"""

import sys

from qt.db.backup import main

if __name__ == "__main__":
    sys.exit(main())

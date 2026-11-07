import os
import sys


def resource_path(rel):
    """Resolve a project-relative path, honoring PyInstaller's bundle dir when frozen."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
BLS_API_KEY = os.environ.get("BLS_API_KEY", "")
PORT = int(os.environ.get("PORT", 5000))

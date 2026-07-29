"""macOS Keychain access via the `security` CLI — no secrets ever touch the repo or disk in plaintext."""

import subprocess

_SERVICE_PREFIX = "cwork"


def get(account, service=_SERVICE_PREFIX):
    result = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def set(account, value, service=_SERVICE_PREFIX):
    subprocess.run(
        ["security", "add-generic-password", "-a", account, "-s", service, "-w", value, "-U"],
        capture_output=True,
        text=True,
        check=True,
    )


def delete(account, service=_SERVICE_PREFIX):
    subprocess.run(
        ["security", "delete-generic-password", "-a", account, "-s", service],
        capture_output=True,
        text=True,
    )

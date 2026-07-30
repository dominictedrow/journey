"""urllib GET/JSON with an outbound domain allowlist — unexpected hosts raise, not connect."""

import json
import urllib.request
from urllib.parse import urlparse

ALLOWED_HOSTS = {
    "overpass-api.de",
}

_TIMEOUT = 15


class DisallowedHostError(Exception):
    pass


def _check_host(url):
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise DisallowedHostError(f"{host!r} is not in net.ALLOWED_HOSTS")
    return host


def get(url, headers=None):
    _check_host(url)
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "cwork/0.1"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def get_json(url, headers=None):
    return json.loads(get(url, headers=headers))

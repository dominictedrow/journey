import logging

import httpx
import pytest

from qt.common import alerts
from qt.common import logging as qtlog
from qt.common.alerts import Alerter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def cfg(settings):
    return settings.config.app.alerts.model_copy(update={"enabled": True})


def make(cfg, status=200, raise_exc=None):
    posts, notes, sunk = [], [], []

    def post(url, json, timeout):
        if raise_exc:
            raise raise_exc
        posts.append(json)
        return httpx.Response(status)

    def note(title, body):
        notes.append(title)
        return True

    clock = Clock()
    a = Alerter(
        cfg, "topic-x", http_post=post, os_notify=note, sink=lambda *r: sunk.append(r), clock=clock
    )
    return a, posts, notes, sunk, clock


def test_log_error_pushes_exactly_once(settings, cfg, tmp_path):
    a, posts, *_ = make(cfg)
    qtlog.setup(settings, log_dir=tmp_path, alerter=a, console=False)
    logging.getLogger("qt.unit").error("feed_failed: fred 500")
    logging.getLogger("qt.unit").warning("just_a_warning")
    assert len(posts) == 1
    assert posts[0]["priority"] == cfg.priority.critical
    assert "feed_failed" in posts[0]["title"] and posts[0]["topic"] == "topic-x"


def test_muted_library_error_is_logged_not_pushed(settings, cfg, tmp_path):
    a, posts, *_ = make(cfg)
    qtlog.setup(settings, log_dir=tmp_path, alerter=a, console=False)
    logging.getLogger("yfinance").error("$AABA: possibly delisted; no timezone found")
    logging.getLogger("yfinance").error("$WYND: possibly delisted; no timezone found")
    assert posts == []
    logged = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert "$AABA: possibly delisted" in logged


def test_mute_is_narrow(settings, cfg, tmp_path):
    a, posts, *_ = make(cfg)
    qtlog.setup(settings, log_dir=tmp_path, alerter=a, console=False)
    logging.getLogger("yfinance").error("$SPY: possibly delisted; no timezone found")
    logging.getLogger("qt.collect.yfinance").error("$AABA: other failure")
    logging.getLogger("other").error("$AABA: possibly delisted")
    assert len(posts) == 3


def test_identical_alerts_coalesce_then_follow_up(cfg):
    a, posts, *_rest, clock = make(cfg)
    for _ in range(5):
        a.alert("critical", "e", "boom")
    assert len(posts) == 1
    clock.t += cfg.coalesce_window_s + 1
    a.alert("critical", "e", "boom")
    assert len(posts) == 3
    assert "x4 more" in posts[1]["title"]


def test_flush_sends_pending_follow_ups(cfg):
    a, posts, *_ = make(cfg)
    a.alert("critical", "e", "boom")
    a.alert("critical", "e", "boom")
    a.flush()
    assert len(posts) == 2 and "x1 more" in posts[1]["title"]


def test_different_alerts_not_coalesced(cfg):
    a, posts, *_ = make(cfg)
    a.alert("info", "trade", "BUY SPY")
    a.alert("info", "trade", "BUY QQQ")
    assert len(posts) == 2 and posts[0]["priority"] == cfg.priority.info


@pytest.mark.parametrize("kw", [{"status": 500}, {"raise_exc": httpx.ConnectError("down")}])
def test_ntfy_down_falls_back_and_never_raises(cfg, kw):
    a, posts, notes, sunk, _ = make(cfg, **kw)
    assert a.alert("critical", "e", "boom") is False
    assert notes == ["boom"]
    assert sunk[-1][-1] is False  # recorded delivered=false


def test_disabled_sends_nothing_but_records(cfg):
    a, posts, notes, sunk, _ = make(cfg.model_copy(update={"enabled": False}))
    a.alert("critical", "e", "boom")
    assert posts == [] and notes == [] and len(sunk) == 1


def test_sink_failure_does_not_raise(cfg):
    a, posts, *_ = make(cfg)
    a.sink = lambda *r: 1 / 0
    assert a.alert("critical", "e", "boom") is True


def test_module_alert_without_install_is_safe():
    alerts.install(None)
    assert alerts.alert("info", "e", "t") is False

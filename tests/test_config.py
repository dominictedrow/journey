import pytest
import yaml

from qt.common import config
from qt.common.config import ConfigError, deep_merge
from qt.common.paths import CONFIG_DIR


def _copy_configs(tmp_path):
    d = tmp_path / "configs"
    (d / "profiles").mkdir(parents=True)
    for p in config.visible_files(CONFIG_DIR, "*.yaml"):
        (d / p.name).write_text(p.read_text())
    for p in config.visible_files(CONFIG_DIR / "profiles", "*.yaml"):
        (d / "profiles" / p.name).write_text(p.read_text())
    return d


def test_profiles_load_and_override_only_their_keys():
    dev, paper = config.load("dev"), config.load("paper")
    assert dev.config.app.logging.level == "DEBUG" and dev.config.app.alerts.enabled is False
    assert paper.config.app.alerts.enabled is True
    assert dev.config.db == paper.config.db  # untouched keys identical


def test_unknown_profile_fails():
    with pytest.raises(ConfigError, match="unknown profile"):
        config.load("live")


def test_unknown_key_fails(tmp_path):
    d = _copy_configs(tmp_path)
    data = yaml.safe_load((d / "backup.yaml").read_text())
    data["keep_dumpz"] = 3
    (d / "backup.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="keep_dumpz"):
        config.load("dev", config_dir=d)


def test_missing_key_fails(tmp_path):
    d = _copy_configs(tmp_path)
    data = yaml.safe_load((d / "backup.yaml").read_text())
    del data["keep_dumps"]
    (d / "backup.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="keep_dumps"):
        config.load("dev", config_dir=d)


def test_profile_cannot_invent_keys():
    with pytest.raises(ConfigError, match="does not exist"):
        deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})


def test_appledouble_files_ignored(tmp_path):
    d = _copy_configs(tmp_path)
    (d / "._db.yaml").write_bytes(b"\x00\x05\x16\x07\xb0junk")
    config.load("dev", config_dir=d)


def test_config_hash_stable_and_sensitive(tmp_path, monkeypatch):
    a, b = config.load("dev"), config.load("dev")
    assert a.config_hash == b.config_hash
    assert config.load("paper").config_hash != a.config_hash
    monkeypatch.setenv("NTFY_TOPIC", "something-else")
    monkeypatch.setenv("POSTGRES_PASSWORD", "other")
    assert config.load("dev").config_hash == a.config_hash  # secrets never affect the hash


def test_paper_guard(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "AKLIVEKEY123")
    with pytest.raises(ConfigError, match="PAPER"):
        config.load("dev")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKPAPER123")
    config.load("dev")


def test_paper_url_is_hardcoded():
    assert config.ALPACA_PAPER_BASE_URL == "https://paper-api.alpaca.markets"

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from nonebot import logger

from .config import Config


@dataclass(frozen=True)
class RuntimeSettings:
    enabled: bool = True
    reply_when_detected: bool = True
    reply: str = "检测到疑似拍屏照片，请尽量发送截图或原图。"
    score_threshold: float = 0.62
    group_whitelist: tuple[int, ...] = ()
    group_blacklist: tuple[int, ...] = ()
    debug: bool = False

    @classmethod
    def from_plugin_config(cls, config: Config) -> "RuntimeSettings":
        return cls(
            enabled=config.paiping_enabled,
            reply_when_detected=config.paiping_reply_when_detected,
            reply=config.paiping_reply,
            score_threshold=config.paiping_score_threshold,
            group_whitelist=tuple(config.paiping_group_whitelist),
            group_blacklist=tuple(config.paiping_group_blacklist),
            debug=config.paiping_debug,
        )

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        defaults: "RuntimeSettings",
    ) -> "RuntimeSettings":
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            reply_when_detected=_as_bool(
                data.get("reply_when_detected"),
                defaults.reply_when_detected,
            ),
            reply=_as_text(data.get("reply"), defaults.reply),
            score_threshold=_as_float(
                data.get("score_threshold"),
                defaults.score_threshold,
                low=0.0,
                high=1.0,
            ),
            group_whitelist=_as_int_tuple(data.get("group_whitelist"), defaults.group_whitelist),
            group_blacklist=_as_int_tuple(data.get("group_blacklist"), defaults.group_blacklist),
            debug=_as_bool(data.get("debug"), defaults.debug),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reply_when_detected": self.reply_when_detected,
            "reply": self.reply,
            "score_threshold": self.score_threshold,
            "group_whitelist": list(self.group_whitelist),
            "group_blacklist": list(self.group_blacklist),
            "debug": self.debug,
        }


class RuntimeConfigStore:
    def __init__(self, path: Path, defaults: RuntimeSettings) -> None:
        self.path = path
        self.defaults = defaults
        self._settings = self._load_or_create()

    def get(self) -> RuntimeSettings:
        return self._settings

    def reload(self) -> RuntimeSettings:
        self._settings = self._load_or_create()
        return self._settings

    def update(self, **changes: Any) -> RuntimeSettings:
        self._settings = replace(self._settings, **changes)
        self._write(self._settings)
        return self._settings

    def _load_or_create(self) -> RuntimeSettings:
        if not self.path.exists():
            self._write_default()

        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("paiping: failed to read runtime config {}, use defaults: {}", self.path, exc)
            return self.defaults

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            logger.warning("paiping: runtime config {} is not a mapping, use defaults", self.path)
            return self.defaults

        settings = RuntimeSettings.from_mapping(raw, self.defaults)
        missing_keys = set(self.defaults.to_mapping()) - set(raw)
        if missing_keys:
            self._write(settings)
        return settings

    def _write_default(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "# nonebot-plugin-paiping runtime config\n"
            "# SuperUser commands will update this file automatically.\n"
            "# score_threshold: lower is more sensitive, higher is stricter.\n"
            f"{yaml.safe_dump(self.defaults.to_mapping(), allow_unicode=True, sort_keys=False)}"
        )
        self.path.write_text(text, encoding="utf-8")
        logger.info("paiping: created runtime config {}", self.path)

    def _write(self, settings: RuntimeSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        text = yaml.safe_dump(settings.to_mapping(), allow_unicode=True, sort_keys=False)
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(self.path)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enable", "enabled", "开", "开启"}:
            return True
        if lowered in {"0", "false", "no", "off", "disable", "disabled", "关", "关闭"}:
            return False
    return default


def _as_text(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _as_float(value: Any, default: float, *, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _as_int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple, set)):
        return default

    result: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in result:
            result.append(number)
    return tuple(result)


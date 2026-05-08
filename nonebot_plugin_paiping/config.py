from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    paiping_config_file: Path = Path("data/paiping/config.yml")
    paiping_enabled: bool = True
    paiping_priority: int = 10
    paiping_reply_when_detected: bool = True
    paiping_reply: str = "检测到疑似拍屏照片，请尽量发送截图或原图。"
    paiping_score_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    paiping_group_whitelist: list[int] = Field(default_factory=list)
    paiping_group_blacklist: list[int] = Field(default_factory=list)
    paiping_max_images_per_message: int = Field(default=4, ge=1, le=20)
    paiping_max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=256 * 1024)
    paiping_request_timeout: float = Field(default=15.0, ge=1.0)
    paiping_min_image_side: int = Field(default=160, ge=64)
    paiping_debug: bool = False

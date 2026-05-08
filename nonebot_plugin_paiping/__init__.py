from __future__ import annotations

import base64
from typing import Any, Iterable

import cv2
import httpx
import numpy as np
from nonebot import get_plugin_config, logger, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, Event, GroupMessageEvent, Message, MessageSegment
from nonebot.exception import ActionFailed
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

from .config import Config
from .detector import DetectionResult, detect_screen_photo
from .runtime_config import RuntimeConfigStore, RuntimeSettings

__version__ = "0.1.0"

__plugin_meta__ = PluginMetadata(
    name="拍屏检测",
    description="检测群聊中直接拍摄屏幕的图片，并提醒发送截图或原图。",
    usage=(
        "群聊发送图片时自动检测。\n"
        "SuperUser 命令：/拍屏 状态、/拍屏 开、/拍屏 关、"
        "/拍屏 提醒 文案、/拍屏 阈值 0.62、/拍屏 本群开、/拍屏 本群关"
    ),
    type="application",
    homepage="https://github.com/nonebot/nonebot2",
    config=Config,
    supported_adapters={"~onebot.v11"},
    extra={"version": __version__},
)

plugin_config = get_plugin_config(Config)
runtime_config = RuntimeConfigStore(
    plugin_config.paiping_config_file,
    RuntimeSettings.from_plugin_config(plugin_config),
)

paiping_matcher = on_message(priority=plugin_config.paiping_priority, block=False)
paiping_command = on_command(
    "拍屏",
    aliases={"paiping", "拍屏检测"},
    permission=SUPERUSER,
    priority=plugin_config.paiping_priority,
    block=True,
)


@paiping_matcher.handle()
async def handle_group_images(bot: Bot, event: GroupMessageEvent) -> None:
    settings = runtime_config.get()
    if not settings.enabled:
        return

    group_id = int(event.group_id)
    if not _is_group_enabled(group_id, settings):
        return

    image_segments = list(_iter_image_segments(event.message))
    if not image_segments:
        return

    image_segments = image_segments[: plugin_config.paiping_max_images_per_message]

    async with httpx.AsyncClient(
        timeout=plugin_config.paiping_request_timeout,
        follow_redirects=True,
        headers={"User-Agent": "nonebot-plugin-paiping/0.1.0"},
    ) as client:
        for index, segment in enumerate(image_segments, start=1):
            image_bytes = await _read_image_bytes(bot, segment, client)
            if not image_bytes:
                continue

            image = _decode_image(image_bytes)
            if image is None:
                logger.debug("paiping: failed to decode image in group {}", group_id)
                continue

            result = detect_screen_photo(
                image,
                threshold=settings.score_threshold,
                min_size=plugin_config.paiping_min_image_side,
            )
            _log_result(group_id, index, result, settings)

            if result.is_screen_photo and settings.reply_when_detected:
                await paiping_matcher.send(_format_reply(result, settings))
                return


def _iter_image_segments(message: Message) -> Iterable[MessageSegment]:
    for segment in message:
        if segment.type == "image":
            yield segment


@paiping_command.handle()
async def handle_paiping_command(event: Event, args: Message = CommandArg()) -> None:
    raw_arg = args.extract_plain_text().strip()
    if not raw_arg or raw_arg in {"状态", "status"}:
        await paiping_command.finish(_format_status(event))

    command, value = _split_command(raw_arg)
    command = command.lower()

    if command in {"帮助", "help"}:
        await paiping_command.finish(_command_help())

    if command in {"开", "开启", "on", "enable", "enabled", "全局开"}:
        settings = runtime_config.update(enabled=True)
        await paiping_command.finish(f"拍屏检测已开启。\n{_format_brief_status(settings)}")

    if command in {"关", "关闭", "off", "disable", "disabled", "全局关"}:
        settings = runtime_config.update(enabled=False)
        await paiping_command.finish(f"拍屏检测已关闭。\n{_format_brief_status(settings)}")

    if command in {"回复", "通知", "提醒开关"}:
        enabled = _parse_bool(value)
        if enabled is None:
            await paiping_command.finish("用法：/拍屏 回复 开 或 /拍屏 回复 关")
        settings = runtime_config.update(reply_when_detected=enabled)
        state = "开启" if settings.reply_when_detected else "关闭"
        await paiping_command.finish(f"拍屏提醒回复已{state}。")

    if command in {"提醒", "文案", "reply"}:
        if not value:
            await paiping_command.finish(f"当前提醒：{runtime_config.get().reply}")
        settings = runtime_config.update(reply=value)
        await paiping_command.finish(f"拍屏提醒已更新：{settings.reply}")

    if command in {"阈值", "threshold"}:
        try:
            threshold = float(value)
        except ValueError:
            await paiping_command.finish("用法：/拍屏 阈值 0.62，范围 0 到 1，越低越敏感。")
        if not 0.0 <= threshold <= 1.0:
            await paiping_command.finish("阈值范围是 0 到 1，越低越敏感。")
        settings = runtime_config.update(score_threshold=threshold)
        await paiping_command.finish(f"拍屏检测阈值已更新为 {settings.score_threshold:.2f}。")

    if command in {"调试", "debug"}:
        enabled = _parse_bool(value)
        if enabled is None:
            await paiping_command.finish("用法：/拍屏 调试 开 或 /拍屏 调试 关")
        settings = runtime_config.update(debug=enabled)
        state = "开启" if settings.debug else "关闭"
        await paiping_command.finish(f"拍屏调试输出已{state}。")

    if command in {"本群开", "当前群开", "group-on"}:
        group_id = _event_group_id(event)
        if group_id is None:
            await paiping_command.finish("这个命令只能在群聊里使用。")
        settings = _enable_group(group_id)
        await paiping_command.finish(f"本群拍屏检测已开启。\n{_format_group_lists(settings)}")

    if command in {"本群关", "当前群关", "group-off"}:
        group_id = _event_group_id(event)
        if group_id is None:
            await paiping_command.finish("这个命令只能在群聊里使用。")
        settings = _disable_group(group_id)
        await paiping_command.finish(f"本群拍屏检测已关闭。\n{_format_group_lists(settings)}")

    if command in {"重载", "reload"}:
        settings = runtime_config.reload()
        await paiping_command.finish(f"拍屏 YAML 配置已重载。\n{_format_brief_status(settings)}")

    await paiping_command.finish(_command_help())


def _is_group_enabled(group_id: int, settings: RuntimeSettings) -> bool:
    whitelist = set(settings.group_whitelist)
    blacklist = set(settings.group_blacklist)
    if whitelist and group_id not in whitelist:
        return False
    return group_id not in blacklist


async def _read_image_bytes(
    bot: Bot,
    segment: MessageSegment,
    client: httpx.AsyncClient,
) -> bytes | None:
    url = await _resolve_image_url(bot, segment)
    if not url:
        logger.debug("paiping: image segment has no downloadable url: {}", segment.data)
        return None

    if url.startswith("base64://"):
        return _decode_base64_url(url)

    if not url.startswith(("http://", "https://")):
        logger.debug("paiping: unsupported image url scheme: {}", url[:48])
        return None

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > plugin_config.paiping_max_image_bytes:
                    logger.debug("paiping: image exceeds max bytes, skip")
                    return None
            return bytes(content)
    except httpx.HTTPError as exc:
        logger.debug("paiping: failed to download image: {}", exc)
        return None


async def _resolve_image_url(bot: Bot, segment: MessageSegment) -> str | None:
    raw_url = segment.data.get("url")
    if isinstance(raw_url, str) and raw_url:
        return raw_url

    file_id = segment.data.get("file")
    if not file_id:
        return None

    try:
        image_info: dict[str, Any] = await bot.get_image(file=file_id)
    except (ActionFailed, RuntimeError, KeyError, TypeError) as exc:
        logger.debug("paiping: get_image failed: {}", exc)
        return None

    for key in ("url", "file"):
        value = image_info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _decode_base64_url(url: str) -> bytes | None:
    try:
        return base64.b64decode(url.removeprefix("base64://"), validate=False)
    except (ValueError, base64.binascii.Error) as exc:
        logger.debug("paiping: failed to decode base64 image: {}", exc)
        return None


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _format_reply(result: DetectionResult, settings: RuntimeSettings) -> str:
    if not settings.debug:
        return settings.reply

    feature_text = ", ".join(
        f"{name}={value:.2f}" for name, value in sorted(result.features.items())
    )
    reason_text = "；".join(result.reasons)
    return (
        f"{settings.reply}\n"
        f"拍屏分数：{result.score:.2f}/{result.threshold:.2f}\n"
        f"特征：{feature_text}\n"
        f"依据：{reason_text}"
    )


def _log_result(
    group_id: int,
    index: int,
    result: DetectionResult,
    settings: RuntimeSettings,
) -> None:
    log_text = (
        f"paiping: group={group_id} image={index} "
        f"detected={result.is_screen_photo} score={result.score:.3f} "
        f"features={result.features}"
    )
    if settings.debug:
        logger.info(log_text)
    else:
        logger.debug(log_text)


def _split_command(raw_arg: str) -> tuple[str, str]:
    parts = raw_arg.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on", "enable", "enabled", "开", "开启"}:
        return True
    if lowered in {"0", "false", "no", "off", "disable", "disabled", "关", "关闭"}:
        return False
    return None


def _event_group_id(event: Event) -> int | None:
    group_id = getattr(event, "group_id", None)
    if group_id is None:
        return None
    try:
        return int(group_id)
    except (TypeError, ValueError):
        return None


def _enable_group(group_id: int) -> RuntimeSettings:
    settings = runtime_config.get()
    blacklist = tuple(group for group in settings.group_blacklist if group != group_id)
    whitelist = settings.group_whitelist
    if whitelist and group_id not in whitelist:
        whitelist = (*whitelist, group_id)
    return runtime_config.update(group_whitelist=whitelist, group_blacklist=blacklist)


def _disable_group(group_id: int) -> RuntimeSettings:
    settings = runtime_config.get()
    whitelist = tuple(group for group in settings.group_whitelist if group != group_id)
    blacklist = settings.group_blacklist
    if group_id not in blacklist:
        blacklist = (*blacklist, group_id)
    return runtime_config.update(group_whitelist=whitelist, group_blacklist=blacklist)


def _format_status(event: Event) -> str:
    settings = runtime_config.get()
    group_id = _event_group_id(event)
    lines = [
        _format_brief_status(settings),
        _format_group_lists(settings),
        f"配置文件：{runtime_config.path}",
    ]
    if group_id is not None:
        state = "开启" if _is_group_enabled(group_id, settings) else "关闭"
        lines.insert(2, f"当前群：{group_id}，检测{state}")
    return "\n".join(lines)


def _format_brief_status(settings: RuntimeSettings) -> str:
    enabled = "开启" if settings.enabled else "关闭"
    reply = "开启" if settings.reply_when_detected else "关闭"
    debug = "开启" if settings.debug else "关闭"
    return (
        f"全局：{enabled}；提醒回复：{reply}；"
        f"阈值：{settings.score_threshold:.2f}；调试：{debug}"
    )


def _format_group_lists(settings: RuntimeSettings) -> str:
    whitelist = _format_group_list(settings.group_whitelist)
    blacklist = _format_group_list(settings.group_blacklist)
    return f"群白名单：{whitelist}；群黑名单：{blacklist}"


def _format_group_list(groups: tuple[int, ...]) -> str:
    if not groups:
        return "空"
    return "、".join(str(group) for group in groups)


def _command_help() -> str:
    return (
        "拍屏检测命令：\n"
        "/拍屏 状态\n"
        "/拍屏 开 或 /拍屏 关\n"
        "/拍屏 本群开 或 /拍屏 本群关\n"
        "/拍屏 提醒 检测到疑似拍屏，请发截图或原图\n"
        "/拍屏 回复 开 或 /拍屏 回复 关\n"
        "/拍屏 阈值 0.62\n"
        "/拍屏 调试 开 或 /拍屏 调试 关\n"
        "/拍屏 重载"
    )

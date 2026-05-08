# nonebot-plugin-paiping

一个用于检测群聊拍屏照片的 NoneBot2 插件。插件会监听 OneBot V11 群消息中的图片，下载后使用 OpenCV 提取摩尔纹、色彩子像素噪声、滚动条纹、屏幕边框等特征，并在判定为拍屏时发送提醒。

## 安装

```bash
pip install -e .
```

然后在 NoneBot 项目中加载：

```python
nonebot.load_plugin("nonebot_plugin_paiping")
```

## YAML 配置

插件启动时会自动生成运行时配置：

```text
data/paiping/config.yml
```

生成内容类似：

```yaml
enabled: true
reply_when_detected: true
reply: 检测到疑似拍屏照片，请尽量发送截图或原图。
score_threshold: 0.62
group_whitelist: []
group_blacklist: []
debug: false
```

这些配置会被 SuperUser 命令自动读写。也可以手动改 YAML，然后执行 `/拍屏 重载`。

`.env` 仍可设置启动级配置和 YAML 初始默认值：

```dotenv
PAIPING_CONFIG_FILE=data/paiping/config.yml
PAIPING_ENABLED=true
PAIPING_SCORE_THRESHOLD=0.62
PAIPING_REPLY="检测到疑似拍屏照片，请尽量发送截图或原图。"
PAIPING_DEBUG=false
```

常用配置：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `PAIPING_CONFIG_FILE` | `data/paiping/config.yml` | 插件自动生成和读取的 YAML 配置路径 |
| `PAIPING_ENABLED` | `true` | 是否启用自动检测 |
| `PAIPING_SCORE_THRESHOLD` | `0.62` | 判定阈值，越低越敏感 |
| `PAIPING_REPLY_WHEN_DETECTED` | `true` | 检出后是否发送群提醒 |
| `PAIPING_GROUP_WHITELIST` | `[]` | 非空时只检测这些群 |
| `PAIPING_GROUP_BLACKLIST` | `[]` | 不检测这些群 |
| `PAIPING_MAX_IMAGES_PER_MESSAGE` | `4` | 单条消息最多检测几张图 |
| `PAIPING_MAX_IMAGE_BYTES` | `10485760` | 单张图最大下载字节数 |
| `PAIPING_DEBUG` | `false` | 回复和日志中输出评分细节 |

## SuperUser 命令

以下命令只有 NoneBot `SUPERUSERS` 可以使用：

```text
/拍屏 状态
/拍屏 开
/拍屏 关
/拍屏 本群开
/拍屏 本群关
/拍屏 提醒 检测到疑似拍屏，请发截图或原图
/拍屏 回复 开
/拍屏 回复 关
/拍屏 阈值 0.62
/拍屏 调试 开
/拍屏 调试 关
/拍屏 重载
```

别名：`/paiping`、`/拍屏检测`。

## 本地测试图片

之后你提供测试图时，可以直接运行：

```bash
python scripts/paiping_probe.py path\to\image_or_dir
```

脚本会输出每张图片的判定、总分和各个 OpenCV 特征分数。阈值可以这样临时指定：

```bash
python scripts/paiping_probe.py samples --threshold 0.58
```

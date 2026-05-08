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
model_enabled: false
model_file: data/paiping/model.json
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
/拍屏 模型 开
/拍屏 模型 关
/拍屏 模型 文件 data/paiping/model.json
/拍屏 调试 开
/拍屏 调试 关
/拍屏 重载
```

别名：`/paiping`、`/拍屏检测`。

## 检测原理

插件不是靠识别图片内容，而是把图片当成信号来分析。群消息里的图片会先下载并用 OpenCV 解码，然后提取一组拍摄屏幕常见的视觉特征，最后根据规则分数或二分类模型概率判断是否为拍屏。

主要特征包括：

| 特征 | 作用 |
| --- | --- |
| `frequency` / `local_frequency` | 检测屏幕像素网格、摩尔纹、局部周期纹理 |
| `chroma` | 检测相机拍屏时容易出现的彩色子像素噪声 |
| `banding` | 检测刷新率和快门造成的横向或纵向条纹 |
| `rectangle` / `screen_aspect` | 检测显示器、窗口、屏幕边框和宽屏比例 |
| `illumination` | 检测拍摄角度、暗角、反光导致的亮度不均匀 |
| `softness` | 检测拍屏常见的轻微失焦、运动模糊、压缩糊感 |
| `camera_artifact` | 综合判断镜头阴影、亮斑、光照痕迹等相机特征 |
| `display_content` | 判断是否存在类似桌面、窗口、图标、文字块的显示内容 |
| `overexposed_display` | 检测显示器局部过曝但仍保留屏幕结构的情况 |
| `screenshot_penalty` | 对干净截图、普通二次元图、自然图片等进行降权，降低误报 |

未启用模型时，插件会把这些特征按权重合成 `score`，默认 `score >= 0.62` 判定为拍屏。阈值越低越敏感，漏判更少但误报可能增加；阈值越高越保守。

启用模型后，插件仍然使用同一批 OpenCV 特征，但不再直接用规则总分判定，而是把特征输入 `data/paiping/model.json` 里的轻量线性二分类模型。模型输出 `model_probability`，概率超过模型自己的阈值时判定为拍屏。这样可以利用你的 `拍屏/正常` 数据集自动学习特征权重，比纯手写规则更适合你的群聊图片分布。

## 本地测试图片

之后你提供测试图时，可以直接运行：

```bash
python scripts/paiping_probe.py path\to\image_or_dir
```

脚本会输出每张图片的判定、总分和各个 OpenCV 特征分数。阈值可以这样临时指定：

```bash
python scripts/paiping_probe.py samples --threshold 0.58
```

## 训练二分类模型

如果你已经把样本分成 `拍屏` 和 `正常` 两个目录，例如：

```text
验证/
  拍屏/
  正常/
```

可以训练一个轻量二分类模型：

```bash
python scripts/train_paiping_model.py --dataset 验证 --output data/paiping/model.json --workers 8 --cache data/paiping/feature_cache.jsonl
```

`--workers 0` 表示使用全部 CPU 核心；`--cache` 会缓存 OpenCV 特征，后续重复训练会快很多。

模型不依赖 `sklearn` 或 `torch`，只使用现有 OpenCV 特征和 `numpy` 训练一个小型线性分类器。启用模型：

```yaml
model_enabled: true
model_file: data/paiping/model.json
```

也可以用命令启用：

```text
/拍屏 模型 开
```

本地测试模型概率：

```bash
python scripts/paiping_probe.py 验证 --model data/paiping/model.json
```

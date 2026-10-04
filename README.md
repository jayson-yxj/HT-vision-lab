# HT Vision Lab

独立的多人视觉理解实验项目，与 `HT-voice-lab` 解耦。

## 当前状态

第一版会话内人脸轨迹已经可运行：

1. 使用 YuNet 检测人脸和五点位置；
2. 使用 SFace 特征、位置重叠和时间连续性建立短时轨迹；
3. 使用保守的完整链接聚类合并跨镜头轨迹，生成 `Face-01/02/03…`；
4. 输出 `visual_tracks.json`、每个人的原始帧证据和带原音频的标注视频；
5. 保留活跃说话人与 A/B/C/D 绑定字段，当前为空，下一阶段由 LR-ASD 填充。

所有模型在本地运行。模型 URL 固定到 OpenCV Zoo 的具体 Git revision，下载后校验文件大小和 SHA-256。YuNet 权重为 MIT，SFace 权重为 Apache-2.0。

## 安装

```bash
cd /home/sunteng/Desktop/HighTorque_vision/HT-vision-lab
bash setup.sh
```

如果当前 Python 已安装 NumPy、SciPy 和 OpenCV，也可以只下载模型：

```bash
./lab fetch-models
./lab status
```

`models/`、`data/raw/` 和 `results/` 不进入 Git。

## 运行

```bash
./lab track-faces /path/to/input.mp4 \
  --output results/example
```

常用选项：

```text
--sample-fps 5                 每秒执行人脸检测的次数
--detection-threshold 0.70     YuNet 检测阈值
--min-face-size 24             忽略过小且无法可靠识别的人脸
--max-gap-seconds 0.8          允许轨迹跨越的短时检测缺口
--reid-threshold 0.55          无位置重叠时继续同一轨迹的特征阈值
--cluster-threshold 0.45       跨镜头轨迹合并阈值
--min-track-observations 3     过滤瞬时误检
--no-render                    跳过标注视频
```

检查已有结果：

```bash
./lab validate results/example/visual_tracks.json
```

## 输出

```text
results/example/
├── visual_tracks.json
├── annotated.mp4
└── evidence/
    ├── Face-01.jpg
    └── Face-02.jpg
```

正式协议位于 [`schemas/visual_tracks.schema.json`](schemas/visual_tracks.schema.json)。主要对象包括：

- `faces`：经过轨迹聚类后的会话内可见人物；
- `tracklets`：镜头内连续人脸轨迹；
- `observations`：采样帧、人脸框、五点位置、质量和匹配置信度；
- `active_speaker_segments`：预留的 LR-ASD 输出；
- `speaker_face_associations`：预留的 A/B/C/D 与 Face-ID 关联证据。

`faces` 是画面中的可见人物候选，不直接等于对话参与者。舞台远景、路人或静默听众可能使可见脸超过四张；下一阶段使用活跃说话人证据筛出最多四名对话参与者，不按数量强制删除视觉证据。

## 已验证样例

- 中文 480p、102 秒舞台片段：过滤无法可靠识别的极小脸后保留 7 个可见人物候选，近景人物跨镜头保持稳定；
- 英文 720p、60 秒四人片段：得到 4 个稳定身份、14 条跨镜头轨迹和 717 条人脸观察；
- 标注视频保留原始 AAC 音频，结构引用和时间范围检查通过。

## 后续顺序

1. 接入 LR-ASD，为每条人脸轨迹生成逐帧说话概率；
2. 将 LR-ASD 时间线与 Sortformer 的 A/B/C/D 时间线累计匹配；
3. 加入画外说话、遮挡和无法判断状态；
4. 验证完成后加入关键帧场景、人物姿态和二维空间关系；
5. 最后实现跨会话人物档案、摄像头实时处理和人工纠错。

语音基线固定为 `HT-voice-lab` 标签 `voice-baseline-2026-09-28`。

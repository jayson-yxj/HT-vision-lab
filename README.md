# HT Vision Lab

独立的多人视觉理解实验项目，与 `HT-voice-lab` 解耦。

## 当前状态

第一版会话内人脸轨迹和主动说话人检测已经可运行：

1. 使用 YuNet 检测人脸和五点位置；
2. 使用 SFace 特征、位置重叠和时间连续性建立短时轨迹；
3. 使用保守的完整链接聚类合并跨镜头轨迹，生成 `Face-01/02/03…`；
4. 输出 `visual_tracks.json`、每个人的原始帧证据和带原音频的标注视频；
5. 使用 LR-ASD TalkSet 权重融合嘴部运动和音频，为每张可见脸生成 25 FPS 的说话分数；
6. 默认按首版“通常不会同时说话”的约束，每 40 毫秒只保留得分最高的人脸；
7. 将 Sortformer 稳定后的 A/B/C/D 时间线与 Face-ID 做一对一重叠匹配，并保留全部候选证据。
8. 将视觉身份投影到语音侧匿名参与者协议，保留已有个人信息并追加可追溯的多模态证据。
9. 在发言时间线上区分 `visible / offscreen / occluded / unknown`，并用切镜检测避免把换镜误判成遮挡。

所有模型在本地运行。模型 URL 固定到上游仓库的具体 Git revision，下载后校验文件大小和 SHA-256。YuNet 和 LR-ASD 为 MIT，SFace 为 Apache-2.0；详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 安装

```bash
cd /home/sunteng/Desktop/HighTorque_vision/HT-vision-lab
bash setup.sh
```

主动说话人检测还需要系统中已有 PyTorch。当前机器可直接创建独立环境并下载固定版本权重：

```bash
bash setup-asd.sh
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
--reid-threshold 0.75          无位置重叠时继续同一轨迹的特征阈值
--cluster-threshold 0.45       跨镜头轨迹合并阈值
--min-track-observations 3     过滤瞬时误检
--no-render                    跳过标注视频
```

检查已有结果：

```bash
./lab validate results/example/visual_tracks.json
```

在已有轨迹上检测主动说话人：

```bash
./lab active-speaker results/example/visual_tracks.json \
  --model talkset \
  --device cuda
```

该命令原地更新 `visual_tracks.json` 并生成 `active_speaker.mp4`。使用 `--no-render` 跳过视频，使用 `--allow-overlap` 允许同一时刻存在多个说话人。

将 HT-voice-lab 的稳定说话人时间线绑定到 Face-ID：

```bash
./lab bind-speakers \
  results/example/visual_tracks.json \
  /home/sunteng/Desktop/HighTorque_vision/HT-voice-lab/results/example/speech_spans.json
```

输入必须是声纹稳定后的 `A/B/C/D`，不接受原始 `speaker0…speaker3` 槽位。命令按时间重叠求全局一对一最优解，并输出 `confirmed`、`candidate`、`ambiguous` 或 `offscreen`。如果音频和视频片段起点不同，可用 `--timeline-offset-ms` 将语音时间整体平移后再匹配。

在人物发言期间分类可见性：

```bash
./lab classify-visibility results/example/visual_tracks.json
```

对应人脸轨迹存在时为 `visible`；同一镜头内前后重新出现的短缺口为低置信度 `occluded`；确认身份发言但人脸不在画面时为 `offscreen`；身份绑定仍有冲突时保持 `unknown`。遮挡判定是一项基于轨迹和切镜的推断，不作为确定的人体遮挡检测结果。

将绑定结果投影到语音侧已有的匿名参与者和个人信息：

```bash
./lab project-participants \
  results/example/visual_tracks.json \
  /path/to/participant_context.json
```

默认生成 `multimodal_participant_context.json`。命令会先用说话人、时间和原文校验两份输入来自同一场对话；已确认的人脸关系写为 `confirmed`，冲突候选写为 `disputed`。原有姓名、属性观察和语音证据保持不变，未知字段继续为空。

## 输出

```text
results/example/
├── visual_tracks.json
├── multimodal_participant_context.json
├── annotated.mp4
├── active_speaker.mp4
└── evidence/
    ├── Face-01.jpg
    └── Face-02.jpg
```

正式协议位于 [`schemas/visual_tracks.schema.json`](schemas/visual_tracks.schema.json)。主要对象包括：

- `faces`：经过轨迹聚类后的会话内可见人物；
- `tracklets`：镜头内连续人脸轨迹；
- `observations`：采样帧、人脸框、五点位置、质量和匹配置信度；
- `active_speaker_scores`：每张可见脸每 40 毫秒的原始分数、概率和最终判定；
- `active_speaker_segments`：连续的可见说话人区间；
- `speaker_face_associations`：A/B/C/D 与 Face-ID 的状态、覆盖率、纯度、第二候选差距和原始重叠证据。
- `speaker_visibility_segments`：参与者在发言期间的可见、画外、疑似遮挡或未知状态。

融合人物协议位于 [`schemas/multimodal_participant_context.schema.json`](schemas/multimodal_participant_context.schema.json)，它复用语音侧的参与者、个人信息观察和证据对象，并增加 `visual_entities` 与人物—视觉身份关联。

`faces` 是画面中的可见人物候选，不直接等于对话参与者。舞台远景、路人、插入镜头或静默听众可能使可见脸超过四张；使用 `active_speaker_segments` 可以筛出实际发言身份，而不删除原始视觉证据。

## 已验证样例

- 中文 480p、30 秒片段：镜头切换后保持 4 个可见身份，LR-ASD 检出的 3 位实际发言者与已知 0–8、9–21、22 秒之后的顺序一致；
- 英文 720p、60 秒片段：从 6 个可见身份中筛出 4 个实际发言身份，同一时刻的静默人脸弱阳性已被排除；
- 中文绑定得到 `A→Face-01`、`B→Face-02`、`C→Face-03`，三项均为 `confirmed`；英文 `A/B` 确认绑定，`C` 因同时积累到两个 Face-ID 的显著证据而标为 `ambiguous`；
- 英文四人上下文投影得到 6 个视觉实体、2 条确认关联、2 条争议关联、5 个可见性事件和 13 条多模态证据；未在 60 秒片段中发言的 D 保持匿名且不产生视觉绑定；
- 中文三位发言者均为 `visible`；英文 A/B 为 `visible`，C 因身份冲突保持 `unknown`，系统没有把冲突错误改写为画外或遮挡；
- 两条样例都通过结构引用、时间范围和单一说话者约束检查，主动说话视频保留原始音频。

## 后续顺序

1. 加入关键帧场景、人物姿态和二维空间关系；
2. 建立跨会话人物档案、摄像头实时处理和人工纠错。

语音基线固定为 `HT-voice-lab` 标签 `voice-baseline-2026-09-28`。

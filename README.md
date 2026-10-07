# HT Vision Lab

独立的多人视觉理解实验项目，与 `HT-voice-lab` 解耦。

## 当前状态

第一版会话内视觉理解流水线已经可运行：

1. 使用 YuNet 检测人脸和五点位置；
2. 使用 SFace 特征、位置重叠和时间连续性建立短时轨迹；
3. 使用人脸特征与同帧互斥约束合并跨镜头轨迹，并把夹在连续轨迹中的短暂身份碎片归回原人物；
4. 输出 `visual_tracks.json`、每个人的原始帧证据和带原音频的标注视频；
5. 使用 LR-ASD TalkSet 权重融合嘴部运动和音频，为每张可见脸生成 25 FPS 的说话分数；
6. 默认按首版“通常不会同时说话”的约束，每 40 毫秒只保留得分最高的人脸；
7. 将 Sortformer 稳定后的 A/B/C/D 时间线与 Face-ID 做一对一重叠匹配，并保留全部候选证据。
8. 用已确认的可见说话人逐条检查 A/B/C/D 串号，保留原时间线并生成可审计的纠正时间线；
9. 将视觉身份投影到语音侧匿名参与者协议，保留已有个人信息并追加可追溯的多模态证据。
10. 在发言时间线上区分 `visible / offscreen / occluded / unknown`，并用切镜检测避免把换镜误判成遮挡。
11. 使用 PySceneDetect 提取镜头和关键帧，并从已有的人脸轨迹生成画面位置与二维空间关系；
12. 使用 Groq 上的 Qwen3.8 27B 结合关键帧、人物位置和对应转写，推断环境、物体与人物交互；
13. 将语义相近的相邻镜头合并成稳定场景，并投影为人物—场景—物体—交互图；
14. 将语音侧的话题、观点和意图按人物与话轮时间接入视觉场景图；
15. 在本地网页中交互查看场景时间轴、关系筛选、证据详情和标注关键帧。
16. 用人工复核的代表帧评测人物绑定、环境、物体、交互和二维空间关系。

人脸检测、特征与主动说话人模型在本地运行；场景语义分析通过 Groq 云 API 调用 Qwen3.8 27B。项目将本地模型 URL 固定到上游仓库的具体 Git revision，下载后校验文件大小和 SHA-256。YuNet 和 LR-ASD 为 MIT，SFace 为 Apache-2.0；详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

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

输入必须是声纹稳定后的 `A/B/C/D`，不接受原始 `speaker0…speaker3` 槽位。命令按时间重叠求全局一对一最优解，并输出 `confirmed`、`candidate`、`ambiguous` 或 `offscreen`；默认至少累计 3 秒一致的可见发言证据才会确认身份，较短证据仍作为候选保留。如果音频和视频片段起点不同，可用 `--timeline-offset-ms` 将语音时间整体平移后再匹配。

如果同一个语音标签落到多张可见人脸，先保留上述初始绑定，再用确认的人脸锚点纠正明显串号：

```bash
./lab reconcile-speakers \
  results/example/visual_tracks.json \
  /path/to/speech_spans.json
```

命令不会修改原始 `speech_spans.json`，而是生成 `reconciled_speech_spans.json`，记录每条修改前后的说话人、Face-ID、时间、覆盖率和主动说话片段证据。它会迭代更新 `visual_tracks.json` 中的绑定，直到纠正集合稳定；默认只接受置信度不低于 0.70 的已确认人脸锚点，并要求至少 1.2 秒重叠、60% 话段覆盖率和 60% 人脸领先幅度。短促插话或视觉证据不足的话段保持原标签。

如果产生了纠正，语音侧已有的 participant context、观点、意图和会话记忆仍包含旧说话人标签；后续集成应以 `reconciled_speech_spans.json` 重新生成这些语义结果，视觉模块不会静默改写原记忆。

用纠正时间线重新生成话轮、观点、意图、会话记忆和个人信息，并接回已确认的人脸：

```bash
./lab rebuild-voice-context \
  results/example/visual_tracks.json \
  results/example/reconciled_speech_spans.json \
  --output results/example/reconciled-voice \
  --groq-proxy http://127.0.0.1:7890
```

该命令把纠正后的 `speech_spans` 提取到新目录，通过冻结的 `HT-voice-lab` 命令行重新执行语义话轮、记忆压缩和个人信息提取，最后生成 `multimodal_participant_context.json`。它不会导入或修改语音项目代码，也不会覆盖原来的语音结果；`voice_context_build.json` 保存全部输入输出哈希。若不需要调用个人信息提取模型，可加 `--no-personal-info`。

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

从已有视觉轨迹生成场景上下文：

```bash
./lab analyze-scenes results/example/visual_tracks.json
```

默认生成 `scene_context.json` 和 `keyframes/`。每个镜头选择一张兼顾清晰度与可见人物的关键帧；连续移动但没有硬切镜的长镜头每 30 秒至少采样一次，可用 `--max-shot-seconds` 调整。结果记录 Face-ID、已确认的 A/B/C/D、归一化位置，以及 `left_of`、`above`、`co_visible_with`、`visually_close_to` 和画面重叠关系。关系只描述二维画面，不推断现实距离；首版有意不测量头部朝向。

检查场景协议：

```bash
./lab validate-scenes results/example/scene_context.json
```

让 Qwen 分析关键帧中的环境、物体和人物交互：

```bash
./lab analyze-scene-semantics \
  results/example/scene_context.json \
  --groq-proxy http://127.0.0.1:7890 \
  --visual-reuse-threshold 0.45

./lab validate-scene-semantics results/example/scene_semantics.json
```

命令优先读取 `GROQ_API_KEY`，其次读取本项目通过 `./lab login-groq` 保存的密钥，最后兼容已有的 `~/.config/ht-voice-lab/groq.key`。默认每次发送两张关键帧，以适配 Groq 当前的输入速率限制；批量响应缺少关键帧时会自动改为逐张重试。成功响应缓存在 `semantic_cache/`，网络中断后再次运行即可从未完成处继续。

重复机位较多的长视频可使用 `--visual-reuse-threshold 0.45`：系统裁掉字幕区后同时比较整帧和 3×3 分区的 HSV 颜色分布，只把每组代表帧发送给 Qwen；这可以区分颜色接近但画面布局不同的户外镜头。相似镜头复用代表帧的环境和物体结果，置信度按视觉相似度折减，人物发言关系则从该镜头自己的纠正转写重新建立。每条结果记录 `semantic_source`、`source_keyframe_id` 和 `visual_similarity`，不会把复用结果伪装成独立模型调用。短片或镜头差异较大的视频可以不启用。

`scene_semantics.json` 中的环境、物体和交互全部标为 `inferred`。系统只允许交互引用当前关键帧可见的 Face-ID，并过滤“对物体 speaking/listening”等不合理关系；它不会从外貌推断真实身份、性格、情绪、意图、视线或头部朝向。

构建稳定场景和多模态关系图：

```bash
./lab build-scene-graph \
  results/example/scene_semantics.json \
  --participant-context results/example/multimodal_participant_context.json

./lab validate-scene-graph results/example/multimodal_scene_graph.json
```

`--participant-context` 可选。提供后，图会用同一份 `visual_tracks.json` 的哈希校验两条处理链确实属于同一段视频，并把已有姓名和 participant ID 放入参与者节点。只有 `confirmed` 的 A/B/C/D—Face-ID 绑定会合并为同一个参与者；有争议或未绑定的 Face-ID 作为 `visual_identity` 证据保留，不计入参与者人数，并以 `identity_candidate` 边记录候选及置信度。场景合并完全在本地执行，默认综合环境描述、物体和人物的相似度，并用重复出现的代表机位吸收短暂近景切换，避免把同一访谈按摄像机角度拆成多个场景。

打开交互式场景图：

```bash
./lab visualize-scene-graph \
  results/example/multimodal_scene_graph.json \
  --host 127.0.0.1 \
  --port 8765
```

页面会自动打开浏览器。会话图默认只显示语音侧确认的参与者，未绑定的 Face-ID 可通过“视觉身份”开关查看；点击时间轴可聚焦场景，点击节点或关系可检查时间、置信度和来源证据，点击场景中的缩略图可查看带 Face-ID 标注的原始关键帧。网页只提供图协议中列出的关键帧，不开放结果目录中的其他文件。

将语音记忆接入场景图：

```bash
./lab build-conversation-graph \
  results/example/multimodal_scene_graph.json \
  /home/sunteng/Desktop/HighTorque_vision/HT-voice-lab/results/example/session_memory.json

./lab validate-conversation-graph \
  results/example/multimodal_conversation_graph.json

./lab visualize-conversation-graph \
  results/example/multimodal_conversation_graph.json \
  --host 127.0.0.1 \
  --port 8765
```

该步骤只读取版本化 JSON，不导入或调用 `HT-voice-lab` 的代码。系统通过 `multimodal_participant_context.json` 中记录的会话 ID 与 `session_memory.json` 哈希确认两条处理链属于同一场对话；人物 A/B/C/D 直接复用现有人物节点。每个观点按话轮与场景至少 120 毫秒的时间交集建立关系，跨场景的话轮会保留多条带覆盖比例的边，视觉范围以外的话轮仍进入会话图但不伪造场景关联。

## 视觉基线评测

仓库中有四个人工复核样例，覆盖四人户外访谈、中文低清舞台、暗光双人固定机位，以及在住宅和庭院间连续移动的跟拍视频。标注只保存视频哈希、关键帧 ID 和人工标签，不包含视频或图片。运行示例：

```bash
./lab evaluate-vision \
  benchmarks/english-4speakers-long-v1.json \
  results/regression-english-full-multimodal-v2/scene_context.json
```

默认从 `scene_context.json` 的来源字段读取 `visual_tracks.json`，并读取同目录的 `scene_semantics.json`；也可通过 `--visual-json` 和 `--semantics-json` 显式指定。输出 `vision_evaluation_report.json`，其中人物绑定和环境使用准确率，物体、交互和二维空间关系使用 Precision、Recall 与 F1，最终宏平均只计算实际有人工标签的维度。

标注文件中省略某个字段表示该维度不确定且不计分；显式写入空数组表示人工确认该帧没有对应对象。`object_vocabulary` 和 `interaction_predicates` 明确限定本次人工复核的范围，避免把同义家具、字幕水印或尚未标注的交互当成模型错误。物体数组应列出范围内全部显著物体，交互和空间关系数组应列出全部已确认关系。评测器会核对视频哈希、时长、关键帧时间、会话 ID 和上游文件哈希，避免把不同运行或不同视频的结果混在一起。

当前人工复核结果：

| 标注 | 复核帧 | 实际计分维度 | 宏平均 |
|---|---:|---|---:|
| `english-4speakers-long-v1.json` | 9 | 身份、环境、物体、发言、二维空间 | 0.967 |
| `cctv-stage-4speakers-v1.json` | 6 | 环境、物体 | 0.980 |
| `english-dark-studio-2speakers-v1.json` | 2 | 身份、环境、物体 | 1.000 |
| `vogue-moving-home-v1.json` | 10 | 身份、环境、物体 | 1.000 |

这些分数只代表 4 段视频中的 27 张人工复核帧。未在某个标注中列出的交互、空间关系或身份不会进入该样本的宏平均；结果用于防止基线回退，不能替代更大规模数据集。

## 输出

```text
results/example/
├── visual_tracks.json
├── reconciled_speech_spans.json
├── multimodal_participant_context.json
├── scene_context.json
├── scene_semantics.json
├── multimodal_scene_graph.json
├── multimodal_conversation_graph.json
├── vision_evaluation_report.json
├── annotated.mp4
├── active_speaker.mp4
├── semantic_cache/
├── keyframes/
│   ├── keyframe-00001.jpg
│   └── keyframe-00001-annotated.jpg
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

视觉辅助说话人纠正协议位于 [`schemas/reconciled_speaker_timeline.schema.json`](schemas/reconciled_speaker_timeline.schema.json)。它保留上游语音时间线的哈希、稳定视觉证据哈希和原始说话人，并只把满足阈值的视觉矛盾记录为独立 correction。

融合人物协议位于 [`schemas/multimodal_participant_context.schema.json`](schemas/multimodal_participant_context.schema.json)，它复用语音侧的参与者、个人信息观察和证据对象，并增加 `visual_entities` 与人物—视觉身份关联。

场景协议位于 [`schemas/scene_context.schema.json`](schemas/scene_context.schema.json)。它是从稳定视觉轨迹派生的独立文件，不会回写或改变 `visual_tracks.json`。

场景语义协议位于 [`schemas/scene_semantics.schema.json`](schemas/scene_semantics.schema.json)。它记录每次云端请求的模型、提示词版本、输入哈希、缓存命中和失败批次，并保留到原始场景、视觉轨迹及语音时间线的来源引用。

多模态场景图协议位于 [`schemas/multimodal_scene_graph.schema.json`](schemas/multimodal_scene_graph.schema.json)。人物和关键帧位置属于观察证据，环境、物体及交互属于推断证据；每个节点和边都保留时间范围、置信度与来源 ID。

多模态会话图协议位于 [`schemas/multimodal_conversation_graph.schema.json`](schemas/multimodal_conversation_graph.schema.json)。它完整保留场景图节点与边，再加入话题、观点、意图以及 `expresses / about / has_intent / occurred_in / discussed_in` 关系，并记录语音记忆、参与者上下文、场景图和场景上下文的来源哈希。

`faces` 是画面中的可见人物候选，不直接等于对话参与者。舞台远景、路人、插入镜头或静默听众可能使可见脸超过四张；使用 `active_speaker_segments` 可以筛出实际发言身份，而不删除原始视觉证据。

## 已验证样例

- 中文 480p、30 秒片段：镜头切换后保持 4 个可见身份，LR-ASD 检出的 3 位实际发言者与已知 0–8、9–21、22 秒之后的顺序一致；
- 英文 720p、60 秒片段：修复跨镜头和短暂身份碎片后得到 4 个可见身份，同一时刻的静默人脸弱阳性已被排除；
- 中文绑定得到 `A→Face-01`、`B→Face-02`、`C→Face-03`，三项均为 `confirmed`；英文 `A/B` 确认绑定，`C` 因同时积累到两个 Face-ID 的显著证据而标为 `ambiguous`；
- 英文四人上下文中的 A/B/C/D 始终计为 4 位参与者；未确认绑定的 Face-ID 作为视觉证据保留，不会再被重复计成人物；
- 英文 9 分 41 秒完整视频保持 4 个 Face-ID；视觉锚点纠正 4/77 条强证据发言后，得到 `A→Face-01`、`B→Face-04`、`C→Face-02`、`D→Face-03` 四项确认绑定；
- 中文三位发言者均为 `visible`；英文 A/B 为 `visible`，C 因身份冲突保持 `unknown`，系统没有把冲突错误改写为画外或遮挡；
- 中文 30 秒片段得到 4 个镜头，边界约为 8.72、21.04、28.32 秒；英文 60 秒片段得到 12 个镜头，其中纯片头画面正确记录为无人关键帧；
- Qwen 场景语义分析覆盖中文样例 4/4 张、英文样例 12/12 张关键帧；英文批量响应缺项时能够自动逐张重试并复用已成功缓存；
- 中文 4 个镜头稳定合并为 1 个演播室场景；英文 12 个镜头合并为室内、片头、户外、室内 4 个连续场景，未把片头或户外镜头错误并入访谈室内；
- 英文场景图网页已验证时间轴聚焦、节点与关系筛选、证据详情、关键帧缩略图和大图预览；
- 英文前 60 秒视觉图已与同场 250 秒语音记忆融合，超出视觉范围的话轮保持可检索且不生成虚假场景关系；
- 两条样例都通过结构引用、时间范围和单一说话者约束检查，主动说话视频保留原始音频。

## 后续顺序

1. 建立跨会话人物档案；
2. 接入摄像头实时处理和人工纠错。

语音基线固定为 `HT-voice-lab` 标签 `voice-baseline-2026-09-28`。

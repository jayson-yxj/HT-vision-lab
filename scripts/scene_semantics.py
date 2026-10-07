from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

from .groq_client import chat_completion
from .models import file_sha256
from .scene_context import validate_scene_context


PROMPT_VERSION = "scene-semantics-qwen3.8-v3"
ENVIRONMENT_CATEGORIES = (
    "indoor_meeting",
    "indoor_office",
    "indoor_studio",
    "indoor_home",
    "indoor_public",
    "outdoor_public",
    "outdoor_nature",
    "stage",
    "vehicle",
    "graphic_or_title",
    "other",
    "unknown",
)
INTERACTION_PREDICATES = (
    "speaking",
    "listening",
    "presenting",
    "showing",
    "holding",
    "using",
    "pointing_to",
    "sitting_with",
    "standing_with",
    "touching",
    "other",
)

SYSTEM_PROMPT = """你负责分析多人会话的关键帧。输入图片已经用 Face-ID 标出人物，文字元数据提供已确认的说话人标签、二维位置、关系和本镜头转写。
严格区分可观察事实和推断：不要根据外貌识别真实身份，不推断性格、情绪、意图、视线或头部朝向，不把同框自动解释为交流。环境和物体只描述图中可见内容；人物交互必须有直接画面证据，或由画面与转写共同支持，证据不足就省略。
环境分类以人物实际所在空间为准：有屋檐但向天空、庭院或室外建筑完全开放的露台仍属于 outdoor_public；只有人物处于四周封闭的办公空间才属于 indoor_office。黑色或纯色背景上的节目名称、字幕、台标或 Logo 属于 graphic_or_title。物体应覆盖显著家具、手持电子设备、车辆和背景建筑；图卡中可见的 Logo 也要记录，不能把相似场景中曾出现的物体补到当前画面。
每张输入关键帧必须返回且只返回一项分析。subject_ref 必须使用该帧可见的 Face-ID。object_ref 只能是可见 Face-ID、object:<物体英文短标签>、group、scene 或 null。仅凭同框不能把 speaking 或 listening 指向某个 Face-ID；没有明确称呼或可见交互对象时使用 group 或 null。环境类别必须使用给定枚举。描述保持简短。"""


def _response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["keyframes"],
        "properties": {
            "keyframes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["keyframe_id", "environment", "objects", "interactions"],
                    "properties": {
                        "keyframe_id": {"type": "string"},
                        "environment": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["category", "description", "confidence"],
                            "properties": {
                                "category": {"type": "string", "enum": list(ENVIRONMENT_CATEGORIES)},
                                "description": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["label", "count", "region", "confidence"],
                                "properties": {
                                    "label": {"type": "string"},
                                    "count": {"type": "integer", "minimum": 1},
                                    "region": {
                                        "type": "string",
                                        "enum": [
                                            "left",
                                            "center",
                                            "right",
                                            "foreground",
                                            "background",
                                            "whole_frame",
                                            "unknown",
                                        ],
                                    },
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                            },
                        },
                        "interactions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "subject_ref",
                                    "predicate",
                                    "object_ref",
                                    "description",
                                    "evidence_basis",
                                    "epistemic_status",
                                    "confidence",
                                ],
                                "properties": {
                                    "subject_ref": {"type": "string"},
                                    "predicate": {"type": "string", "enum": list(INTERACTION_PREDICATES)},
                                    "object_ref": {"type": ["string", "null"]},
                                    "description": {"type": "string"},
                                    "evidence_basis": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["image", "transcript", "geometry"]},
                                    },
                                    "epistemic_status": {"type": "string", "enum": ["observed", "inferred"]},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _timeline_items(payload) -> List[dict]:
    if isinstance(payload, dict):
        for key in ("speech_spans", "utterances", "segments"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    return payload if isinstance(payload, list) else []


def _speech_context(scene: dict, warnings: List[str]) -> Tuple[List[dict], dict]:
    visual_path = Path(scene["source"]["visual_tracks_path"])
    if not visual_path.is_file() or file_sha256(visual_path) != scene["source"]["visual_tracks_sha256"]:
        raise ValueError("scene context visual_tracks source is unavailable or has changed")
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    binding = visual.get("processing", {}).get("speaker_face_binding", {})
    raw_path = binding.get("speech_timeline_path")
    expected_hash = binding.get("speech_timeline_sha256")
    if not raw_path or not expected_hash:
        warnings.append("No bound speech timeline is available; semantic analysis uses visual evidence only.")
        return [], {"path": None, "sha256": None, "timeline_offset_ms": 0}
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != expected_hash:
        warnings.append("The bound speech timeline is unavailable or changed; semantic analysis uses visual evidence only.")
        return [], {"path": str(path), "sha256": expected_hash, "timeline_offset_ms": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    offset_ms = int(binding.get("timeline_offset_ms", 0))
    duration_ms = int(scene["source"]["duration_ms"])
    spans = []
    for index, item in enumerate(_timeline_items(payload), start=1):
        if not isinstance(item, dict) or item.get("status", "final") != "final":
            continue
        if item.get("attribution", "single_speaker") != "single_speaker":
            continue
        speaker = item.get("speaker")
        if speaker not in set("ABCD"):
            continue
        start = item.get("start_s", item.get("start"))
        end = item.get("end_s", item.get("end"))
        if start is None or end is None:
            continue
        start_ms = max(0, int(round(float(start) * 1000)) + offset_ms)
        end_ms = min(duration_ms, int(round(float(end) * 1000)) + offset_ms)
        if end_ms <= start_ms:
            continue
        spans.append(
            {
                "source_id": str(item.get("id") or f"speech-{index:06d}"),
                "speaker_label": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": str(item.get("text") or ""),
            }
        )
    return spans, {"path": str(path), "sha256": expected_hash, "timeline_offset_ms": offset_ms}


def _frame_metadata(scene: dict, keyframe: dict, speech: Sequence[dict]) -> dict:
    state_by_id = {item["person_state_id"]: item for item in scene["person_states"]}
    states = [state_by_id[item] for item in keyframe["person_state_ids"]]
    shot = next(item for item in scene["shots"] if item["shot_id"] == keyframe["shot_id"])
    relations = []
    for item in scene["spatial_relations"]:
        if item["keyframe_id"] != keyframe["keyframe_id"]:
            continue
        relations.append(
            {
                "subject_ref": state_by_id[item["subject_state_id"]]["face_id"],
                "predicate": item["predicate"],
                "object_ref": state_by_id[item["object_state_id"]]["face_id"],
                "confidence": item["confidence"],
            }
        )
    transcript = [
        item
        for item in speech
        if item["end_ms"] > shot["start_ms"] and item["start_ms"] < shot["end_ms"]
    ][:20]
    return {
        "keyframe_id": keyframe["keyframe_id"],
        "shot_id": keyframe["shot_id"],
        "timestamp_ms": keyframe["timestamp_ms"],
        "shot_start_ms": shot["start_ms"],
        "shot_end_ms": shot["end_ms"],
        "visible_people": [
            {
                "face_id": item["face_id"],
                "speaker_label": item["speaker_label"],
                "identity_status": item["identity_status"],
                "horizontal_region": item["horizontal_region"],
                "vertical_region": item["vertical_region"],
            }
            for item in states
        ],
        "spatial_relations": relations,
        "transcript": transcript,
    }


def _visual_reuse_plan(
    scene_path: Path, scene: dict, max_distance: Optional[float]
) -> Tuple[List[dict], dict]:
    keyframes = scene["keyframes"]
    if max_distance is None:
        return keyframes, {
            item["keyframe_id"]: {
                "source_keyframe_id": item["keyframe_id"],
                "visual_similarity": 1.0,
            }
            for item in keyframes
        }
    if not 0 <= max_distance <= 1:
        raise ValueError("visual reuse threshold must be between zero and one")
    clusters = []
    assignments = {}
    for keyframe in keyframes:
        image_path = (scene_path.parent / keyframe["image_path"]).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"keyframe is unavailable: {image_path}")
        image = image[: max(1, int(image.shape[0] * 0.82))]
        image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(histogram, histogram)
        spatial_histograms = []
        for top, bottom in ((0, 30), (30, 60), (60, 90)):
            for left, right in ((0, 53), (53, 106), (106, 160)):
                spatial = cv2.calcHist(
                    [hsv[top:bottom, left:right]],
                    [0, 1, 2],
                    None,
                    [12, 6, 6],
                    [0, 180, 0, 256, 0, 256],
                )
                spatial_histograms.append(cv2.normalize(spatial, spatial))
        nearest = min(
            (
                (
                    max(
                        cv2.compareHist(
                            histogram,
                            cluster["histogram"],
                            cv2.HISTCMP_BHATTACHARYYA,
                        ),
                        sum(
                            cv2.compareHist(
                                left, right, cv2.HISTCMP_BHATTACHARYYA
                            )
                            for left, right in zip(
                                spatial_histograms,
                                cluster["spatial_histograms"],
                            )
                        )
                        / len(spatial_histograms),
                    ),
                    cluster,
                )
                for cluster in clusters
            ),
            default=(float("inf"), None),
            key=lambda item: item[0],
        )
        if nearest[1] is None or nearest[0] > max_distance:
            cluster = {
                "histogram": histogram,
                "spatial_histograms": spatial_histograms,
                "representative": keyframe,
            }
            clusters.append(cluster)
            distance = 0.0
        else:
            distance, cluster = nearest
        assignments[keyframe["keyframe_id"]] = {
            "source_keyframe_id": cluster["representative"]["keyframe_id"],
            "visual_similarity": round(1.0 - float(distance), 6),
        }
    return [item["representative"] for item in clusters], assignments


def _transcript_interactions(metadata: dict) -> List[dict]:
    faces_by_speaker = {
        item["speaker_label"]: item["face_id"]
        for item in metadata["visible_people"]
        if item["speaker_label"]
    }
    interactions = []
    for speaker in dict.fromkeys(item["speaker_label"] for item in metadata["transcript"]):
        face_id = faces_by_speaker.get(speaker)
        if face_id is None:
            continue
        interactions.append(
            {
                "subject_ref": face_id,
                "predicate": "speaking",
                "object_ref": "group",
                "description": f"{face_id} is speaking during this shot.",
                "evidence_basis": ["transcript"],
                "epistemic_status": "inferred",
                "confidence": 0.9,
            }
        )
    return interactions


def _expand_visual_reuse(
    model_analyses: Sequence[dict], scene: dict, speech: Sequence[dict], reuse_plan: dict
) -> Tuple[List[dict], int, int]:
    by_keyframe = {item["keyframe_id"]: item for item in model_analyses}
    expanded = []
    object_count = interaction_count = 0
    for keyframe in scene["keyframes"]:
        reuse = reuse_plan[keyframe["keyframe_id"]]
        source = by_keyframe.get(reuse["source_keyframe_id"])
        if source is None:
            continue
        similarity = reuse["visual_similarity"]
        if source["keyframe_id"] == keyframe["keyframe_id"]:
            analysis = copy.deepcopy(source)
            semantic_source = "model"
        else:
            environment = copy.deepcopy(source["environment"])
            environment["confidence"] = round(environment["confidence"] * similarity, 6)
            objects = []
            for item in source["objects"]:
                copied = copy.deepcopy(item)
                copied["confidence"] = round(copied["confidence"] * similarity, 6)
                objects.append(copied)
            analysis = {
                "keyframe_id": keyframe["keyframe_id"],
                "shot_id": keyframe["shot_id"],
                "environment": environment,
                "objects": objects,
                "interactions": _transcript_interactions(
                    _frame_metadata(scene, keyframe, speech)
                ),
            }
            semantic_source = "visual_reuse"
        transcript_interactions = _transcript_interactions(
            _frame_metadata(scene, keyframe, speech)
        )
        existing = {
            (item["subject_ref"], item["predicate"], item.get("object_ref"))
            for item in analysis["interactions"]
        }
        analysis["interactions"].extend(
            item
            for item in transcript_interactions
            if (item["subject_ref"], item["predicate"], item.get("object_ref"))
            not in existing
        )
        analysis["analysis_id"] = f"scene-analysis-{len(expanded) + 1:06d}"
        analysis["semantic_source"] = semantic_source
        analysis["source_keyframe_id"] = source["keyframe_id"]
        analysis["visual_similarity"] = similarity
        for item in analysis["objects"]:
            object_count += 1
            item["semantic_object_id"] = f"semantic-object-{object_count:06d}"
        for item in analysis["interactions"]:
            interaction_count += 1
            item["interaction_id"] = f"interaction-{interaction_count:06d}"
            item["epistemic_status"] = "inferred"
        expanded.append(analysis)
    return expanded, object_count, interaction_count


def _image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _batch_payload(model: str, metadata: Sequence[dict], image_paths: Sequence[Path]) -> dict:
    content = [
        {
            "type": "text",
            "text": "请按输入顺序分析以下关键帧，严格返回 JSON Schema。元数据：\n"
            + json.dumps(list(metadata), ensure_ascii=False),
        }
    ]
    for item, path in zip(metadata, image_paths):
        content.extend(
            [
                {"type": "text", "text": "以下图片对应 " + item["keyframe_id"]},
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
            ]
        )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 3000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "visual_scene_semantics",
                "strict": True,
                "schema": _response_schema(),
            },
        },
    }


def _request_hash(model: str, metadata: Sequence[dict], image_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode())
    digest.update(model.encode())
    digest.update(json.dumps(list(metadata), sort_keys=True, ensure_ascii=False).encode())
    for path in image_paths:
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def _parse_response(response: dict, expected_ids: Sequence[str]) -> List[dict]:
    try:
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        analyses = parsed["keyframes"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Groq returned an invalid scene-semantics response") from error
    if not isinstance(analyses, list):
        raise ValueError("Groq scene-semantics response has no keyframes array")
    returned_ids = [item.get("keyframe_id") for item in analyses if isinstance(item, dict)]
    if returned_ids != list(expected_ids):
        raise ValueError(
            "Groq scene-semantics keyframes do not match the request: "
            + repr(returned_ids)
        )
    return analyses


class GroqSceneAnalyzer:
    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",
        timeout: float = 60.0,
        proxy: Optional[str] = None,
        minimum_interval: float = 0.2,
        transport: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        if timeout <= 0 or minimum_interval < 0:
            raise ValueError("Groq timeout must be positive and minimum interval cannot be negative")
        self.model = model
        self.timeout = timeout
        self.proxy = proxy
        self.minimum_interval = minimum_interval
        self.transport = transport or self._post
        self.last_request_started: Optional[float] = None

    def _post(self, payload: dict) -> dict:
        return chat_completion(payload, timeout=self.timeout, proxy=self.proxy)

    def analyze(self, metadata: Sequence[dict], image_paths: Sequence[Path]) -> List[dict]:
        if self.last_request_started is not None:
            wait = max(0.0, self.minimum_interval - (time.monotonic() - self.last_request_started))
            if wait:
                time.sleep(wait)
        self.last_request_started = time.monotonic()
        response = self.transport(_batch_payload(self.model, metadata, image_paths))
        return _parse_response(response, [item["keyframe_id"] for item in metadata])


def _cached_request(
    analyzer: GroqSceneAnalyzer,
    model: str,
    metadata: Sequence[dict],
    image_paths: Sequence[Path],
    cache_dir: Path,
) -> Tuple[List[dict], str, bool]:
    request_hash = _request_hash(model, metadata, image_paths)
    cache_path = cache_dir / f"{request_hash}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fallback"):
            raise ValueError(str(cached.get("error") or "cached batch requires single-image fallback"))
        raw = cached["analyses"]
        expected = [item["keyframe_id"] for item in metadata]
        if [item.get("keyframe_id") for item in raw] != expected:
            raise ValueError("semantic cache keyframes do not match the request")
        return raw, request_hash, True
    raw = analyzer.analyze(metadata, image_paths)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"request_sha256": request_hash, "analyses": raw}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return raw, request_hash, False


def _seed_cached_analyses(
    cache_dir: Path,
    model: str,
    scene_path: Path,
    scene: dict,
    speech: Sequence[dict],
    representative_ids: set,
) -> dict:
    """Recover individual representatives from valid older batch cache entries."""
    keyframes = {item["keyframe_id"]: item for item in scene["keyframes"]}
    recovered = {}
    for cache_path in sorted(cache_dir.glob("*.json"), key=lambda path: path.stat().st_mtime):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        raw = cached.get("analyses")
        if not isinstance(raw, list) or not raw:
            continue
        identifiers = [item.get("keyframe_id") for item in raw if isinstance(item, dict)]
        if len(identifiers) != len(raw) or any(identifier not in keyframes for identifier in identifiers):
            continue
        selected = [keyframes[identifier] for identifier in identifiers]
        metadata = [_frame_metadata(scene, item, speech) for item in selected]
        image_paths = [
            (scene_path.parent / item["annotated_image_path"]).resolve()
            for item in selected
        ]
        expected = _request_hash(model, metadata, image_paths)
        if cache_path.stem != expected or cached.get("request_sha256") != expected:
            continue
        for item in raw:
            if item["keyframe_id"] in representative_ids:
                recovered[item["keyframe_id"]] = {
                    "analysis": item,
                    "request_sha256": expected,
                }
    return recovered


def _mark_batch_for_fallback(
    model: str,
    metadata: Sequence[dict],
    image_paths: Sequence[Path],
    cache_dir: Path,
    error: Exception,
) -> None:
    request_hash = _request_hash(model, metadata, image_paths)
    path = cache_dir / f"{request_hash}.json"
    path.write_text(
        json.dumps(
            {"request_sha256": request_hash, "fallback": True, "error": str(error)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _repair_graphic_environment(environment: dict, objects: Sequence[dict]) -> dict:
    repaired = copy.deepcopy(environment)
    labels = {str(item.get("label", "")).strip().lower() for item in objects}
    if repaired.get("confidence", 0) <= 0.2 and labels & {
        "logo",
        "title",
        "title card",
        "graphic",
    }:
        repaired.update(
            category="graphic_or_title",
            description="Graphic or title card with a visible logo or title.",
            confidence=0.6,
        )
    return repaired


def _is_scene_object(value: dict, environment: dict) -> bool:
    label = str(value.get("label", "")).strip().lower()
    if label in {"subtitle", "text"}:
        return False
    return label != "logo" or environment.get("category") == "graphic_or_title"


def _normalize_analyses(
    raw: Sequence[dict],
    scene: dict,
    analysis_offset: int,
    object_offset: int,
    interaction_offset: int,
) -> Tuple[List[dict], int, int]:
    keyframes = {item["keyframe_id"]: item for item in scene["keyframes"]}
    normalized = []
    object_count, interaction_count = object_offset, interaction_offset
    for index, item in enumerate(raw, start=1):
        keyframe_id = item["keyframe_id"]
        environment = _repair_graphic_environment(item["environment"], item["objects"])
        objects = []
        for value in item["objects"]:
            if not _is_scene_object(value, environment):
                continue
            object_count += 1
            objects.append(
                {
                    "semantic_object_id": f"semantic-object-{object_count:06d}",
                    **value,
                    "epistemic_status": "inferred",
                }
            )
        interactions = []
        for value in item["interactions"]:
            if value["predicate"] in {"speaking", "listening"} and value.get(
                "object_ref"
            ) not in {None, "group"}:
                continue
            interaction_count += 1
            interactions.append(
                {
                    "interaction_id": f"interaction-{interaction_count:06d}",
                    **value,
                    "epistemic_status": "inferred",
                }
            )
        normalized.append(
            {
                "analysis_id": f"scene-analysis-{analysis_offset + index:06d}",
                "keyframe_id": keyframe_id,
                "shot_id": keyframes[keyframe_id]["shot_id"],
                "environment": {**environment, "epistemic_status": "inferred"},
                "objects": objects,
                "interactions": interactions,
            }
        )
    return normalized, object_count, interaction_count


def analyze_scene_semantics(
    scene_path: Path,
    output_path: Optional[Path] = None,
    model: str = "qwen/qwen3.8-27b",
    timeout: float = 60.0,
    proxy: Optional[str] = None,
    minimum_interval: float = 0.2,
    batch_size: int = 2,
    visual_reuse_threshold: Optional[float] = None,
    transport: Optional[Callable[[dict], dict]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    if not 1 <= batch_size <= 3:
        raise ValueError("Qwen image batch size must be between one and three")
    scene_path = scene_path.expanduser().resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    errors = validate_scene_context(scene)
    if errors:
        raise ValueError("invalid scene context: " + "; ".join(errors))
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else scene_path.parent / "scene_semantics.json"
    )
    protected_paths = {
        scene_path,
        Path(scene["source"]["visual_tracks_path"]).expanduser().resolve(),
        Path(scene["source"]["video_path"]).expanduser().resolve(),
    }
    if output_path in protected_paths:
        raise ValueError("scene semantics output must not overwrite a source file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / "semantic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    speech, speech_source = _speech_context(scene, warnings)
    analyzer = GroqSceneAnalyzer(model, timeout, proxy, minimum_interval, transport)
    analyses, requests, failed_batches = [], [], []
    object_count = interaction_count = 0
    all_keyframes = scene["keyframes"]
    keyframes, reuse_plan = _visual_reuse_plan(
        scene_path, scene, visual_reuse_threshold
    )
    if visual_reuse_threshold is not None:
        warnings.append(
            f"Visual reuse selected {len(keyframes)} representative keyframes for "
            f"{len(all_keyframes)} total keyframes."
        )
        if progress:
            progress(
                f"selected {len(keyframes)} visual representatives for "
                f"{len(all_keyframes)} keyframes"
            )
    seeded = _seed_cached_analyses(
        cache_dir,
        model,
        scene_path,
        scene,
        speech,
        {item["keyframe_id"] for item in keyframes},
    )
    for keyframe in keyframes:
        cached = seeded.get(keyframe["keyframe_id"])
        if cached is None:
            continue
        normalized, next_object_count, next_interaction_count = _normalize_analyses(
            [cached["analysis"]],
            scene,
            len(analyses),
            object_count,
            interaction_count,
        )
        if _validate_analysis_refs(normalized, scene):
            seeded.pop(keyframe["keyframe_id"], None)
            continue
        analyses.extend(normalized)
        object_count, interaction_count = next_object_count, next_interaction_count
        requests.append(
            {
                "batch_id": f"semantic-seed-{keyframe['keyframe_id']}",
                "keyframe_ids": [keyframe["keyframe_id"]],
                "request_sha256": cached["request_sha256"],
                "cache_hit": True,
            }
        )
    if seeded and progress:
        progress(f"recovered {len(seeded)} representatives from compatible batch cache")
    keyframes = [item for item in keyframes if item["keyframe_id"] not in seeded]
    stop = False
    for batch_number, start in enumerate(range(0, len(keyframes), batch_size), start=1):
        batch = keyframes[start : start + batch_size]
        metadata = [_frame_metadata(scene, item, speech) for item in batch]
        image_paths = [
            (scene_path.parent / item["annotated_image_path"]).resolve() for item in batch
        ]
        if any(not path.is_file() for path in image_paths):
            missing = [str(path) for path in image_paths if not path.is_file()]
            raise FileNotFoundError("annotated keyframes are unavailable: " + ", ".join(missing))
        batch_id = f"semantic-batch-{batch_number:05d}"

        def process_unit(unit_metadata, unit_paths, unit_id):
            nonlocal object_count, interaction_count
            raw, request_hash, cache_hit = _cached_request(
                analyzer, model, unit_metadata, unit_paths, cache_dir
            )
            normalized, next_object_count, next_interaction_count = _normalize_analyses(
                raw, scene, len(analyses), object_count, interaction_count
            )
            batch_errors = _validate_analysis_refs(normalized, scene)
            if batch_errors:
                raise ValueError("; ".join(batch_errors))
            analyses.extend(normalized)
            object_count, interaction_count = next_object_count, next_interaction_count
            requests.append(
                {
                    "batch_id": unit_id,
                    "keyframe_ids": [item["keyframe_id"] for item in unit_metadata],
                    "request_sha256": request_hash,
                    "cache_hit": cache_hit,
                }
            )

        try:
            if progress:
                progress(
                    f"analyzing {batch_id}: "
                    + ", ".join(item["keyframe_id"] for item in batch)
                )
            process_unit(metadata, image_paths, batch_id)
            if progress:
                progress(f"completed {batch_id}")
        except ValueError as error:
            if len(batch) > 1:
                _mark_batch_for_fallback(model, metadata, image_paths, cache_dir, error)
                warnings.append(f"{batch_id} returned incomplete references and was retried one image at a time.")
                if progress:
                    progress(f"{batch_id} incomplete; retrying one image at a time")
                for part, (item_metadata, image_path) in enumerate(
                    zip(metadata, image_paths), start=1
                ):
                    part_id = f"{batch_id}-part-{part:02d}"
                    try:
                        process_unit([item_metadata], [image_path], part_id)
                        if progress:
                            progress(f"completed {part_id}")
                    except (OSError, ValueError, KeyError, TypeError) as part_error:
                        failed_batches.append(
                            {
                                "batch_id": part_id,
                                "keyframe_ids": [item_metadata["keyframe_id"]],
                                "error": str(part_error),
                            }
                        )
                        warnings.append(f"{part_id} failed; rerun to resume from cached successful batches.")
                        stop = True
                        break
                if not stop:
                    continue
                break
            failed_batches.append(
                {
                    "batch_id": batch_id,
                    "keyframe_ids": [item["keyframe_id"] for item in batch],
                    "error": str(error),
                }
            )
            warnings.append(f"{batch_id} failed; rerun to resume from cached successful batches.")
            break
        except (OSError, KeyError, TypeError) as error:
            failed_batches.append(
                {
                    "batch_id": batch_id,
                    "keyframe_ids": [item["keyframe_id"] for item in batch],
                    "error": str(error),
                }
            )
            warnings.append(f"{batch_id} failed; rerun to resume from cached successful batches.")
            break

    model_analysis_count = len(analyses)
    analyses, object_count, interaction_count = _expand_visual_reuse(
        analyses, scene, speech, reuse_plan
    )
    status = (
        "complete"
        if not failed_batches and len(analyses) == len(all_keyframes)
        else "partial"
    )
    data = {
        "schema_version": 1,
        "context_type": "visual_scene_semantics",
        "session_id": scene["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": {
            "scene_context_path": str(scene_path),
            "scene_context_sha256": file_sha256(scene_path),
            "visual_tracks_path": scene["source"]["visual_tracks_path"],
            "visual_tracks_sha256": scene["source"]["visual_tracks_sha256"],
            "speech_timeline": speech_source,
        },
        "provider": {
            "name": "groq",
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "batch_size": batch_size,
            "visual_reuse_threshold": visual_reuse_threshold,
            "reuse_algorithm": (
                "cropped_spatial_hsv_histogram_bhattacharyya_v2"
                if visual_reuse_threshold is not None
                else None
            ),
        },
        "analyses": analyses,
        "requests": requests,
        "failed_batches": failed_batches,
        "statistics": {
            "keyframes": len(all_keyframes),
            "analyzed_keyframes": len(analyses),
            "model_analyzed_keyframes": model_analysis_count,
            "reused_keyframes": len(analyses) - model_analysis_count,
            "semantic_objects": object_count,
            "interactions": interaction_count,
            "requests": len(requests),
            "failed_batches": len(failed_batches),
        },
        "warnings": warnings,
    }
    validation_errors = validate_scene_semantics(data)
    if validation_errors:
        raise RuntimeError("scene semantics validation failed: " + "; ".join(validation_errors))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return data


def _validate_analysis_refs(analyses: Sequence[dict], scene: dict) -> List[str]:
    states = {item["person_state_id"]: item for item in scene["person_states"]}
    visible_by_keyframe = {
        item["keyframe_id"]: {states[state_id]["face_id"] for state_id in item["person_state_ids"]}
        for item in scene["keyframes"]
    }
    errors = []
    for analysis in analyses:
        visible = visible_by_keyframe.get(analysis["keyframe_id"], set())
        for interaction in analysis["interactions"]:
            subject = interaction["subject_ref"]
            object_ = interaction["object_ref"]
            if subject not in visible:
                errors.append(f"{interaction['interaction_id']} subject is not visible in its keyframe")
            if object_ and object_.startswith("Face-") and object_ not in visible:
                errors.append(f"{interaction['interaction_id']} object is not visible in its keyframe")
            if object_ and not (
                object_.startswith("Face-")
                or object_.startswith("object:")
                or object_ in {"group", "scene"}
            ):
                errors.append(f"{interaction['interaction_id']} has an invalid object reference")
    return errors


def validate_scene_semantics(data: dict) -> List[str]:
    required = {
        "schema_version",
        "context_type",
        "session_id",
        "created_at",
        "status",
        "source",
        "provider",
        "analyses",
        "requests",
        "failed_batches",
        "statistics",
        "warnings",
    }
    missing = required - set(data)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    errors = []
    analysis_ids = [item["analysis_id"] for item in data["analyses"]]
    keyframe_ids = [item["keyframe_id"] for item in data["analyses"]]
    object_ids = [
        item["semantic_object_id"] for analysis in data["analyses"] for item in analysis["objects"]
    ]
    interaction_ids = [
        item["interaction_id"]
        for analysis in data["analyses"]
        for item in analysis["interactions"]
    ]
    for label, values in (
        ("analysis", analysis_ids),
        ("analyzed keyframe", keyframe_ids),
        ("semantic object", object_ids),
        ("interaction", interaction_ids),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label} IDs")
    if data["status"] == "complete" and data["failed_batches"]:
        errors.append("complete semantics contain failed batches")
    if data["status"] == "complete" and len(data["analyses"]) != data["statistics"]["keyframes"]:
        errors.append("complete semantics do not cover every keyframe")
    for analysis in data["analyses"]:
        semantic_source = analysis.get("semantic_source")
        if semantic_source not in {None, "model", "visual_reuse"}:
            errors.append(f"{analysis['analysis_id']} has an invalid semantic source")
        if semantic_source == "visual_reuse" and not analysis.get("source_keyframe_id"):
            errors.append(f"{analysis['analysis_id']} has no reuse source keyframe")
        similarity = analysis.get("visual_similarity")
        if similarity is not None and not 0 <= similarity <= 1:
            errors.append(f"{analysis['analysis_id']} has invalid visual similarity")
        environment = analysis["environment"]
        if environment.get("category") not in ENVIRONMENT_CATEGORIES:
            errors.append(f"{analysis['analysis_id']} has an invalid environment category")
        if environment.get("epistemic_status") != "inferred":
            errors.append(f"{analysis['analysis_id']} environment is not marked inferred")
        if not 0 <= environment.get("confidence", -1) <= 1:
            errors.append(f"{analysis['analysis_id']} environment has invalid confidence")
        for item in analysis["objects"]:
            if item.get("epistemic_status") != "inferred":
                errors.append(f"{item['semantic_object_id']} is not marked inferred")
            if not 0 <= item.get("confidence", -1) <= 1:
                errors.append(f"{item['semantic_object_id']} has invalid confidence")
        for item in analysis["interactions"]:
            if item.get("predicate") not in INTERACTION_PREDICATES:
                errors.append(f"{item['interaction_id']} has an invalid predicate")
            if item.get("epistemic_status") != "inferred":
                errors.append(f"{item['interaction_id']} is not marked inferred")
            if not 0 <= item.get("confidence", -1) <= 1:
                errors.append(f"{item['interaction_id']} has invalid confidence")
    expected = {
        "keyframes": data["statistics"]["keyframes"],
        "analyzed_keyframes": len(data["analyses"]),
        "semantic_objects": len(object_ids),
        "interactions": len(interaction_ids),
        "requests": len(data["requests"]),
        "failed_batches": len(data["failed_batches"]),
    }
    if "model_analyzed_keyframes" in data["statistics"]:
        expected["model_analyzed_keyframes"] = sum(
            item.get("semantic_source") in {None, "model"} for item in data["analyses"]
        )
    if "reused_keyframes" in data["statistics"]:
        expected["reused_keyframes"] = sum(
            item.get("semantic_source") == "visual_reuse" for item in data["analyses"]
        )
    if data["statistics"] != expected:
        errors.append("statistics do not match scene semantics contents")
    return errors

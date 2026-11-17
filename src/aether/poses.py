"""ท่ามือที่ผู้ใช้สอนเอง (custom poses) — จำจากตำแหน่ง landmark 21 จุด ไม่ใช้รูปภาพ ไม่ต้องเทรนโมเดล

ใช้แทน MediaPipe Model Maker ซึ่งติดตั้งบน Windows ไม่ได้และต้องเก็บรูปจากกล้องไว้เทรน
(ดู ``docs/DECISIONS.md`` D40) วิธีที่ใช้คือ nearest neighbour บน "รูปทรงมือ":

1. :func:`pose_features` แปลงมือเป็นเวกเตอร์ 60 ค่า (20 จุด × x, y, z) ที่ไม่ขึ้นกับตำแหน่ง
   ขนาด และการหมุนในระนาบภาพ — ย้ายข้อมือไปที่จุดกำเนิด หารด้วยขนาดฝ่ามือ หมุนให้ข้อมือ→โคนนิ้วกลาง
   ชี้ขึ้น และกลับซ้าย/ขวาของมือซ้ายให้เหมือนมือขวา ทิศของมือ (องศา) แยกเก็บไว้ต่างหาก
2. แต่ละท่าเก็บตัวอย่างที่อัดตอนสอน (:class:`PoseTemplate`) พร้อม "มือปกติ" ของผู้ใช้
   (``negatives`` — ตอนชี้/จีบตามปกติ) ไว้เป็นตัวเปรียบเทียบ
3. :class:`PoseClassifier` วัดระยะ (RMS ต่อจุด หน่วยเป็นเท่าของขนาดฝ่ามือ) จากมือในเฟรมไปยัง
   ตัวอย่างที่ใกล้ที่สุด ``K_NEAREST`` ตัวของแต่ละท่า ท่าที่ใกล้สุดจะ "ผ่าน" เมื่อ
   ใกล้กว่าเกณฑ์ของท่านั้น ใกล้กว่ามือปกติ และไม่ก้ำกึ่งกับท่าอื่น

ไฟล์ ``poses.json`` (ข้างตัวแอป เหมือน ``calibration.json``) เก็บแค่ตัวเลขเหล่านี้ ไม่มีภาพ
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .hotkeys import parse_action
from .tracker import LANDMARK_COUNT, LM, Hand

__all__ = [
    "FEATURE_SIZE",
    "K_NEAREST",
    "MAX_NEGATIVES",
    "MAX_POSES",
    "MAX_SAMPLES",
    "POSES_VERSION",
    "POSE_PREFIX",
    "PoseClassifier",
    "PoseFileError",
    "PoseLibrary",
    "PoseMatch",
    "PoseTemplate",
    "angle_difference",
    "feature_distance",
    "knn_distance",
    "load_poses",
    "make_template",
    "pose_features",
    "save_poses",
    "spread",
    "validate_pose_name",
]

POSE_PREFIX = "pose:"  # ชื่อท่าใน engine/event = "pose:<ชื่อ>" (ไม่ชนกับท่าสำเร็จรูปของโมเดล)
POSES_VERSION = 1

FEATURE_POINTS = tuple(range(1, LANDMARK_COUNT))  # ข้อมือเป็นจุดกำเนิด จึงไม่นับ
FEATURE_SIZE = len(FEATURE_POINTS) * 3
Z_WEIGHT = 0.5  # z จากกล้องตัวเดียวแกว่งกว่า x/y จึงให้น้ำหนักครึ่งเดียว

K_NEAREST = 3
MAX_SAMPLES = 40  # ตัวอย่างต่อท่า (เลือกให้กระจายตลอดช่วงที่อัด)
MAX_NEGATIVES = 120
MAX_POSES = 12
THRESHOLD_FACTOR = 1.2  # เกณฑ์ของท่า = ความแกว่งของตัวอย่างตอนสอน × ค่านี้
THRESHOLD_RANGE = (0.10, 0.25)
ORIENTATION_TOLERANCE = 60.0  # มือต้องหันทิศเดียวกับตอนสอน ±องศานี้ (👍 กับ 👎 จึงเป็นคนละท่า)
AMBIGUITY_RATIO = 0.8  # ท่าที่ใกล้สุดต้องใกล้กว่าท่าอันดับสองอย่างน้อยเท่านี้ ไม่งั้นถือว่าก้ำกึ่ง

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,23}")

Vector = tuple[float, ...]


class PoseFileError(ValueError):
    """``poses.json`` อ่านไม่ได้หรือรูปแบบผิด (ข้อความเป็นคำแนะนำสำหรับผู้ใช้)"""


def validate_pose_name(name: str) -> str:
    """ชื่อท่า: a–z, 0–9, ``_``, ``-`` ยาวไม่เกิน 24 ตัว (คืนชื่อที่ตัดช่องว่างและเป็นตัวเล็กแล้ว)"""
    cleaned = name.strip().lower()
    if not _NAME.fullmatch(cleaned):
        raise ValueError(
            f"ชื่อท่า {name!r} ใช้ไม่ได้ — ใช้ a-z, 0-9, _ หรือ - ยาวไม่เกิน 24 ตัว (ขึ้นต้นด้วยตัวอักษร/ตัวเลข)"
        )
    return cleaned


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def pose_features(hand: Hand, width: int, height: int) -> tuple[Vector, float] | None:
    """รูปทรงของมือ → ``(เวกเตอร์ FEATURE_SIZE ค่า, ทิศของมือเป็นองศา)``

    ทิศ 0° = ข้อมือ→โคนนิ้วกลางชี้ขึ้น ค่าบวก = เอียงไปทางนิ้วโป้งของมือขวา (ตามเข็มนาฬิกาบนจอ)
    คืน ``None`` ถ้ามือเล็กจนวัดไม่ได้หรือ landmark ไม่ครบ
    """
    points = hand.landmarks
    if len(points) != LANDMARK_COUNT or width <= 0 or height <= 0:
        return None
    flip = -1.0 if hand.handedness == "left" else 1.0
    wrist = points[LM.WRIST]
    local = [
        (
            (p.x - wrist.x) * width * flip,
            (p.y - wrist.y) * height,
            (p.z - wrist.z) * width,  # MediaPipe: z ใช้สเกลใกล้เคียงกับ x
        )
        for p in points
    ]
    palm = (
        math.dist(local[LM.WRIST], local[LM.INDEX_MCP])
        + math.dist(local[LM.WRIST], local[LM.PINKY_MCP])
        + math.dist(local[LM.INDEX_MCP], local[LM.PINKY_MCP])
        + math.dist(local[LM.WRIST], local[LM.MIDDLE_MCP])
    ) / 4.0
    if palm < 1e-6:
        return None
    mx, my, _ = local[LM.MIDDLE_MCP]
    angle = math.atan2(mx, -my)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    features: list[float] = []
    for index in FEATURE_POINTS:
        x, y, z = local[index]
        # หมุนกลับ (ทวนเข็มบนจอ) ให้ข้อมือ→โคนนิ้วกลางชี้ขึ้น
        features.append((x * cos_a + y * sin_a) / palm)
        features.append((-x * sin_a + y * cos_a) / palm)
        features.append(z * Z_WEIGHT / palm)
    return tuple(features), math.degrees(angle)


def feature_distance(a: Vector, b: Vector) -> float:
    """ระยะ RMS ต่อจุด (หน่วย = ขนาดฝ่ามือ): 0.1 ≈ แต่ละจุดคลาดกันราว 10% ของฝ่ามือ"""
    return math.dist(a, b) / math.sqrt(len(FEATURE_POINTS))


def knn_distance(samples: Sequence[Vector], features: Vector, k: int = K_NEAREST) -> float:
    """ค่าเฉลี่ยระยะไปยังตัวอย่างที่ใกล้ที่สุด ``k`` ตัว (``inf`` ถ้าไม่มีตัวอย่าง)"""
    if not samples:
        return math.inf
    nearest = heapq.nsmallest(k, (math.dist(s, features) for s in samples))
    return sum(nearest) / len(nearest) / math.sqrt(len(FEATURE_POINTS))


def angle_difference(a: float, b: float) -> float:
    """ผลต่างของมุมสองมุม (องศา, 0–180)"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _circular_mean(angles: Iterable[float]) -> float:
    angles = list(angles)
    sx = sum(math.sin(math.radians(a)) for a in angles)
    cx = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sx, cx)) if angles else 0.0


def spread[T](items: Sequence[T], count: int) -> list[T]:
    """เลือก ``count`` ตัวที่กระจายเท่าๆ กันตลอดลำดับ (ไม่เกินจำนวนที่มี)"""
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]] if count == 1 else []
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


# --------------------------------------------------------------------------- #
# Templates / library
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PoseTemplate:
    """ท่าหนึ่งท่า: ตัวอย่างรูปทรงมือ + ทิศเฉลี่ย + เกณฑ์ระยะที่เรียนจากตัวอย่าง"""

    name: str
    action: str  # ค่าแบบเดียวกับ [actions] ใน config: ปุ่มลัด | "minimize_foreground" | "none"
    samples: tuple[Vector, ...]
    orientation: float
    threshold: float

    @property
    def label(self) -> str:
        """ชื่อใน engine/event: ``pose:<ชื่อ>``"""
        return POSE_PREFIX + self.name


def make_template(name: str, action: str, samples: Sequence[tuple[Vector, float]]) -> PoseTemplate:
    """สร้างท่าจากตัวอย่าง ``(features, orientation)`` ที่อัดไว้ (เลือกให้เหลือไม่เกิน MAX_SAMPLES)"""
    name = validate_pose_name(name)
    parse_action(action)  # ValueError ถ้าปุ่มลัดผิด
    if not samples:
        raise ValueError("ไม่มีตัวอย่างของท่านี้")
    chosen = spread(samples, MAX_SAMPLES)
    vectors = tuple(tuple(round(v, 4) for v in f) for f, _ in chosen)
    orientation = _circular_mean(angle for _, angle in chosen)
    # ความแกว่งของท่าตอนสอน: ระยะจากตัวอย่างแต่ละตัวถึงค่าเฉลี่ย (เปอร์เซ็นไทล์ที่ 90)
    mean = tuple(sum(column) / len(vectors) for column in zip(*vectors))
    distances = sorted(feature_distance(v, mean) for v in vectors)
    p90 = distances[min(len(distances) - 1, math.ceil(0.9 * len(distances)) - 1)]
    low, high = THRESHOLD_RANGE
    threshold = min(max(p90 * THRESHOLD_FACTOR, low), high)
    return PoseTemplate(name, action.strip(), vectors, round(orientation, 2), round(threshold, 4))


@dataclass(frozen=True, slots=True)
class PoseLibrary:
    poses: tuple[PoseTemplate, ...] = ()
    negatives: tuple[Vector, ...] = ()  # มือปกติของผู้ใช้ (ตอนชี้/จีบ) — ไม่ใช่ท่าใดๆ

    def __len__(self) -> int:
        return len(self.poses)

    def get(self, name: str) -> PoseTemplate | None:
        return next((p for p in self.poses if p.name == name), None)

    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.poses)

    def with_pose(self, template: PoseTemplate) -> PoseLibrary:
        """เพิ่มท่า (ชื่อซ้ำ = แทนที่ท่าเดิม ตำแหน่งเดิม)"""
        if self.get(template.name) is not None:
            poses = tuple(template if p.name == template.name else p for p in self.poses)
        else:
            if len(self.poses) >= MAX_POSES:
                raise ValueError(f"สอนได้ไม่เกิน {MAX_POSES} ท่า — ลบท่าเก่าก่อน")
            poses = (*self.poses, template)
        return replace(self, poses=poses)

    def without(self, name: str) -> PoseLibrary:
        return replace(self, poses=tuple(p for p in self.poses if p.name != name))

    def with_action(self, name: str, action: str) -> PoseLibrary:
        parse_action(action)
        return replace(
            self,
            poses=tuple(
                replace(p, action=action.strip()) if p.name == name else p for p in self.poses
            ),
        )

    def with_negatives(self, negatives: Iterable[Vector]) -> PoseLibrary:
        """เพิ่มตัวอย่างมือปกติ (เก็บล่าสุดไม่เกิน MAX_NEGATIVES ตัว)"""
        merged = (*self.negatives, *(tuple(round(v, 4) for v in n) for n in negatives))
        return replace(self, negatives=merged[-MAX_NEGATIVES:])

    def actions(self) -> dict[str, str]:
        """``{"pose:<ชื่อ>": action}`` สำหรับ :class:`aether.actions.ActionExecutor`"""
        return {p.label: p.action for p in self.poses}

    def to_dict(self) -> dict:
        return {
            "version": POSES_VERSION,
            "poses": [
                {
                    "name": p.name,
                    "action": p.action,
                    "orientation": p.orientation,
                    "threshold": p.threshold,
                    "samples": [list(s) for s in p.samples],
                }
                for p in self.poses
            ],
            "negatives": [list(n) for n in self.negatives],
        }

    @classmethod
    def from_dict(cls, data: object) -> PoseLibrary:
        if not isinstance(data, dict) or data.get("version") != POSES_VERSION:
            raise PoseFileError(f"ไม่ใช่ไฟล์ท่ามือเวอร์ชัน {POSES_VERSION}")
        try:
            poses: list[PoseTemplate] = []
            for entry in data.get("poses", []):
                name = validate_pose_name(entry["name"])
                action = str(entry["action"])
                parse_action(action)
                samples = tuple(_vector(s) for s in entry["samples"])
                if not samples:
                    raise ValueError(f"ท่า {name} ไม่มีตัวอย่าง")
                threshold = float(entry["threshold"])
                if not 0.0 < threshold <= 1.0:
                    raise ValueError(f"ท่า {name}: threshold {threshold} อยู่นอกช่วง")
                if any(p.name == name for p in poses):
                    raise ValueError(f"ชื่อท่า {name} ซ้ำ")
                poses.append(
                    PoseTemplate(name, action, samples, float(entry["orientation"]), threshold)
                )
            negatives = tuple(_vector(n) for n in data.get("negatives", []))
        except (KeyError, TypeError, ValueError) as exc:
            raise PoseFileError(f"รูปแบบไม่ถูกต้อง — {exc}") from exc
        if len(poses) > MAX_POSES:
            raise PoseFileError(f"มีท่าเกิน {MAX_POSES} ท่า")
        return cls(tuple(poses), negatives[-MAX_NEGATIVES:])


def _vector(values: object) -> Vector:
    if not isinstance(values, list) or len(values) != FEATURE_SIZE:
        raise ValueError(f"ตัวอย่างต้องมี {FEATURE_SIZE} ค่า")
    vector = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in vector):
        raise ValueError("ตัวอย่างมีค่าที่ไม่ใช่ตัวเลข")
    return vector


def load_poses(path: Path) -> PoseLibrary:
    """อ่าน ``poses.json`` — ไม่มีไฟล์ = ยังไม่ได้สอนท่า (คืน library ว่าง)"""
    if not path.is_file():
        return PoseLibrary()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PoseFileError(f"{path}: อ่านไม่ได้ — {exc} (ลบไฟล์นี้แล้วสอนท่าใหม่ได้)") from exc
    try:
        return PoseLibrary.from_dict(data)
    except PoseFileError as exc:
        raise PoseFileError(f"{path}: {exc} (ลบไฟล์นี้แล้วสอนท่าใหม่ได้)") from exc


def save_poses(path: Path, library: PoseLibrary) -> None:
    """เขียนแบบ atomic (เขียนไฟล์ชั่วคราวก่อนแล้วค่อยแทนที่)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(library.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PoseMatch:
    """ท่าที่ใกล้มือในเฟรมที่สุด — ``accepted=False`` พร้อมเหตุผลถ้าไม่ผ่านเกณฑ์

    ``reason``: ``"far"`` ไกลเกินเกณฑ์ | ``"neutral"`` ใกล้มือปกติมากกว่า |
    ``"ambiguous"`` ก้ำกึ่งกับท่าอื่น (ใช้แสดงในหน้าต่าง debug ตอนปรับ ``poses.sensitivity``)
    """

    name: str
    distance: float
    threshold: float
    accepted: bool
    reason: str = ""

    @property
    def label(self) -> str:
        return POSE_PREFIX + self.name


class PoseClassifier:
    """จำแนกมือหนึ่งข้างว่าตรงกับท่าที่สอนไว้ท่าไหน (pure, ไม่มี state ระหว่างเฟรม)

    ``sensitivity`` > 1 รับง่ายขึ้น (เกณฑ์กว้างขึ้น), < 1 เข้มขึ้น
    """

    def __init__(self, library: PoseLibrary, sensitivity: float = 1.0) -> None:
        self._library = library
        self._sensitivity = sensitivity

    @property
    def library(self) -> PoseLibrary:
        return self._library

    def __bool__(self) -> bool:
        return bool(self._library.poses)

    def classify(self, hand: Hand, width: int, height: int) -> PoseMatch | None:
        """``None`` = ไม่มีท่าที่สอนไว้ หรือมือหันคนละทิศกับทุกท่า"""
        if not self._library.poses:
            return None
        measured = pose_features(hand, width, height)
        if measured is None:
            return None
        return self.classify_features(*measured)

    def classify_features(self, features: Vector, orientation: float) -> PoseMatch | None:
        scored = sorted(
            (
                (knn_distance(p.samples, features), p)
                for p in self._library.poses
                if angle_difference(orientation, p.orientation) <= ORIENTATION_TOLERANCE
            ),
            key=lambda item: item[0],
        )
        if not scored:
            return None
        distance, pose = scored[0]
        threshold = pose.threshold * self._sensitivity

        def result(accepted: bool, reason: str = "") -> PoseMatch:
            return PoseMatch(pose.name, distance, threshold, accepted, reason)

        if distance > threshold:
            return result(False, "far")
        if len(scored) > 1 and distance > scored[1][0] * AMBIGUITY_RATIO:
            return result(False, "ambiguous")
        if knn_distance(self._library.negatives, features) <= distance:
            return result(False, "neutral")
        return result(True)

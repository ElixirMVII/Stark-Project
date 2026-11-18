"""สอนตัวจำแนกท่ามือล่วงหน้าด้วยข้อมูลสังเคราะห์ (หรือข้อมูลจริงจาก Hand lab) — pure, ไม่ต้องมีกล้อง

ขั้นตอน:

1. :func:`generate` — สุ่มมือสังเคราะห์ 3D (:mod:`aether.handmodel`) ของแต่ละท่าใน :data:`POSE_SPECS` แบบหลากหลาย:
   หมุน/เอียง/หันข้าง, ระยะห่างกล้อง, ตำแหน่งในภาพ, ขนาดและสัดส่วนมือ, งอนิ้วไม่เท่ากันทุกครั้ง, มือซ้าย/ขวา,
   และสัญญาณรบกวนแบบ landmark จริง → เวกเตอร์รูปทรงมือ (:func:`aether.poses.pose_features`)
2. :func:`train_pack` — เลือกตัวอย่างที่ "ครอบคลุม" ที่สุด ``MAX_SAMPLES`` ตัวต่อท่า (farthest-point sampling)
   + มือที่ไม่ใช่ท่าใดๆ เป็น negatives แล้วตั้งเกณฑ์ระยะของแต่ละท่าจากชุด validation (คนละ seed)
3. :func:`evaluate` — วัดผลกับชุดทดสอบ (คนละ seed) ได้ confusion matrix, recall ต่อท่า, อัตรารับผิด

ผลเป็น :class:`~aether.poses.PoseLibrary` แบบเดียวกับ ``poses.json`` (action = ``"none"``) — ผู้ใช้เลือกเพิ่ม
ท่าสำเร็จรูปเข้าชุดของตัวเองและผูกปุ่มลัดได้ในหน้าต่างสอนท่า
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import assets_dir
from .handmodel import HandShape, HandView, build_hand
from .poses import (
    MAX_NEGATIVES,
    MAX_SAMPLES,
    PoseClassifier,
    PoseFileError,
    PoseLibrary,
    PoseTemplate,
    Vector,
    knn_distance,
    pose_features,
)
from .tracker import Hand

__all__ = [
    "NEGATIVE_CLASSES",
    "PACK_POSES",
    "POSE_SPECS",
    "EvalReport",
    "PoseSpec",
    "builtin_pack_path",
    "evaluate",
    "farthest_points",
    "generate",
    "load_builtin_pack",
    "sample_hand",
    "train_from_samples",
    "train_pack",
]

NONE = "none"  # ป้ายของ "ไม่ใช่ท่าใด" ใน confusion matrix
THRESHOLD_QUANTILE = 0.95  # เกณฑ์ = ระยะที่ครอบตัวอย่าง validation ของท่านั้น 95%
THRESHOLD_MARGIN = 1.05
THRESHOLD_LIMITS = (0.10, 0.30)
# ตัวอย่างต่อท่าเท่ากับท่าที่สอนเอง — ลองเก็บ 100 ตัวแล้ว: ในชุดสังเคราะห์ดีขึ้นแต่เกณฑ์แคบลงจนมือที่ต่างไป
# (ทดสอบด้วยมือสังเคราะห์อีกแบบ) รับได้น้อยลง (ดู docs/DECISIONS.md D52)
PACK_SAMPLES = MAX_SAMPLES
FRAME = (640, 480)

EXT = (0.0, 0.12)  # นิ้วเหยียด
CURL = (0.85, 1.0)  # นิ้วงอเข้าฝ่ามือ
THUMB_OUT = (0.0, 0.15)
THUMB_IN = (0.8, 1.0)


@dataclass(frozen=True, slots=True)
class PoseSpec:
    curl: tuple[tuple[float, float], ...]  # ช่วงของ curl ต่อนิ้ว (โป้ง → ก้อย)
    description: str = ""
    spread: tuple[float, float] = (0.3, 0.9)
    gesture: str = "None"  # ท่าสำเร็จรูปของ MediaPipe ที่ใกล้เคียง (ข้อมูลอ้างอิง)


POSE_SPECS: dict[str, PoseSpec] = {
    # ---- ท่าสำเร็จรูปในชุด (ไม่ซ้ำกับท่าที่ MediaPipe/engine ใช้อยู่แล้ว) ----
    "rock": PoseSpec((THUMB_IN, EXT, CURL, CURL, EXT), "ชี้ + ก้อย 🤘"),
    "shaka": PoseSpec((THUMB_OUT, CURL, CURL, CURL, EXT), "โป้ง + ก้อย 🤙"),
    "gun": PoseSpec((THUMB_OUT, EXT, CURL, CURL, CURL), "โป้ง + ชี้ เป็นรูปตัว L"),
    "three": PoseSpec((THUMB_IN, EXT, EXT, EXT, CURL), "ชี้ + กลาง + นาง (สามนิ้ว)"),
    "four": PoseSpec(((0.85, 1.0), EXT, EXT, EXT, EXT), "สี่นิ้ว พับโป้งเข้าฝ่ามือ"),
    "pinky": PoseSpec((THUMB_IN, CURL, CURL, CURL, EXT), "ก้อยนิ้วเดียว"),
    # ---- มือที่ไม่ใช่ท่าใดๆ (negatives): ท่าที่ engine ใช้ + มือพัก + มือตอนจีบ ----
    "open": PoseSpec((THUMB_OUT, EXT, EXT, EXT, EXT), gesture="Open_Palm"),
    "fist": PoseSpec((THUMB_IN, CURL, CURL, CURL, CURL), gesture="Closed_Fist"),
    "point": PoseSpec((THUMB_IN, EXT, CURL, CURL, CURL), gesture="Pointing_Up"),
    "peace": PoseSpec((THUMB_IN, EXT, EXT, CURL, CURL), gesture="Victory"),
    "love": PoseSpec((THUMB_OUT, EXT, CURL, CURL, EXT), gesture="ILoveYou"),
    "relaxed": PoseSpec(((0.2, 0.6), (0.15, 0.5), (0.15, 0.55), (0.2, 0.6), (0.2, 0.6))),
    "pinch": PoseSpec(((0.45, 0.65), (0.4, 0.6), (0.1, 0.5), (0.1, 0.55), (0.1, 0.6))),
}
# "four" ไม่อยู่ในชุด: ต่างจากแบมือแค่นิ้วโป้ง — แบมือของมือสังเคราะห์อีกแบบถูกรับเป็น four ราว 10%
# (แบมือเป็นท่าที่เกิดบ่อยที่สุด จึงเสี่ยงที่สุด)
PACK_POSES = ("rock", "shaka", "gun", "three", "pinky")
NEGATIVE_CLASSES = ("open", "fist", "point", "peace", "love", "relaxed", "pinch")

# ความหลากหลายของมุมมอง (ช่วงสม่ำเสมอ)
VIEW = {
    "roll": (-25.0, 25.0),
    "pitch": (-25.0, 25.0),
    "yaw": (-30.0, 30.0),
    "distance": (0.35, 0.85),
    "x": (0.3, 0.7),
    "y": (0.5, 0.8),
    "size": (0.85, 1.15),
    "finger_length": (0.92, 1.08),
    "noise": (0.0015, 0.004),
}
JOINT_JITTER = 6.0  # องศา (σ) ต่อข้อนิ้ว
# domain randomization: ช่วงของตัวคูณมุมงอ (MCP, PIP, DIP) และการเลื่อนนิ้วโป้งที่พับ (x, y, z)
FLEX_SCALE = ((0.6, 1.05), (0.8, 1.05), (0.55, 1.15))
THUMB_BIAS = ((-0.3, 0.15), (-0.1, 0.1), (-0.15, 0.15))
THUMB_ANGLE = (35.0, 85.0)  # นิ้วโป้งที่กาง: ทแยง (แบมือ) … เกือบแนวนอน (ท่า L / 🤙)


def sample_hand(spec: PoseSpec, rng: random.Random, *, side: str | None = None) -> Hand:
    """มือสังเคราะห์หนึ่งข้างของท่า ``spec`` ที่สุ่มมุม/ขนาด/การงอ"""

    def uniform(key: str) -> float:
        low, high = VIEW[key]
        return rng.uniform(low, high)

    shape = HandShape(
        curl=tuple(rng.uniform(low, high) for low, high in spec.curl),  # type: ignore[arg-type]
        spread=rng.uniform(*spec.spread),
        size=uniform("size"),
        finger_length=uniform("finger_length"),
        jitter=tuple(rng.gauss(0.0, JOINT_JITTER) for _ in range(12)),
        flex_scale=tuple(rng.uniform(low, high) for low, high in FLEX_SCALE),  # type: ignore[arg-type]
        thumb_bias=tuple(rng.uniform(low, high) for low, high in THUMB_BIAS),  # type: ignore[arg-type]
        thumb_angle=rng.uniform(*THUMB_ANGLE),
    )
    view = HandView(
        x=uniform("x"),
        y=uniform("y"),
        distance=uniform("distance"),
        roll=uniform("roll"),
        pitch=uniform("pitch"),
        yaw=uniform("yaw"),
        side=side or rng.choice(("right", "left")),
        width=FRAME[0],
        height=FRAME[1],
        noise=uniform("noise"),
    )
    return build_hand(shape, view, rng=rng)


Sample = tuple[Vector, float]


def generate(names: Iterable[str], count: int, seed: int) -> dict[str, list[Sample]]:
    """``count`` ตัวอย่างต่อท่า → ``{ชื่อ: [(features, orientation), ...]}``"""
    rng = random.Random(seed)
    data: dict[str, list[Sample]] = {}
    for name in names:
        spec = POSE_SPECS[name]
        samples: list[Sample] = []
        while len(samples) < count:
            measured = pose_features(sample_hand(spec, rng), *FRAME)
            if measured is not None:
                samples.append(measured)
        data[name] = samples
    return data


def farthest_points(vectors: Sequence[Vector], count: int, seed: int = 0) -> list[int]:
    """เลือก ``count`` ตัวที่กระจายครอบคลุมที่สุด (k-center greedy) — ตัวแรกคือตัวที่ใกล้ค่าเฉลี่ย"""
    if len(vectors) <= count:
        return list(range(len(vectors)))
    data = np.asarray(vectors, dtype=np.float64)
    first = int(np.argmin(np.linalg.norm(data - data.mean(axis=0), axis=1)))
    chosen = [first]
    nearest = np.linalg.norm(data - data[first], axis=1)
    while len(chosen) < count:
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, np.linalg.norm(data - data[pick], axis=1))
    return chosen


def _circular_mean(angles: Iterable[float]) -> float:
    angles = list(angles)
    sx = sum(math.sin(math.radians(a)) for a in angles)
    cx = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sx, cx)) if angles else 0.0


def _template(
    name: str, samples: Sequence[Sample], threshold: float = 0.25, count: int = MAX_SAMPLES
) -> PoseTemplate:
    picked = [samples[i] for i in farthest_points([f for f, _ in samples], count)]
    vectors = tuple(tuple(round(v, 4) for v in f) for f, _ in picked)
    orientation = _circular_mean(angle for _, angle in picked)
    return PoseTemplate(name, NONE, vectors, round(orientation, 2), threshold)


def _negatives(data: dict[str, list[Sample]], names: Sequence[str]) -> tuple[Vector, ...]:
    per_class = max(1, MAX_NEGATIVES // max(len(names), 1))
    negatives: list[Vector] = []
    for name in names:
        vectors = [f for f, _ in data[name]]
        negatives += [vectors[i] for i in farthest_points(vectors, per_class)]
    return tuple(tuple(round(v, 4) for v in n) for n in negatives[:MAX_NEGATIVES])


def _tune_thresholds(library: PoseLibrary, validation: dict[str, list[Sample]]) -> PoseLibrary:
    """เกณฑ์ของแต่ละท่า = ระยะ k-NN ที่ครอบตัวอย่าง validation ของท่านั้น ``THRESHOLD_QUANTILE`` (+ ขอบ)"""
    tuned = []
    low, high = THRESHOLD_LIMITS
    for pose in library.poses:
        distances = sorted(knn_distance(pose.samples, f) for f, _ in validation.get(pose.name, []))
        if distances:
            q = distances[min(len(distances) - 1, int(THRESHOLD_QUANTILE * len(distances)))]
            threshold = min(max(q * THRESHOLD_MARGIN, low), high)
        else:
            threshold = pose.threshold
        tuned.append(
            PoseTemplate(
                pose.name, pose.action, pose.samples, pose.orientation, round(threshold, 4)
            )
        )
    return PoseLibrary(tuple(tuned), library.negatives)


def train_pack(
    train: dict[str, list[Sample]],
    validation: dict[str, list[Sample]],
    *,
    poses: Sequence[str] = PACK_POSES,
    negatives: Sequence[str] = NEGATIVE_CLASSES,
    samples_per_pose: int = PACK_SAMPLES,
) -> PoseLibrary:
    library = PoseLibrary(
        tuple(_template(name, train[name], count=samples_per_pose) for name in poses),
        _negatives(train, [n for n in negatives if n in train]),
    )
    return _tune_thresholds(library, validation)


@dataclass
class EvalReport:
    labels: list[str]  # แถว = ป้ายจริง
    predicted: list[str]  # คอลัมน์ = ท่าที่ตัวจำแนกรับ หรือ "none"
    matrix: dict[str, Counter] = field(default_factory=dict)

    def recall(self, label: str) -> float:
        row = self.matrix.get(label, Counter())
        total = sum(row.values())
        target = label if label in self.predicted else NONE
        return row[target] / total if total else 0.0

    def false_accept(self, label: str) -> float:
        """สัดส่วนของมือป้าย ``label`` ที่ถูกรับเป็นท่าที่ผิด"""
        row = self.matrix.get(label, Counter())
        total = sum(row.values())
        wrong = sum(n for p, n in row.items() if p not in (label, NONE))
        return wrong / total if total else 0.0

    @property
    def pose_recall(self) -> float:
        poses = [label for label in self.labels if label in self.predicted]
        return sum(self.recall(p) for p in poses) / max(len(poses), 1)

    @property
    def wrong_accept_rate(self) -> float:
        return sum(self.false_accept(label) for label in self.labels) / max(len(self.labels), 1)

    def lines(self) -> list[str]:
        columns = [*self.predicted, NONE]
        width = max(len(c) for c in [*columns, *self.labels]) + 2
        out = ["".ljust(width) + "".join(c.rjust(width) for c in columns) + "recall".rjust(9)]
        for label in self.labels:
            row = self.matrix.get(label, Counter())
            cells = "".join(str(row[c]).rjust(width) for c in columns)
            out.append(label.ljust(width) + cells + f"{self.recall(label):9.1%}")
        out.append("")
        out.append(f"recall เฉลี่ยของท่าในชุด: {self.pose_recall:.1%}")
        out.append(f"รับผิดท่าเฉลี่ย (ทุกป้าย): {self.wrong_accept_rate:.2%}")
        return out

    def to_dict(self) -> dict:
        return {
            "pose_recall": round(self.pose_recall, 4),
            "wrong_accept_rate": round(self.wrong_accept_rate, 4),
            "recall": {label: round(self.recall(label), 4) for label in self.labels},
        }


def evaluate(library: PoseLibrary, test: dict[str, list[Sample]]) -> EvalReport:
    classifier = PoseClassifier(library)
    report = EvalReport(list(test), list(library.names()))
    for label, samples in test.items():
        row: Counter = Counter()
        for features, orientation in samples:
            match = classifier.classify_features(features, orientation)
            row[match.name if match is not None and match.accepted else NONE] += 1
        report.matrix[label] = row
    return report


# --------------------------------------------------------------------------- #
# ชุดท่าสำเร็จรูปที่แถมมากับแอป
# --------------------------------------------------------------------------- #


def builtin_pack_path() -> Path:
    return assets_dir() / "poses" / "builtin.json"


def load_builtin_pack(path: Path | None = None) -> tuple[PoseLibrary, dict]:
    """ชุดท่าสำเร็จรูป + ข้อมูลประกอบ (คำอธิบายท่า ผลทดสอบ) — ไม่มีไฟล์ = ชุดว่าง"""
    path = path or builtin_pack_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PoseLibrary(), {}
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseFileError(f"อ่านชุดท่าสำเร็จรูปไม่ได้: {exc}") from exc
    return PoseLibrary.from_dict(data), dict(data.get("meta", {}))


def pack_document(library: PoseLibrary, report: EvalReport, *, seed: int, count: int) -> dict:
    document = library.to_dict()
    document["meta"] = {
        "source": "synthetic 3D hands (aether.handmodel) — ยังไม่ได้ยืนยันกับกล้องจริง",
        "seed": seed,
        "samples_per_pose": count,
        "descriptions": {name: POSE_SPECS[name].description for name in library.names()},
        "evaluation": report.to_dict(),
    }
    return document


# --------------------------------------------------------------------------- #
# สอนจากตัวอย่างจริงที่เก็บด้วย Hand lab
# --------------------------------------------------------------------------- #


def train_from_samples(
    records: Iterable[dict], *, holdout: float = 0.25, seed: int = 1
) -> tuple[PoseLibrary, EvalReport | None]:
    """สอนจากตัวอย่างใน ``hand_samples.jsonl`` (ป้าย ``none``/``neutral`` = มือปกติ เป็น negatives)

    แบ่ง ``holdout`` ของแต่ละป้ายไว้ทดสอบ (ถ้ามีตัวอย่างพอ) คืน (library, report หรือ None)
    """
    from .studio.lab import sample_hand as record_hand

    rng = random.Random(seed)
    by_label: dict[str, list[Sample]] = {}
    for record in records:
        hand = record_hand(record)
        measured = pose_features(
            hand, int(record.get("width", 640)), int(record.get("height", 480))
        )
        if measured is not None:
            by_label.setdefault(str(record["label"]).strip().lower(), []).append(measured)
    train: dict[str, list[Sample]] = {}
    test: dict[str, list[Sample]] = {}
    for label, samples in by_label.items():
        rng.shuffle(samples)
        cut = int(len(samples) * holdout) if len(samples) >= 8 else 0
        test[label], train[label] = samples[:cut], samples[cut:]
    neutral = [name for name in train if name in ("none", "neutral", "relaxed")]
    poses = [name for name in train if name not in neutral]
    if not poses:
        raise ValueError("ไม่มีตัวอย่างของท่าใดเลย (มีแต่มือปกติ)")
    library = PoseLibrary(
        tuple(_template(name, train[name]) for name in poses),
        _negatives(train, neutral) if neutral else (),
    )
    validation = {name: test[name] or train[name] for name in poses}
    library = _tune_thresholds(library, validation)
    test = {label: samples for label, samples in test.items() if samples}
    return library, (evaluate(library, test) if test else None)

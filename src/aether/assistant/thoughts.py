"""โครงข่ายความคิดของ Claude (pure — ไม่มี Qt): ข้อความ "สรุปความคิด" ที่ stream มาจาก API → โหนด + เส้นเชื่อม

ที่มาของข้อมูล: Claude API ส่ง *สรุป* ของการคิด (``thinking.display = "summarized"``) ไม่ใช่การทำงานของ
เซลล์ประสาทจริง — โครงข่ายนี้เป็นภาพที่แอปสร้างจากข้อความนั้น:

- แต่ละประโยค = หนึ่งโหนด, หัวข้อตัวหนา (``**Planning the answer**``) = โหนดศูนย์กลาง (hub) ที่ประโยคถัดไปห้อยอยู่
- เส้น ``chain`` = ลำดับความคิด, ``hub`` = หัวข้อต่อหัวข้อ, ``link`` = สองความคิดพูดถึงคำสำคัญเดียวกัน,
  ``answer`` = ความคิดที่นำไปสู่คำตอบ
- จัดวางแบบ force-directed (สปริงตามเส้น + แรงผลักระหว่างโหนด) ขยับต่อเนื่องเหมือนมีชีวิต ลากโหนดได้
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

__all__ = ["Edge", "Thought", "ThoughtGraph", "keywords", "split_thoughts"]

MAX_NODES = 60
MIN_CHARS = 12  # ประโยคสั้นกว่านี้ไปรวมกับประโยคถัดไป
MAX_CHARS = 220  # ยาวกว่านี้ตัดที่จุลภาค/ช่องว่าง
EDGE_LENGTH = 1.0
HUB_LENGTH = 1.8
REPULSION = 0.9
SPRING = 2.2
GRAVITY = 0.08
DAMPING = 0.86
SPAWN_DISTANCE = 0.6
BIRTH_SECONDS = 0.9
CALM_ENERGY = 0.08  # พลังงานต่ำกว่านี้ = เกือบนิ่ง
STILL_ENERGY = 5e-3  # ต่ำกว่านี้ขยับไม่ถึง ~2 px/วินาที = ไม่ต้องวาดใหม่

_STOPWORDS = frozenset(
    [
        "about",
        "above",
        "after",
        "again",
        "against",
        "also",
        "although",
        "among",
        "another",
        "answer",
        "anything",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "cannot",
        "could",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "either",
        "else",
        "even",
        "every",
        "few",
        "first",
        "from",
        "further",
        "given",
        "going",
        "have",
        "having",
        "here",
        "into",
        "itself",
        "just",
        "know",
        "like",
        "likely",
        "made",
        "make",
        "many",
        "maybe",
        "might",
        "more",
        "most",
        "much",
        "must",
        "need",
        "needs",
        "next",
        "only",
        "other",
        "otherwise",
        "over",
        "perhaps",
        "please",
        "quite",
        "rather",
        "really",
        "same",
        "should",
        "since",
        "some",
        "something",
        "still",
        "such",
        "sure",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "this",
        "those",
        "though",
        "through",
        "thus",
        "together",
        "under",
        "until",
        "upon",
        "user",
        "users",
        "very",
        "want",
        "wants",
        "well",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "will",
        "with",
        "within",
        "without",
        "would",
        "your",
        "yours",
        "let's",
        "okay",
        "first",
        "second",
        "also",
        "maybe",
        "should",
        "probably",
        "actually",
        "basically",
        "going",
    ]
)
_WORD = re.compile(
    r"0x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9\-']{3,}|\d+(?:\.\d+)?|[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*"
)
_BOUNDARY = re.compile(r"(?<=[.!?。])\s+|\n+")
_BOLD = re.compile(r"^\*\*(.+?)\*\*:?$")


def keywords(text: str) -> frozenset[str]:
    """คำสำคัญ (อังกฤษ ≥ 4 ตัว ไม่ใช่คำทั่วไป, ตัวเลข, สูตรเคมี) — ภาษาไทยไม่มีช่องว่างจึงใช้เฉพาะคำละตินที่ปนอยู่"""
    found = set()
    for word in _WORD.findall(text):
        lowered = word.lower().strip("-'")
        lowered = lowered.removesuffix("'s")
        if lowered in _STOPWORDS or len(lowered) < 3:
            continue
        if lowered.endswith("s") and len(lowered) > 4 and not lowered.endswith(("ss", "us", "is")):
            lowered = lowered[:-1]  # plural แบบง่าย: errors → error
        found.add(lowered)
    return frozenset(found)


def _clean(segment: str) -> tuple[str, bool]:
    """ตัด markdown ออก คืน (ข้อความ, เป็นหัวข้อไหม)"""
    text = segment.strip()
    title = bool(_BOLD.match(text))
    if title:
        text = _BOLD.match(text).group(1)  # type: ignore[union-attr]
    text = re.sub(r"^[#>\-*•\d.)\s]+(?=\S)", "", text) if not title else text
    text = text.replace("**", "").replace("`", "").strip()
    return text, title


def _split_long(text: str) -> list[str]:
    parts = []
    while len(text) > MAX_CHARS:
        cut = text.rfind(", ", 0, MAX_CHARS)
        if cut < MAX_CHARS // 2:
            cut = text.rfind(" ", 0, MAX_CHARS)
        if cut < MAX_CHARS // 3:
            cut = MAX_CHARS
        parts.append(text[: cut + 1].strip())
        text = text[cut + 1 :].strip()
    if text:
        parts.append(text)
    return parts


def split_thoughts(text: str) -> list[tuple[str, bool]]:
    """แบ่งข้อความทั้งก้อนเป็นความคิด ``[(ข้อความ, เป็นหัวข้อ)]`` (ใช้กับข้อความที่จบแล้ว)"""
    out: list[tuple[str, bool]] = []
    pending = ""
    for segment in _BOUNDARY.split(text):
        cleaned, title = _clean(segment)
        if not cleaned:
            continue
        if title:
            if pending:
                out.extend((part, False) for part in _split_long(pending))
                pending = ""
            out.append((cleaned, True))
            continue
        pending = f"{pending} {cleaned}".strip() if pending else cleaned
        if len(pending) >= MIN_CHARS:
            out.extend((part, False) for part in _split_long(pending))
            pending = ""
    if pending:
        out.extend((part, False) for part in _split_long(pending))
    return out


@dataclass
class Thought:
    id: int
    text: str
    kind: str  # "question" | "hub" | "thought" | "answer"
    keywords: frozenset[str]
    born: float
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    pinned: bool = False

    @property
    def weight(self) -> float:
        """ขนาดโหนด 0–1: หัวข้อ/คำถาม/คำตอบใหญ่ ความคิดตามจำนวนคำสำคัญ"""
        if self.kind in ("question", "answer"):
            return 1.0
        if self.kind == "hub":
            return 0.8
        return min(0.6, 0.25 + 0.05 * len(self.keywords))


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    kind: str  # "chain" | "hub" | "link" | "answer"
    born: float


@dataclass
class _View:
    cx: float = 0.0
    cy: float = 0.0
    span_x: float = 3.0  # ความกว้างของพื้นที่ (หน่วยโครงข่าย) ที่ต้องเห็น
    span_y: float = 3.0

    @property
    def span(self) -> float:
        return max(self.span_x, self.span_y)


@dataclass
class ThoughtGraph:
    max_nodes: int = MAX_NODES
    seed: int = 7
    nodes: list[Thought] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    thinking: str = ""  # ข้อความสรุปความคิดทั้งหมดที่ได้มา

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._buffer = ""
        self._hub: Thought | None = None
        self._last: Thought | None = None
        self._dragging: int | None = None
        self._energy = 1.0
        self.view = _View()

    # ---- สร้างโครงข่าย ------------------------------------------------------- #

    def reset(self, question: str = "", now: float = 0.0) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.thinking = ""
        self._buffer = ""
        self._hub = self._last = None
        self._dragging = None
        self.view = _View()
        self._rng = random.Random(self.seed)
        if question:
            root = Thought(0, question.strip(), "question", keywords(question), now, pinned=True)
            self.nodes.append(root)
            self._last = root

    @property
    def root(self) -> Thought | None:
        return self.nodes[0] if self.nodes and self.nodes[0].kind == "question" else None

    @property
    def thoughts(self) -> list[Thought]:
        return [n for n in self.nodes if n.kind in ("hub", "thought")]

    @property
    def answer(self) -> Thought | None:
        return next((n for n in reversed(self.nodes) if n.kind == "answer"), None)

    def feed(self, delta: str, now: float) -> list[Thought]:
        """ข้อความความคิดชิ้นใหม่จาก stream → คืนโหนดที่เกิดใหม่ (ประโยคที่จบแล้วเท่านั้น)"""
        self.thinking += delta
        self._buffer += delta
        pieces = _BOUNDARY.split(self._buffer)
        if len(pieces) == 1:
            return []
        self._buffer = pieces[-1]  # ส่วนท้ายยังพิมพ์ไม่จบ รอชิ้นต่อไป
        return self._add_segments("\n".join(pieces[:-1]), now)

    def flush(self, now: float) -> list[Thought]:
        """จบการคิด: ข้อความที่ค้างอยู่กลายเป็นโหนด"""
        rest, self._buffer = self._buffer, ""
        return self._add_segments(rest, now)

    def add_answer(self, text: str, now: float) -> Thought | None:
        self.flush(now)
        if not text.strip() or self.answer is not None:
            return None
        short = text.strip().splitlines()[0][:160]
        node = self._spawn(short, "answer", keywords(text), now, self._last)
        if self._last is not None:
            self.edges.append(Edge(self._last.id, node.id, "answer", now))
        scored = sorted(
            (len(node.keywords & n.keywords), n.id) for n in self.thoughts if n is not self._last
        )
        for score, other in reversed(scored[-3:]):
            if score >= 1:
                self.edges.append(Edge(other, node.id, "answer", now))
        return node

    def _add_segments(self, text: str, now: float) -> list[Thought]:
        born = []
        for segment, title in split_thoughts(text):
            if len(self.thoughts) >= self.max_nodes:
                last = self._last
                if last is not None and last.kind == "thought":  # เต็มแล้ว: ต่อท้ายความคิดล่าสุดแทน
                    last.text = (last.text + " " + segment)[:400]
                    last.keywords = last.keywords | keywords(segment)
                continue
            born.append(self._add(segment, title, now))
        return born

    def _add(self, text: str, title: bool, now: float) -> Thought:
        words = keywords(text)
        if title:
            parent = self._hub or self.root or self._last
            node = self._spawn(text, "hub", words, now, parent)
            if parent is not None:
                self.edges.append(Edge(parent.id, node.id, "hub", now))
            self._hub = node
        else:
            parent = self._last or self.root
            node = self._spawn(text, "thought", words, now, parent)
            if parent is not None:
                self.edges.append(Edge(parent.id, node.id, "chain", now))
            self._link(node, exclude={parent.id} if parent is not None else set(), now=now)
        self._last = node
        return node

    def _link(self, node: Thought, exclude: set[int], now: float) -> None:
        """เชื่อมกับความคิดก่อนหน้าที่พูดถึงคำสำคัญเดียวกัน (สูงสุด 2 เส้น)"""
        candidates = []
        others = self.thoughts + ([self.root] if self.root is not None else [])
        for other in others:
            if other.id == node.id or other.id in exclude:
                continue
            shared = node.keywords & other.keywords
            if len(shared) >= 2 or any(len(word) >= 7 for word in shared):
                candidates.append((len(shared), other.id))
        for _score, other in sorted(candidates, reverse=True)[:2]:
            self.edges.append(Edge(other, node.id, "link", now))

    def _spawn(
        self, text: str, kind: str, words: frozenset[str], now: float, parent: Thought | None
    ) -> Thought:
        angle = self._rng.uniform(0.0, 2.0 * math.pi)
        distance = SPAWN_DISTANCE * (1.6 if kind == "hub" else 1.0)
        px, py = (parent.x, parent.y) if parent is not None else (0.0, 0.0)
        node = Thought(
            len(self.nodes),
            text,
            kind,
            words,
            now,
            px + math.cos(angle) * distance,
            py + math.sin(angle) * distance,
        )
        self.nodes.append(node)
        self._energy = 1.0  # มีโหนดใหม่: กลับมาเด้งแบบมีชีวิต
        return node

    # ---- จัดวาง (force-directed) ---------------------------------------------- #

    def step(self, dt: float) -> float:
        """ขยับหนึ่งช่วงเวลา คืนพลังงานจลน์ (ใกล้ 0 = นิ่งแล้ว)"""
        dt = min(max(dt, 0.0), 0.05)
        if not self.nodes or dt == 0.0:
            return 0.0
        forces = {n.id: [0.0, 0.0] for n in self.nodes}
        nodes = self.nodes
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                dx, dy = a.x - b.x, a.y - b.y
                d2 = dx * dx + dy * dy
                if d2 < 1e-6:
                    dx, dy, d2 = self._rng.uniform(-0.01, 0.01), 0.01, 1e-4
                if d2 > 36.0:  # ไกลมาก: แรงผลักน้อยจนไม่ต้องคิด
                    continue
                strength = REPULSION / d2
                d = math.sqrt(d2)
                fx, fy = dx / d * strength, dy / d * strength
                forces[a.id][0] += fx
                forces[a.id][1] += fy
                forces[b.id][0] -= fx
                forces[b.id][1] -= fy
        by_id = {n.id: n for n in nodes}
        for edge in self.edges:
            a, b = by_id[edge.a], by_id[edge.b]
            dx, dy = b.x - a.x, b.y - a.y
            d = math.hypot(dx, dy) or 1e-3
            rest = HUB_LENGTH if edge.kind == "hub" else EDGE_LENGTH
            stiffness = SPRING * (0.35 if edge.kind == "link" else 1.0)
            pull = stiffness * (d - rest)
            fx, fy = dx / d * pull, dy / d * pull
            forces[a.id][0] += fx
            forces[a.id][1] += fy
            forces[b.id][0] -= fx
            forces[b.id][1] -= fy
        energy = 0.0
        # เกือบนิ่งแล้ว: หน่วงแรงขึ้นให้หยุดสนิท (ไม่ต้องวาดใหม่ทุกเฟรมไปเรื่อยๆ) — ยังเด้งตามปกติตอนมีโหนดใหม่/ถูกลาก
        damping = DAMPING if self._energy > CALM_ENERGY or self._dragging is not None else 0.6
        for node in nodes:
            if node.pinned or node.id == self._dragging:
                node.vx = node.vy = 0.0
                continue
            fx, fy = forces[node.id]
            fx -= GRAVITY * node.x
            fy -= GRAVITY * node.y
            node.vx = (node.vx + fx * dt) * damping
            node.vy = (node.vy + fy * dt) * damping
            speed = math.hypot(node.vx, node.vy)
            if speed > 4.0:  # กันระเบิด
                node.vx, node.vy = node.vx / speed * 4.0, node.vy / speed * 4.0
            node.x += node.vx * dt
            node.y += node.vy * dt
            energy += node.vx * node.vx + node.vy * node.vy
        self._fit_view(dt)
        self._energy = energy
        return energy

    def _fit_view(self, dt: float) -> None:
        left, top, right, bottom = self.bounds()
        blend = 1.0 - math.exp(-dt * 3.0)
        view = self.view
        view.cx += ((left + right) / 2.0 - view.cx) * blend
        view.cy += ((top + bottom) / 2.0 - view.cy) * blend
        view.span_x += (max(right - left, 1.5) + 1.2 - view.span_x) * blend
        view.span_y += (max(bottom - top, 1.5) + 1.2 - view.span_y) * blend

    def settle(self, steps: int = 200, dt: float = 0.03) -> float:
        energy = 0.0
        for _ in range(steps):
            energy = self.step(dt)
        return energy

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.nodes:
            return (-1.0, -1.0, 1.0, 1.0)
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (min(xs), min(ys), max(xs), max(ys))

    def active(self, now: float, energy: float = 0.0) -> bool:
        """ยังต้องวาดใหม่ไหม (เพิ่งเกิดโหนดใหม่ หรือยังขยับอยู่)"""
        recent = any(now - n.born < BIRTH_SECONDS for n in self.nodes)
        return recent or energy > STILL_ENERGY or self._dragging is not None

    # ---- พิกัดจอ / ลากโหนด ----------------------------------------------------- #

    def scales(self, width: float, height: float, *, stretch: bool = False) -> tuple[float, float]:
        """px ต่อหน่วยโครงข่าย (แกน x, y) — ปกติเท่ากันสองแกน ``stretch`` = ยืดเต็มกรอบ (แถบเล็กบน HUD)"""
        sx = width / max(self.view.span_x, 1e-3)
        sy = height / max(self.view.span_y, 1e-3)
        if stretch:
            return sx, min(sy, sx)
        scale = min(sx, sy)
        return scale, scale

    def to_screen(
        self,
        x: float,
        y: float,
        left: float,
        top: float,
        width: float,
        height: float,
        *,
        stretch: bool = False,
    ) -> tuple[float, float]:
        sx, sy = self.scales(width, height, stretch=stretch)
        return (
            left + width / 2.0 + (x - self.view.cx) * sx,
            top + height / 2.0 + (y - self.view.cy) * sy,
        )

    def from_screen(
        self, sx: float, sy: float, left: float, top: float, width: float, height: float
    ) -> tuple[float, float]:
        scale_x, scale_y = self.scales(width, height)
        return (
            self.view.cx + (sx - left - width / 2.0) / scale_x,
            self.view.cy + (sy - top - height / 2.0) / scale_y,
        )

    def nearest(self, x: float, y: float, radius: float) -> Thought | None:
        """โหนดที่ใกล้จุด (หน่วยโครงข่าย) ที่สุดภายใน ``radius``"""
        best, best_d = None, radius
        for node in self.nodes:
            d = math.hypot(node.x - x, node.y - y)
            if d <= best_d:
                best, best_d = node, d
        return best

    def grab(self, node_id: int) -> None:
        self._dragging = node_id
        self._energy = 1.0

    def drag_to(self, x: float, y: float) -> None:
        if self._dragging is None:
            return
        node = self.nodes[self._dragging]
        node.x, node.y = x, y
        node.vx = node.vy = 0.0

    def release(self) -> int | None:
        node, self._dragging = self._dragging, None
        self._energy = 1.0  # ปล่อยแล้วดีดกลับ
        return node

    @property
    def dragging(self) -> int | None:
        return self._dragging

    def neighbors(self, node_id: int) -> set[int]:
        return {e.b if e.a == node_id else e.a for e in self.edges if node_id in (e.a, e.b)}

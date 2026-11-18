"""ข้อความระบบ (system prompt) และตัวสร้างคำขอของแต่ละฟีเจอร์ — ไม่มีอะไรมาจากกล้อง

ทุกฟังก์ชันคืน ``(system, content)`` ให้ :class:`aether.assistant.service.ClaudeService` ส่งต่อ
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .service import image_block

__all__ = [
    "BASE_SYSTEM",
    "config_request",
    "drawing_request",
    "element_request",
    "flask_request",
    "free_question",
    "gesture_question",
    "log_request",
    "quiz_request",
    "recipe_request",
]

BASE_SYSTEM = (
    "You are Claude, built into AETHER — a Windows app that lets people control their PC with hand "
    "gestures in front of a webcam, plus a creative 'Studio' (air drawing, physics sandbox, theremin, "
    "a virtual periodic table and mixing flask). You never see the camera; you only get the text or "
    "drawings the user chose to send.\n"
    "Reply in Thai. Keep technical terms, chemical formulas, file names, config keys and key names in "
    "English. Be concise and friendly: answers appear on a small card, so prefer 2–6 short sentences "
    "or a few short bullets. Plain text only — no markdown headings, tables or code fences."
)

_CHEM_SYSTEM = (
    BASE_SYSTEM
    + "\n\nThe user is playing in the Elements lab, a simulation for curiosity — atoms are "
    "dropped into a virtual flask and the app recognises compounds by their atom counts. Explain "
    "chemistry at the level of formulas, structure, properties, history, everyday uses and why "
    "reactions happen (including dramatic or dangerous ones — that is part of the fun). Do not give "
    "real-world procedures: no step-by-step instructions, quantities, conditions or sourcing for "
    "making hazardous substances."
)


def free_question(question: str) -> tuple[str, str]:
    return BASE_SYSTEM, question.strip()


def gesture_question(clipboard: str) -> tuple[str, str]:
    """ท่ามือ ask_claude: ส่งข้อความที่ผู้ใช้ copy ไว้"""
    text = clipboard.strip()
    if len(text) > 12000:
        text = text[:12000] + "\n…(ตัดเหลือ 12,000 ตัวอักษรแรก)"
    content = (
        "I copied this text and made a hand gesture to ask you about it. If it is a question, answer "
        "it. If it is an error message, explain the likely cause and the fix. Otherwise summarise it "
        "and point out anything important. Fit it on a small overlay card.\n\n"
        f"<copied_text>\n{text}\n</copied_text>"
    )
    return BASE_SYSTEM, content


# ---- Elements --------------------------------------------------------------- #


def element_request(info: Mapping[str, Any], question: str | None = None) -> tuple[str, str]:
    facts = json.dumps(dict(info), ensure_ascii=False)
    ask = (
        question.strip()
        if question
        else (
            "Tell me about this element: what it looks/behaves like, where it is found or used, and one "
            "surprising fact. 4–5 sentences."
        )
    )
    return _CHEM_SYSTEM, f"Element shown in the app: {facts}\n\n{ask}"


def flask_request(
    counts: Mapping[str, int], formula: str, match: str | None, question: str | None = None
) -> tuple[str, str]:
    atoms = ", ".join(f"{symbol}×{n}" for symbol, n in counts.items()) or "(empty)"
    known = f"The app recognised it as: {match}." if match else "The app has no recipe for it."
    ask = (
        question.strip()
        if question
        else (
            "What could these atoms form? If it is a known compound, explain it; if not, suggest the "
            "closest real compound(s) and which atoms to add or remove. 3–5 sentences."
        )
    )
    return _CHEM_SYSTEM, f"Atoms in the flask: {atoms} (formula {formula}). {known}\n\n{ask}"


def recipe_request(
    known_formulas: Iterable[str], focus: str | None, count: int = 5
) -> tuple[str, str]:
    known = ", ".join(sorted(set(known_formulas)))
    about = f" that contain {focus}" if focus else ""
    content = (
        f"Suggest {count} real compounds{about} that are NOT in this list (the app already has them): "
        f"{known}\n\nMix interesting everyday ones with fun or dramatic ones. Each formula must be a "
        "plain neutral formula using element symbols, digits and parentheses only (no charges, dots or "
        "hydrates), with at most 24 atoms in total. kind is one of everyday, fun, toxic, corrosive, "
        "explosive. danger: 0 safe, 1 caution, 2 dangerous, 3 extreme. fact: one Thai sentence."
    )
    return _CHEM_SYSTEM, content


def quiz_request(focus: str | None, count: int = 3) -> tuple[str, str]:
    about = (
        f" about {focus} and its compounds" if focus else " about the periodic table and compounds"
    )
    content = (
        f"Make a {count}-question multiple-choice quiz{about} for a curious beginner, in Thai. Each "
        "question has 3–4 short choices, exactly one correct; answer is the 0-based index of the correct "
        "choice; explanation is one Thai sentence."
    )
    return _CHEM_SYSTEM, content


# ---- Draw ------------------------------------------------------------------- #


def drawing_request(png: bytes, question: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    ask = (
        question.strip()
        if question
        else (
            "Guess what I drew (one line, start with your guess), then one kind sentence of feedback and "
            "one fun idea to add."
        )
    )
    content = [
        image_block(png),
        {
            "type": "text",
            "text": "I drew this in the air with my finger (hand tracking, so the lines are wobbly). "
            + ask,
        },
    ]
    return BASE_SYSTEM, content


# ---- ผู้ช่วยตั้งค่า / log ------------------------------------------------------ #

_CONFIG_SYSTEM = (
    BASE_SYSTEM
    + "\n\nYou help tune AETHER's config.toml. Only propose changes to keys that already "
    "exist in the file shown (section.name form, e.g. smoothing.min_cutoff), never to the [claude] "
    "section. Each value must be a valid TOML literal of the same type as the current value. Propose "
    "the smallest set of changes that addresses the request (at most 6); if nothing should change, "
    "return an empty list and explain why in summary. summary and reason are in Thai."
)


def config_request(config_text: str, request: str, notes: Sequence[str] = ()) -> tuple[str, str]:
    extra = ("\n\nNotes from recent sessions:\n" + "\n".join(notes)) if notes else ""
    content = (
        f"<config_toml>\n{config_text}\n</config_toml>{extra}\n\nWhat I want: {request.strip()}"
    )
    return _CONFIG_SYSTEM, content


def log_request(
    log_tail: str, report_lines: Sequence[str], question: str | None = None
) -> tuple[str, str]:
    report = "\n".join(report_lines) or "(no session statistics yet)"
    ask = (
        question.strip()
        if question
        else (
            "Read these and tell me: what problems you see (most important first), the likely cause of "
            "each, and what to change (config key or action). If everything looks healthy, say so."
        )
    )
    content = (
        f"<session_report>\n{report}\n</session_report>\n\n<aether_log>\n{log_tail}\n</aether_log>\n\n"
        f"{ask}"
    )
    return BASE_SYSTEM, content

"""รูปแบบคำตอบแบบมีโครงสร้าง (structured outputs) ที่ขอจาก Claude — ตรวจซ้ำฝั่งแอปก่อนใช้เสมอ"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ConfigChange",
    "ConfigProposal",
    "Quiz",
    "QuizQuestion",
    "RecipeIdea",
    "RecipeIdeas",
]


class RecipeIdea(BaseModel):
    formula: str = Field(description="สูตรเคมีแบบอะตอมล้วน เช่น NaCl, H2O, C6H12O6 (ไม่มีประจุ ไม่มีจุด)")
    name: str = Field(description="ชื่อภาษาอังกฤษ")
    thai: str = Field(description="ชื่อภาษาไทย")
    fact: str = Field(description="เกร็ดสั้นๆ ภาษาไทย 1 ประโยค")
    kind: Literal["everyday", "fun", "toxic", "corrosive", "explosive"]
    danger: int = Field(ge=0, le=3, description="0 ปลอดภัย … 3 อันตรายมาก")


class RecipeIdeas(BaseModel):
    ideas: list[RecipeIdea]


class QuizQuestion(BaseModel):
    question: str
    choices: list[str]
    answer: int = Field(description="index ของคำตอบที่ถูกใน choices (เริ่มที่ 0)")
    explanation: str


class Quiz(BaseModel):
    questions: list[QuizQuestion]


class ConfigChange(BaseModel):
    key: str = Field(description="คีย์แบบ section.name เช่น smoothing.min_cutoff")
    value: str = Field(
        description='ค่าใหม่ในรูป TOML literal เช่น 1.2, true, "ctrl+c", [0.1, 0.1, 0.9, 0.9]'
    )
    reason: str = Field(description="เหตุผลสั้นๆ ภาษาไทย")


class ConfigProposal(BaseModel):
    summary: str = Field(description="สรุปสั้นๆ ภาษาไทย")
    changes: list[ConfigChange]

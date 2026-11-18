"""เรียก Claude API (SDK ``anthropic``) — ใช้ใน worker thread เท่านั้น (ทุกเมธอด block จนได้คำตอบ)

- โมเดล/effort/max_tokens มาจาก ``[claude]`` ใน config.toml (ค่าเริ่มต้น ``claude-opus-5-5`` effort ``low``)
- streaming เสมอ (คำตอบยาวไม่ timeout และขึ้นทีละคำบน HUD ได้)
- เปิด server-side fallback (``fallbacks="default"``) — ถ้าระบบความปลอดภัยของโมเดลปฏิเสธคำขอ API จะลองโมเดลสำรอง
  ที่ Anthropic แนะนำให้เองในคำขอเดียวกัน
- ``stop_reason == "refusal"`` (ทั้งสายปฏิเสธ) → :class:`ClaudeError` พร้อมข้อความไทย
- ไม่มีอะไรในโมดูลนี้แตะกล้อง: ``content`` มาจากผู้เรียกเท่านั้น (ข้อความ / ภาพที่วาดเองใน Studio)
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from ..config import CLAUDE_EFFORTS, ClaudeConfig
from .secret import load_api_key

__all__ = [
    "FALLBACK_BETA",
    "ClaudeError",
    "ClaudeService",
    "image_block",
    "supports_fallback",
]

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T")
_UNSET: Any = object()
Content = str | Sequence[dict[str, Any]]


class ClaudeError(RuntimeError):
    """ข้อผิดพลาดที่พร้อมแสดงให้ผู้ใช้ (ข้อความไทย)"""


def supports_fallback(model: str) -> bool:
    """Haiku ไม่มี server-side fallback (ส่ง ``fallbacks`` ไปจะไม่มีผล/ถูกปฏิเสธ)"""
    return not model.startswith("claude-haiku")


def image_block(png: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png).decode("ascii"),
        },
    }


def _default_client_factory(api_key: str | None) -> Any:
    import anthropic  # import ตอนใช้จริง: แอปเปิดได้แม้ไม่มี SDK (ฟีเจอร์ Claude จะแจ้งว่าใช้ไม่ได้)

    return anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=120.0)


class ClaudeService:
    def __init__(
        self,
        config: ClaudeConfig,
        *,
        key_loader: Callable[[], str | None] = load_api_key,
        client_factory: Callable[[str | None], Any] | None = None,
    ) -> None:
        self._config = config
        self._key_loader = key_loader
        self._factory = client_factory or _default_client_factory
        self._client: Any = None
        self._key: Any = _UNSET  # อ่านไฟล์ key (+ถอด DPAPI) ครั้งเดียว — UI ถาม ready_reason ทุกเฟรม

    # ---- สถานะ ------------------------------------------------------------ #

    @property
    def config(self) -> ClaudeConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def update_config(self, config: ClaudeConfig) -> None:
        self._config = config

    def api_key(self) -> str | None:
        if self._key is _UNSET:
            self._key = self._key_loader()
        return self._key or os.environ.get("ANTHROPIC_API_KEY") or None

    def has_key(self) -> bool:
        return self.api_key() is not None

    def ready_reason(self) -> str | None:
        """``None`` = พร้อมใช้ ไม่งั้นคือเหตุผล (ข้อความไทย)"""
        if not self.enabled:
            return "ฟีเจอร์ Claude ปิดอยู่ — เปิดได้ที่ tray → ผู้ช่วย Claude…"
        if not self.has_key():
            return "ยังไม่มี API key — ใส่ได้ที่ tray → ผู้ช่วย Claude…"
        return None

    def reset_client(self) -> None:
        """เรียกหลังเปลี่ยน/ลบ API key (อ่าน key ใหม่ + สร้าง client ใหม่รอบหน้า)"""
        self._client = None
        self._key = _UNSET

    # ---- เรียก API --------------------------------------------------------- #

    def stream_text(
        self,
        system: str,
        content: Content,
        on_text: Callable[[str], None] | None = None,
        *,
        max_tokens: int | None = None,
        on_thinking: Callable[[str], None] | None = None,
        effort: str | None = None,
    ) -> str:
        """ส่งข้อความหนึ่งรอบ คืนข้อความคำตอบทั้งหมด (``on_text`` ได้ทีละชิ้นระหว่างทาง)

        ``on_thinking`` = อยากได้สรุปความคิด (``thinking.display = "summarized"``) ทีละชิ้น — ใช้วาดโครงข่ายความคิด
        ``effort`` = ความลึกเฉพาะคำขอนี้ (แทนค่าใน config)
        """
        kwargs = self._request(system, content, max_tokens, effort)
        if on_thinking is not None and self._config.show_thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        else:
            on_thinking = None  # ไม่ได้ขอ = ไม่ส่งต่อ (ปิด show_thinking ไว้)
        with self._guard(), self._get_client().beta.messages.stream(**kwargs) as stream:
            for event in stream:
                kind = getattr(event, "type", "")
                if kind == "text" and on_text is not None and event.text:
                    on_text(event.text)
                elif kind == "thinking" and on_thinking is not None and event.thinking:
                    on_thinking(event.thinking)
            message = stream.get_final_message()
        self._check(message)
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()

    def parse(
        self,
        system: str,
        content: Content,
        model_cls: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T:
        """คำตอบแบบโครงสร้าง (pydantic model) — ใช้กับสูตรผสมใหม่/ควิซ/ข้อเสนอแก้ config"""
        kwargs = self._request(system, content, max_tokens, None)
        with self._guard():
            message = self._get_client().beta.messages.parse(output_format=model_cls, **kwargs)
        self._check(message)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise ClaudeError("Claude ตอบกลับมาในรูปแบบที่อ่านไม่ได้ — ลองใหม่อีกครั้ง")
        return parsed

    # ---- internal --------------------------------------------------------- #

    def _request(
        self, system: str, content: Content, max_tokens: int | None, effort: str | None
    ) -> dict[str, Any]:
        reason = self.ready_reason()
        if reason is not None:
            raise ClaudeError(reason)
        cfg = self._config
        if effort is not None and effort not in CLAUDE_EFFORTS:
            raise ClaudeError(f"effort {effort!r} ใช้ไม่ได้")
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": max_tokens or cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort or cfg.effort},
        }
        if supports_fallback(cfg.model):
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = self._factory(self.api_key())
            except ImportError as exc:
                raise ClaudeError("ไม่พบไลบรารี anthropic — ติดตั้งด้วย pip install anthropic") from exc
        return self._client

    @staticmethod
    def _check(message: Any) -> None:
        reason = getattr(message, "stop_reason", None)
        if reason == "refusal":
            raise ClaudeError("Claude ปฏิเสธคำขอนี้ (ระบบความปลอดภัยของโมเดล) — ลองถามแบบอื่น")
        if reason == "max_tokens":
            log.info("คำตอบของ Claude ถูกตัดที่ max_tokens")

    def _guard(self) -> _ErrorGuard:
        return _ErrorGuard(self)


class _ErrorGuard:
    """แปลงข้อผิดพลาดของ SDK เป็น :class:`ClaudeError` ข้อความไทย (ไม่ log ตัว key)"""

    def __init__(self, service: ClaudeService) -> None:
        self._service = service

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, ClaudeError):
            return False
        message = _thai_error(exc)
        if message is None:
            return False
        if "API key" in message:
            self._service.reset_client()
        log.warning("เรียก Claude ไม่สำเร็จ: %s (%s)", message, type(exc).__name__)
        raise ClaudeError(message) from exc


def _thai_error(exc: BaseException) -> str | None:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return None
    if isinstance(exc, anthropic.AuthenticationError):
        return "API key ไม่ถูกต้องหรือถูกยกเลิก — ตรวจ key ใหม่ในหน้าต่างผู้ช่วย Claude"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "API key นี้ไม่มีสิทธิ์ใช้โมเดลที่ตั้งไว้ — ตรวจ [claude] model ใน config.toml"
    if isinstance(exc, anthropic.NotFoundError):
        return "ไม่พบโมเดลที่ตั้งไว้ — ตรวจ [claude] model ใน config.toml"
    if isinstance(exc, anthropic.RateLimitError):
        return "เรียกถี่เกินโควตา — รอสักครู่แล้วลองใหม่"
    if isinstance(exc, anthropic.APITimeoutError):
        return "Claude ตอบช้าเกินไป (timeout) — ลองใหม่อีกครั้ง"
    if isinstance(exc, anthropic.APIConnectionError):
        return "ต่อ Claude ไม่ได้ — ตรวจอินเทอร์เน็ต/proxy/firewall"
    if isinstance(exc, anthropic.BadRequestError):
        return f"คำขอไม่ถูกต้อง: {getattr(exc, 'message', exc)}"
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", "?")
        return f"Claude API ขัดข้อง (HTTP {status}) — ลองใหม่ภายหลัง"
    if isinstance(exc, anthropic.AnthropicError):
        return f"เรียก Claude ไม่สำเร็จ: {exc}"
    return None

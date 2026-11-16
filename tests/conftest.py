"""Fixture กลาง — test ทั้งหมดรันได้โดยไม่ต้องมีกล้อง จอ หรือไฟล์โมเดล"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ให้ Qt ทำงานได้บนเครื่องที่ไม่มีจอ (CI / container)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tool(name: str):
    """import สคริปต์ใน ``tools/`` (ไม่ได้เป็น package) เป็น module"""
    module_name = f"aether_tools_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(["aether-tests"])

#!/usr/bin/env python
"""สอนชุดท่าสำเร็จรูปด้วยมือสังเคราะห์ 3D (หรือสอนจากตัวอย่างจริงที่เก็บด้วย Hand lab ของ Studio)

    python tools/train_poses.py                         # สร้าง assets/poses/builtin.json + docs/POSE_PACK.md
    python tools/train_poses.py --count 600 --seed 7    # ข้อมูลมากขึ้น / seed อื่น
    python tools/train_poses.py --samples hand_samples.jsonl --out poses_trained.json

แบบ ``--samples``: ป้าย ``none`` / ``neutral`` / ``relaxed`` = มือปกติ (ใช้เป็น negatives) ป้ายอื่นเป็นท่า —
แบ่ง 25% ของแต่ละป้ายไว้ทดสอบ แล้วเขียนไฟล์แบบ poses.json (ปุ่มลัด = "none" ไปผูกต่อในหน้าต่างสอนท่า
หรือแก้ในไฟล์) ไม่มีภาพใดๆ ในขั้นตอนนี้ — ใช้แค่ตัวเลข landmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aether.studio.lab import read_samples
from aether.training import (
    NEGATIVE_CLASSES,
    PACK_POSES,
    POSE_SPECS,
    evaluate,
    generate,
    pack_document,
    train_from_samples,
    train_pack,
)

VALIDATION_SEED_OFFSET = 1000
TEST_SEED_OFFSET = 2000


def build_pack(count: int, seed: int, out: Path, report_path: Path | None) -> int:
    names = [*PACK_POSES, *NEGATIVE_CLASSES]
    started = time.perf_counter()
    train = generate(names, count, seed)
    validation = generate(names, max(count // 2, 50), seed + VALIDATION_SEED_OFFSET)
    test = generate(names, max(count // 2, 50), seed + TEST_SEED_OFFSET)
    library = train_pack(train, validation)
    report = evaluate(library, test)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = pack_document(library, report, seed=seed, count=count)
    out.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    lines = report.lines()
    print("\n".join(lines))
    print(
        f"\nเขียน {out} ({len(library)} ท่า, {len(library.negatives)} negatives) ใน {elapsed:.1f} วินาที"
    )
    if report_path is not None:
        write_report(report_path, library, lines, count=count, seed=seed)
        print(f"เขียน {report_path}")
    return 0


def write_report(path: Path, library, lines: list[str], *, count: int, seed: int) -> None:
    rows = "\n".join(
        f"| `{p.name}` | {POSE_SPECS[p.name].description} | {p.threshold:.3f} | {p.orientation:+.1f}° |"
        for p in library.poses
    )
    text = f"""# ชุดท่าสำเร็จรูป (สร้างโดย `tools/train_poses.py`)

สร้างจากมือสังเคราะห์ 3 มิติ (`aether.handmodel`) — **ยังไม่ได้ยืนยันกับกล้องจริง**
ข้อมูลสอน {count} ตัวอย่าง/ป้าย (seed {seed}), validation และชุดทดสอบใช้ seed ต่างกัน
ความหลากหลาย: หมุน ±25°, เอียงเข้า/ออก ±25°, หันข้าง ±30°, ห่างกล้อง 0.35–0.85 ม., ขนาดมือ ±15%,
สัดส่วนนิ้ว ±8%, มุมข้อนิ้วสุ่ม (σ 6°, งอโคนนิ้ว 60–105%), ตำแหน่งนิ้วโป้งสุ่ม, มือซ้าย/ขวา, noise แบบ landmark จริง

| ท่า | ลักษณะ | เกณฑ์ระยะ | ทิศเฉลี่ย |
|---|---|---|---|
{rows}

negatives (มือที่ต้องไม่ถูกรับเป็นท่าใด): {", ".join(NEGATIVE_CLASSES)}

## ผลกับชุดทดสอบ (มือสังเคราะห์ที่ไม่เคยเห็นตอนสอน)

แถว = ท่าจริง, คอลัมน์ = ท่าที่ตัวจำแนกรับ (`none` = ไม่รับ)

```
{chr(10).join(lines)}
```

## ข้อควรรู้ (domain gap)

ทดสอบกับมือสังเคราะห์อีกแบบที่วาดนิ้วต่างไป (`tests/posesim.py` — ไม่ได้ใช้สอน) ได้ recall เฉลี่ยราว 65%
(pinky ~96%, gun ~84%, three ~75%, rock ~69%, **shaka ~0%** เพราะวาดนิ้วโป้งต่างกันมาก) และ **ไม่รับผิดเลย**
(แบมือ กำมือ ชี้ ✌ 🤟 สี่นิ้ว ไม่ถูกรับเป็นท่าใดในชุด) — คือพลาดแบบปลอดภัย: ไม่ติด แต่ไม่ติดผิด

มือจริงจากกล้องอาจต่างจากมือสังเคราะห์ทั้งสองแบบ ถ้าท่าไหนติดยาก: สอนทับด้วยชื่อเดียวกันในหน้าต่างสอนท่า
หรือเก็บตัวอย่างจริงด้วย Hand lab ของ Studio แล้วรัน `python tools/train_poses.py --samples hand_samples.jsonl`
"""
    path.write_text(text, encoding="utf-8")


def from_samples(path: Path, out: Path) -> int:
    records = read_samples(path)
    if not records:
        print(f"ไม่มีตัวอย่างใน {path}", file=sys.stderr)
        return 1
    try:
        library, report = train_from_samples(records)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out.write_text(json.dumps(library.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(records)} ตัวอย่าง → {len(library)} ท่า: {', '.join(library.names())}")
    if report is not None:
        print("\n".join(report.lines()))
    else:
        print("ตัวอย่างน้อยเกินไปที่จะแบ่งไว้ทดสอบ (ต้องมีอย่างน้อย 8 ต่อป้าย)")
    print(f"เขียน {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (
        sys.stdout,
        sys.stderr,
    ):  # คอนโซล/ไปป์ของ Windows อาจเป็น cp1252: ข้อความไทยต้องไม่ทำให้ล้ม
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--count", type=int, default=400, help="ตัวอย่างสอนต่อป้าย (ค่าเริ่มต้น 400)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "POSE_PACK.md")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--samples", type=Path, help="hand_samples.jsonl จาก Hand lab (สอนจากข้อมูลจริง)"
    )
    args = parser.parse_args(argv)
    if args.samples is not None:
        return from_samples(args.samples, args.out or Path("poses_trained.json"))
    out = args.out or ROOT / "assets" / "poses" / "builtin.json"
    return build_pack(args.count, args.seed, out, None if args.no_report else args.report)


if __name__ == "__main__":
    sys.exit(main())

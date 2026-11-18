"""AETHER Studio — ใช้มือสร้าง/ทดลองสิ่งต่างๆ (ไม่ได้คุมคอม): วาดรูปในอากาศ, กระบะทรายฟิสิกส์,
เครื่องดนตรีอากาศ และห้องทดลองมือ

ทุกโหมดรับ :class:`~aether.tracker.HandFrame` ผ่าน :class:`~aether.studio.input.StudioInput` แล้วได้
:class:`~aether.studio.input.StudioHand` (ตำแหน่งบนผืนผ้าใบ 0–1, จีบหรือไม่, ท่ามือ) — logic ทั้งหมดเป็น pure
ทดสอบได้โดยไม่ต้องมีกล้อง ส่วนหน้าต่างอยู่ใน :mod:`aether.studio.window`
"""

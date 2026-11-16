# ที่มาและ license ของ asset

ไฟล์ binary ทั้งหมดไม่ได้เก็บใน git — โหลดด้วย `python tools/download_assets.py`
(ชุดเสียง Kenney ต้องโหลดเองผ่านเบราว์เซอร์ สคริปต์จะบอกขั้นตอน)

| ไฟล์ | ที่มา | เจ้าของ | License |
|---|---|---|---|
| `models/gesture_recognizer.task` | <https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task> | Google (MediaPipe) | Apache License 2.0 |
| `models/hand_landmarker.task` (สำรอง ไม่บังคับ) | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task> | Google (MediaPipe) | Apache License 2.0 |
| `fonts/Orbitron.ttf` (variable font, แกน `wght` 400–900) | <https://github.com/google/fonts/tree/main/ofl/orbitron> | Copyright 2018 The Orbitron Project Authors | SIL Open Font License 1.1 — ข้อความเต็มใน `fonts/OFL-Orbitron.txt` |
| `fonts/ShareTechMono-Regular.ttf` | <https://github.com/google/fonts/tree/main/ofl/sharetechmono> | Copyright (c) 2012 Carrois Type Design, Ralph du Carrois | SIL Open Font License 1.1 — ข้อความเต็มใน `fonts/OFL-ShareTechMono.txt` |
| `_downloads/kenney_interface-sounds.zip` | <https://kenney.nl/assets/interface-sounds> | Kenney (kenney.nl) | Creative Commons Zero (CC0 1.0) |
| `_downloads/kenney_sci-fi-sounds.zip` | <https://kenney.nl/assets/sci-fi-sounds> | Kenney (kenney.nl) | Creative Commons Zero (CC0 1.0) |
| `sounds/*.wav` | แปลงจากชุดเสียง Kenney สองชุดข้างบนด้วย `tools/convert_sounds.py` (ตัดความยาว, fade-out, ปรับความดัง) — สคริปต์พิมพ์ว่าแต่ละไฟล์มาจากไฟล์ต้นทางไหน | Kenney (kenney.nl) | Creative Commons Zero (CC0 1.0) |

หมายเหตุ

- SIL OFL กำหนดให้แนบข้อความ license ไปกับฟอนต์เมื่อแจกจ่าย ไฟล์ `OFL-*.txt` จึงต้องอยู่ใน
  build ที่แจก (PyInstaller) ด้วย
- กราฟิกของ HUD วาดสดด้วย `QPainter` ทั้งหมด ไม่มีไฟล์รูปภาพ
- ไม่มี asset ใดมาจากหรือเลียนแบบภาพยนตร์/แบรนด์ใด

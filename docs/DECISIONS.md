# DECISIONS — จุดที่ทำต่างจาก spec หรือ spec ไม่ได้ระบุ

จดเรียงตามเวลา แต่ละข้อบอกว่า spec ว่าอย่างไร เจออะไร และตัดสินใจอย่างไร

## M0

### D1 · root ของ repo คือโฟลเดอร์ `aether/` ใน spec
spec วาดโครงสร้างไว้ใต้ `aether/` ส่วน repo นี้ใช้ root ของ repo เป็นโฟลเดอร์นั้นเลย
(`pyproject.toml`, `config.toml`, `src/aether/` อยู่ที่ root) ไม่ซ้อนโฟลเดอร์อีกชั้น

### D2 · mediapipe 1.2.x (ไม่ใช่ 0.10.x)
- เวอร์ชันล่าสุดบน PyPI ตอนทำคือ **mediapipe 1.2.3** (wheel `py3-none-win_amd64`)
  จึงกำหนด `mediapipe~=1.2` ใน `pyproject.toml`
- ตรวจจาก package ที่ติดตั้งจริง: ไม่มี `solutions` แล้ว, `mediapipe.tasks.python.vision` มี
  `GestureRecognizer`, `RunningMode.VIDEO`, `recognize_for_video(image, timestamp_ms)` และ
  `HandLandmarksConnections.HAND_CONNECTIONS` ตามที่ spec คาดไว้
- `recognize_for_video` โยน `ValueError` ถ้า timestamp ไม่เพิ่ม → `HandTracker.process`
  บังคับให้เพิ่มอย่างน้อย +1ms เสมอ
- มี test กันไม่ให้ `mp.solutions` / `mediapipe.solutions` โผล่ในโค้ด (`tests/test_tracker.py`)

### D3 · OpenCV มาจาก mediapipe
`pip install mediapipe` ดึง **`opencv-contrib-python`** (5.0.0.93) มาเอง จึงไม่ใส่ OpenCV ใน
dependency ของเรา และ **ห้ามลง `opencv-python` เพิ่ม** (ทั้งคู่ให้ module `cv2` จะชนกัน)

### D4 · Handedness: ต้องสลับ label เมื่อ flip ภาพ (ต่างจาก spec)
- **spec ว่า:** MediaPipe ถือว่าภาพเป็นแบบกระจก พอ flip แนวนอนแล้ว label ซ้าย/ขวาจะตรงกับมือจริง
- **ที่เจอ (mediapipe 1.2.3 + `gesture_recognizer.task`):** กลับกัน — ภาพที่ **ไม่ flip** ให้ label
  ตรงกับมือจริง ส่วนภาพที่ flip แล้ว label จะสลับข้าง ทดสอบกับภาพตัวอย่างของ MediaPipe 3 ภาพ
  (`pointing_up.jpg` มือซ้ายจริง → `Left`, `victory.jpg` และ `thumbs_down.jpg` มือขวาจริง →
  `Right`; พอ flip ทั้งสามภาพให้ label ตรงข้าม)
- **ตัดสินใจ:** ยัง flip ภาพก่อนส่งเข้าโมเดลตาม spec (ให้ทิศการขยับเหมือนส่องกระจก) แล้วสลับ
  label ใน `tracker.user_handedness()` เมื่อ `camera.mirror = true` ถ้าปิด mirror จะใช้ label ตรงๆ
- **ต้องยืนยันกับกล้องจริง** ในขั้นทดสอบ M0 (ยกมือขวา → ต้องขึ้น `RIGHT`)

### D5 · ไฟล์ `.task` ไม่ได้ขึ้นต้นด้วย `PK`
ไฟล์โมเดลเป็น zip ที่มี byte `00 00` นำหน้า signature การตรวจไฟล์ใน `download_assets.py`
จึงใช้ `zipfile.is_zipfile()` แทนการเทียบ byte แรก

### D6 · ชุดเสียง Kenney ต้องโหลดเองผ่านเบราว์เซอร์
- **spec ว่า:** ให้ `download_assets.py` โหลดทุกไฟล์ ถ้าโหลดอัตโนมัติไม่ได้ให้บอกวิธีโหลดเอง
- **ที่เจอ:** หน้า asset ของ kenney.nl ไม่มีลิงก์ `.zip` ตรงแล้ว ปุ่ม Download เปิดหน้าต่างบริจาค
  แล้วใช้ JavaScript ขอ "signed link" แบบใช้ครั้งเดียว และเว็บตอบ 403 กับ client อัตโนมัติบางตัว
- **ตัดสินใจ:** ไม่เขียนสคริปต์เลี่ยงขั้นตอนของเว็บ สคริปต์จะตรวจว่ามี zip ใน `assets/_downloads/`
  หรือยัง (ยอมรับชื่อไฟล์ที่เบราว์เซอร์ตั้งให้ เช่นมี ` (1)` ต่อท้าย) ถ้ายังไม่มีจะบอกลิงก์และที่วาง
- **ยังไม่ได้ยืนยัน:** ตารางเลือกเสียงใน `convert_sounds.py` (`ROLES`) เขียนจากชื่อไฟล์ที่คาดว่าอยู่ใน
  ชุดเสียง (เช่น `click_002.ogg`, `computerNoise_000.ogg`) โดยยังไม่ได้เปิด zip จริง จึงใส่ pattern
  สำรองไว้ทุกเสียง พร้อม `--list` (ดูชื่อไฟล์ที่มี) และ `--map ROLE=FILE` (เลือกเอง)
  ต้องรันกับ zip จริงบนเครื่องผู้ใช้แล้วปรับตารางถ้ามีเสียงที่หาไม่เจอหรือฟังไม่เข้ากัน

### D7 · ส่งโมเดลเข้า MediaPipe เป็น bytes
`HandTracker` อ่านไฟล์โมเดลด้วย Python แล้วส่งเป็น `model_asset_buffer` แทน `model_asset_path`
เป็นการกันไว้ก่อนสำหรับ path ที่มีอักษรไทย (เช่นชื่อ user ของ Windows) ซึ่งฝั่ง C อาจเปิดไม่ได้
(ยังไม่ได้ทดสอบว่า path แบบนั้นพังจริงหรือไม่)

### D8 · UI หยิบเฟรมล่าสุดเอง ไม่รับทุกเฟรมผ่าน signal queue
worker thread (กล้อง → tracker) เก็บ `FramePacket` ล่าสุดไว้ แล้วส่ง signal `frame_ready`
เฉพาะเมื่อ UI หยิบอันก่อนไปแล้ว ถ้า UI วาดไม่ทัน เฟรมเก่าจะถูกทิ้งแทนที่จะสะสมในคิวจนภาพหน่วง
(ฝั่งส่ง input ให้ OS ใน M1 จะทำใน worker thread ตาม spec จึงไม่ได้รับผลจากความเร็วของ UI)

### D9 · key ที่ไม่รู้จักใน `config.toml` เป็น error
spec ให้ validate config แต่ไม่ได้ระบุกรณี key แปลกปลอม เลือกให้ **error พร้อมบอก key ที่ใช้ได้**
แทนการข้ามเงียบๆ เพราะค่าพวก threshold มีผลต่อความปลอดภัยของการคุมเมาส์ พิมพ์ผิดแล้วไม่รู้ตัวจะอันตราย

### D10 · สิ่งที่เลื่อนไป milestone ถัดไป
- `tools/record_landmarks.py` — ทำใน M1 (ใช้อัด fixture สำหรับ test gesture)
- `Orbitron.ttf` ที่โหลดมาเป็น variable font ไฟล์เดียว (`Orbitron[wght].ttf` ใน repo google/fonts)
  การเลือกน้ำหนักฟอนต์ใน Qt จะจัดการตอนทำ HUD (M2)

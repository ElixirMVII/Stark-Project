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

## M1

### D11 · Global hotkey ใช้ `RegisterHotKey` (Win32) แทน pynput
- **spec ว่า:** ใช้ pynput ทำ global hotkey
- **ที่เจอ (อ่านจาก source ของ pynput 1.8.2):** `GlobalHotKeys` บน Windows เทียบ hotkey จาก
  *ตัวอักษร* ที่ปุ่มนั้นพิมพ์ออกมาใน layout ปัจจุบัน (`Listener.canonical` → `char_from_scan`)
  ถ้าเปิดแป้นไทยอยู่ ปุ่ม G จะให้ "เ" → `<ctrl>+<alt>+g` ไม่ติด ซึ่งรับไม่ได้สำหรับปุ่มหยุดฉุกเฉิน
- **ตัดสินใจ:** บน Windows ใช้ `RegisterHotKey` ผ่าน ctypes (spec อนุญาต ctypes/Win32) ใน thread
  ของตัวเองพร้อม message loop — เทียบจาก virtual-key code จึงไม่ขึ้นกับภาษาของแป้น และไม่ต้องใช้
  keyboard hook รูปแบบใน config ยังเป็น `<ctrl>+<alt>+g` ตาม spec (parse เองใน `hotkeys.py`)
  นอก Windows (เครื่อง dev) ถอยไปใช้ pynput
- **ยังไม่ได้ยืนยันบน Windows จริง** — ส่วน loop/การลงทะเบียน test ด้วย API ปลอม แต่ signature ของ
  ctypes ต้องลองบนเครื่องจริง (อยู่ใน checklist M1)
- pynput ยังใช้สำหรับเมาส์ตาม spec (`SetCursorPos` + `SendInput`)

### D12 · โมดูลที่เพิ่มจากโครงสร้างใน spec
- `pipeline.py` — ร้อย tracker → engine → executor ภายใต้ lock เดียว ไม่ขึ้นกับ Qt (test ง่าย)
  สถานะ ARM มีแหล่งความจริงเดียวคือ engine; hotkey/watchdog/worker เรียกผ่าน lock เดียวกัน
- `hotkeys.py` — ดู D11
- `recording.py` — อ่าน/เขียน JSONL ของ landmark และเล่นซ้ำเข้า engine (ใช้ทั้ง tool และ test)

### D13 · รายละเอียดของ gesture engine ที่ spec ไม่ได้ระบุ
- **คลิกเกิดตอนปล่อยนิ้ว** (ต้องรอดูว่าจะกลายเป็นการลากหรือไม่) ส่วน **คลิกขวาเกิดทันที** ที่จีบ
  โป้ง+กลางครบ 40ms (ไม่มีการลากด้วยปุ่มขวา)
- **ดับเบิลคลิก:** จีบครั้งที่สองภายในเวลาดับเบิลคลิกของ OS (`GetDoubleClickTime`) และห่างจากจุดแรก
  ไม่เกิน ~32px จะถูกส่งเป็นคลิกที่ **ตำแหน่งเดิมของคลิกแรก** เพื่อให้ OS นับเป็นดับเบิลคลิกแน่นอน
  แม้มือจะขยับเล็กน้อย
- **การจีบที่เห็นแค่เฟรมเดียว (< 40ms) ไม่นับ** — กัน landmark เพี้ยนชั่วเฟรม
- **ระหว่างลาก ต้องเห็นนิ้วห่างต่อเนื่อง 30ms จึงปล่อย** — กันของหลุดมือเพราะ landmark เพี้ยนเฟรมเดียว
- **ต้องเห็นนิ้วห่างเกิน `pinch.off` หนึ่งครั้งก่อนจีบได้** หลัง arm หรือหลังมือเพิ่งเข้าเฟรม —
  กันคลิก/ลากเองถ้ามือเข้ามาในท่านิ้วชิดกันอยู่แล้ว
- **กำมือ ≠ จีบ:** ถ้าปลายนิ้วพับกลับมาใกล้ข้อมือกว่าโคนนิ้ว (`geometry.finger_curled`, ใช้ world
  landmark 3D) จะไม่นับเป็นการจีบ และการจีบที่กำลังรออยู่จะถูกยกเลิก ค่า `CURL_RATIO = 0.8`
  ตั้งจากสัดส่วนมือโดยประมาณ **ยังไม่ได้ปรับกับข้อมูลจริง**
- **โป้งชิดทั้งนิ้วชี้และนิ้วกลาง:** นิ้วที่ชิดกว่าชนะ
- **Pinch-lock แบบไม่กระโดด:** `mapping.PointerTracker` สะสมระยะที่เคอร์เซอร์ตามหลังเป้าตอน gain
  ต่ำ แล้วค่อยๆ เก็บคืนตามที่มือขยับ (50% ของระยะขยับต่อเฟรม) แทนการกระโดดกลับไปหาเป้า
- **มือหายช่วงสั้น (< 300ms) คงสถานะเดิม** ไม่ปล่อยปุ่ม ไม่คลิกซ้ำ; ครบ 300ms จึงปล่อยและกลับ IDLE
- **เลือกมือ:** ถ้าเจอมือข้างถนัดสองมือในเฟรมเดียว เลือกมือที่อยู่ใกล้ตำแหน่งเดิม
- **Majority vote:** เฟรมที่ยังไม่เห็น (เพิ่งเริ่ม/reset) นับเป็น `None` ท่าใหม่ต้องเห็นเกินครึ่ง window เสมอ

### D14 · สกรอล
- สกรอลตาม **ระยะที่มือขยับ** (เหมือนลากหน้ากระดาษ) โดย `scroll.deadzone` เป็น "ระยะฟรี" ที่มือขยับ
  ได้ก่อนเริ่มสกรอล (กันมือสั่น) ไม่ใช่ deadzone ต่อเฟรม
- มือขึ้น = เลื่อนขึ้น (เหมือนหมุนล้อเมาส์ขึ้น) เพิ่ม key `scroll.invert` (ไม่มีใน spec) ไว้กลับทิศ
- ล็อกแกนเดียวระหว่างสกรอล หยุด 300ms จึงเลือกแกนใหม่
- ส่งเป็น **notch เต็ม** ของล้อเมาส์ (สะสมเศษไว้) ไม่ส่ง delta ย่อย เพราะบางโปรแกรมไม่รองรับ
  high-resolution wheel; `SCROLL_NOTCHES_PER_UNIT = 30` (ขยับมือเต็มความสูงเฟรม = 30 notch)

### D15 · Watchdog ปล่อยปุ่มเมื่อกล้องค้าง
spec ให้ปล่อยปุ่มถ้ามือหายเกิน 300ms แต่ถ้ากล้องค้าง worker thread จะไม่ได้เฟรมมาตัดสินเลย
จึงมี timer ใน UI thread เรียก `Pipeline.check_stall()` ทุก 100ms: ไม่มีเฟรมเกิน 300ms → engine
ทำเหมือนมือหาย (ปล่อยปุ่ม กลับ IDLE)

### D16 · Fixture ใน `tests/fixtures/` เป็นข้อมูลสังเคราะห์ (ตอนนี้)
ยังไม่มีกล้องให้อัดจริง ไฟล์ `synthetic_*.jsonl` จึงสร้างจากมือจำลอง (`tests/handsim.py` ผ่าน
`tests/make_synthetic_fixtures.py`) ใช้ทดสอบเส้นทาง อัด → อ่าน → เล่นซ้ำ ไฟล์ที่อัดจากกล้องจริงด้วย
`tools/record_landmarks.py --expect ...` วางในโฟลเดอร์เดียวกันแล้ว `pytest` จะตรวจให้อัตโนมัติ
**มือจำลองไม่มี noise/มุมมือแบบของจริง** ค่า threshold ทั้งหมดจึงต้องยืนยันกับกล้องจริงอีกครั้ง

### D17 · ย้ายเคอร์เซอร์ด้วย `SetCursorPos` (ผ่าน pynput)
pynput บน Windows ย้ายเคอร์เซอร์ด้วย `SetCursorPos` และกดปุ่ม/ล้อด้วย `SendInput` ซึ่งเป็นแบบเดียวกับ
ที่ pyautogui ใช้ ถ้าทดสอบจริงแล้วการลากใช้ไม่ได้กับบางโปรแกรม ทางแก้คือเปลี่ยน `PynputBackend.move`
ไปใช้ `SendInput` แบบ absolute (แก้จุดเดียวใน `actions.py`)

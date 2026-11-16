# AETHER

ควบคุม Windows ด้วยมือผ่านเว็บแคม พร้อม HUD โฮโลแกรมแบบโปร่งใส
("AETHER" เป็นชื่อโค้ดเนม)

> **สถานะ: M0 — Setup + เห็นมือ**
> ตอนนี้แอปเปิดกล้อง จับมือ และแสดงผลในหน้าต่าง debug ได้ **ยังไม่คุมเมาส์/คีย์บอร์ด** และยังไม่มี HUD
> (ดูแผนทั้งหมดในหัวข้อ [Milestones](#milestones))

## สิ่งที่ต้องมี

- Windows 10/11
- **Python 3.12 แบบ 64-bit** (MediaPipe ยังไม่รองรับ 3.13/3.14)
- เว็บแคม

## ติดตั้ง (PowerShell)

```powershell
git clone <repo> ; cd Stark-Project

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # ถ้าโดนบล็อก: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -e ".[dev]"

python tools\download_assets.py       # โหลดโมเดล + ฟอนต์ ลง assets\
```

ตรวจว่า OpenCV ไม่ชนกัน — ต้องมี `opencv-contrib-python` ตัวเดียว (mediapipe ดึงมาให้เอง):

```powershell
pip show opencv-contrib-python        # ต้องเจอ
pip show opencv-python                # ต้องขึ้น "Package(s) not found"
```

ถ้ามี `opencv-python` ติดมาด้วย: `pip uninstall opencv-python` แล้ว
`pip install --force-reinstall opencv-contrib-python`

### เสียงเอฟเฟกต์ (ทำตอนไหนก็ได้ก่อน M2)

ชุดเสียงของ Kenney (CC0) ต้องโหลดเองผ่านเบราว์เซอร์ เพราะเว็บไม่มีลิงก์ตรงให้สคริปต์โหลด:

1. เปิด <https://kenney.nl/assets/interface-sounds> และ <https://kenney.nl/assets/sci-fi-sounds>
   กด **Download** (ข้ามหน้าบริจาคได้)
2. วางไฟล์ zip ทั้งสองไว้ที่ `assets\_downloads\` โดยไม่ต้องแตกไฟล์
3. รัน:

```powershell
python tools\download_assets.py --check   # ต้องขึ้นว่า asset ครบ
python tools\convert_sounds.py            # เลือกเสียง → แปลงเป็น WAV ลง assets\sounds\
```

ถ้า `convert_sounds.py` บอกว่าหาเสียงไหนไม่เจอ ให้ดูชื่อไฟล์ที่มีด้วย `--list` แล้วเลือกเองด้วย
`--map ROLE=FILE` เช่น `python tools\convert_sounds.py --map click=select_003.ogg`

## ใช้งาน

```powershell
python -m aether --debug --dry-run
```

| ตัวเลือก | ความหมาย |
|---|---|
| `--debug` | เปิดหน้าต่าง debug: ภาพกล้อง + landmark + ชื่อท่า + FPS (ปิดหน้าต่าง = ปิดแอป) |
| `--dry-run` | ไม่ส่ง input ให้ OS จริง แค่ log (M0 ยังไม่ส่ง input อยู่แล้ว) |
| `--no-hud` | ไม่แสดง HUD (HUD จะมาใน M2) |
| `--config PATH` | ใช้ไฟล์ config อื่นแทน `config.toml` |

ถ้าไม่ใส่ `--debug` แอปจะไม่มีหน้าต่าง แต่ log ค่า FPS ลง console ทุก 5 วินาที กด `Ctrl+C` เพื่อปิด

### อ่านหน้าต่าง debug

- เส้นโครงมือสี **ฟ้า** = มือข้างถนัด (`tracking.dominant_hand`), สี **เหลืองอำพัน** = มืออีกข้าง
- ป้ายเหนือมือ: ข้างของมือ + ท่าที่โมเดลเห็น + ความมั่นใจ เช่น `RIGHT · Victory 91%`
- กรอบเส้นประ = active region (พื้นที่ในเฟรมที่จะ map เต็มจอใน M1)
- แถบมุมซ้ายล่าง: `FPS` (เปลี่ยนเป็นสีเหลืองถ้าต่ำกว่า 25), `process` = เวลาประมวลผลต่อเฟรม,
  จำนวนมือ, ความละเอียดกล้อง, สถานะ mirror

## ปรับค่า

แก้ `config.toml` แล้วเปิดแอปใหม่ ใส่เฉพาะ key ที่อยากเปลี่ยนก็ได้ (ที่เหลือใช้ค่าเริ่มต้น)
key ที่พิมพ์ผิดหรือค่าที่อยู่นอกช่วงจะถูกแจ้งตอนเปิดแอป ค่าที่มีผลแล้วใน M0:

| key | ความหมาย |
|---|---|
| `camera.index` | กล้องตัวที่เท่าไร (0 = ตัวแรก) |
| `camera.width` / `height` / `fps` | ความละเอียดและ FPS ที่ขอจากกล้อง |
| `camera.mirror` | flip ภาพแนวนอนเหมือนส่องกระจก |
| `tracking.dominant_hand` | มือข้างถนัด `"left"` / `"right"` |
| `tracking.num_hands` | จำนวนมือสูงสุดที่จับ (1–2) |

## ทดสอบ

```powershell
pytest
```

test ทั้งหมดรันได้โดยไม่ต้องมีกล้อง

## Milestones

| | เนื้อหา | สถานะ |
|---|---|---|
| M0 | Setup + เห็นมือ (debug view) | ✅ รอทดสอบด้วยมือ |
| M1 | Pointer + คลิก + ลาก + สกรอล | ยังไม่เริ่ม |
| M2 | HUD overlay + เสียง | ยังไม่เริ่ม |
| M3 | ท่าพิเศษ (swipe, กำมือ, ซูมสองมือ, 🤟) | ยังไม่เริ่ม |
| M4 | Tray, calibration, build เป็น `.exe` | ยังไม่เริ่ม |

การตัดสินใจที่ต่างจาก spec จดไว้ใน [`docs/DECISIONS.md`](docs/DECISIONS.md)
ที่มาและ license ของ asset อยู่ใน [`assets/LICENSES.md`](assets/LICENSES.md)

## แก้ปัญหา

| อาการ | ลองทำ |
|---|---|
| `ไม่พบโมเดล ...gesture_recognizer.task` | รัน `python tools\download_assets.py` |
| `เปิดกล้อง index 0 ไม่ได้` | ปิดโปรแกรมอื่นที่ใช้กล้อง (Teams, Zoom, แอป Camera) · เปิดสิทธิ์ที่ Settings > Privacy & security > Camera > *Let desktop apps access your camera* · ลองเปลี่ยน `camera.index` เป็น 1 |
| FPS ต่ำกว่า 25 | เพิ่มแสง (กล้องจะลด FPS เองเมื่อแสงน้อย) · ปิดโปรแกรมที่กิน CPU · ตรวจว่า `camera.width`/`height` เป็น 640×480 |
| ซ้าย/ขวาขึ้นสลับข้าง | แจ้งกลับมาพร้อมค่า `camera.mirror` ที่ใช้ (ดู D4 ใน `docs/DECISIONS.md`) |
| จับมือไม่ติด / หลุดบ่อย | ให้มีแสงส่องหน้า นั่งห่างกล้อง 50–80 ซม. พื้นหลังไม่รก |
| ตัวหนังสือไทยใน console เพี้ยน | ใช้ Windows Terminal แทน console รุ่นเก่า |

## ข้อจำกัดที่รู้อยู่แล้ว

- คุมหน้าต่างที่รันแบบ **Administrator** ไม่ได้ (เช่น Task Manager) เพราะ Windows UIPI บล็อก input
  จากโปรแกรมที่สิทธิ์ต่ำกว่า — แอปนี้จะไม่ขอสิทธิ์ admin
- ส่ง `Ctrl+Alt+Del` ไม่ได้
- ต้องมีแสงพอ ควรมีแสงส่องหน้า นั่งห่างกล้อง 50–80 ซม. พื้นหลังที่รกอาจทำให้จับมือพลาด
- รองรับจอหลักจอเดียว (v1)

## Privacy

- ประมวลผลในเครื่องทั้งหมด ตอนรันแอปไม่มีการต่อ network
  (มีแค่ `tools\download_assets.py` ที่ต่อ network ตอนติดตั้ง)
- ไม่บันทึกภาพจากกล้องลงดิสก์

"""Test ของสคริปต์ใน tools/ (ไม่ต่อ network — ใช้ไฟล์ปลอมใน tmp_path)"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from conftest import load_tool

download_assets = load_tool("download_assets")
convert_sounds = load_tool("convert_sounds")

SR = 44_100


def _zip_bytes(members: dict[str, bytes], prefix: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return prefix + buffer.getvalue()


# ---- download_assets ---------------------------------------------------- #


def test_verify_file_accepts_zip_with_leading_bytes(tmp_path):
    """ไฟล์ .task ของ MediaPipe เป็น zip ที่มี byte นำหน้า 2 ตัว"""
    path = tmp_path / "model.task"
    path.write_bytes(_zip_bytes({"a.tflite": b"x" * 2000}, prefix=b"\x00\x00"))
    assert download_assets.verify_file(path, "zip", min_size=1000) is None


def test_verify_file_rejects_html_error_page_saved_as_zip(tmp_path):
    path = tmp_path / "pack.zip"
    path.write_bytes(b"<!DOCTYPE html><html>403 Forbidden</html>" * 100)
    assert "รูปแบบไฟล์" in download_assets.verify_file(path, "zip", min_size=100)


def test_verify_file_missing_and_too_small(tmp_path):
    assert download_assets.verify_file(tmp_path / "nope.ttf", "ttf", 10) == "ไม่พบไฟล์"
    small = tmp_path / "small.ttf"
    small.write_bytes(b"\x00\x01\x00\x00")
    assert "เล็กผิดปกติ" in download_assets.verify_file(small, "ttf", 1000)


def test_verify_file_font_magic(tmp_path):
    good = tmp_path / "good.ttf"
    good.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 100)
    bad = tmp_path / "bad.ttf"
    bad.write_bytes(b"<html>" + b"\x00" * 100)
    assert download_assets.verify_file(good, "ttf", 50) is None
    assert download_assets.verify_file(bad, "ttf", 50) is not None


def test_verify_file_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        download_assets.verify_file(tmp_path / "x", "exe", 1)


def test_locate_accepts_alternative_browser_filename(tmp_path):
    """ผู้ใช้โหลดผ่านเบราว์เซอร์ ชื่อไฟล์อาจไม่ตรงเป๊ะ (เช่นมี " (1)" ต่อท้าย)"""
    asset = download_assets.Asset(
        dest="_downloads/kenney_interface-sounds.zip",
        kind="zip",
        min_size=10,
        alt_name=r"interface[-_ ]?sounds.*\.zip$",
    )
    downloads = tmp_path / "_downloads"
    downloads.mkdir()
    assert download_assets.locate(asset, tmp_path)[1] == "ไม่พบไฟล์"

    alt = downloads / "Kenney_Interface-Sounds (1).zip"
    alt.write_bytes(_zip_bytes({"Audio/click_001.ogg": b"x" * 100}))
    (downloads / "unrelated.zip").write_bytes(_zip_bytes({"a": b"x" * 100}))
    found, problem = download_assets.locate(asset, tmp_path)
    assert problem is None and found == alt


@pytest.mark.parametrize(
    ("dest", "browser_name"),
    [
        ("_downloads/kenney_interface-sounds.zip", "kenney_interface-sounds (1).zip"),
        ("_downloads/kenney_interface-sounds.zip", "Kenney Interface Sounds.zip"),
        ("_downloads/kenney_sci-fi-sounds.zip", "kenney_sci-fi-sounds(2).zip"),
        ("_downloads/kenney_sci-fi-sounds.zip", "kenney_scifi_sounds.zip"),
    ],
)
def test_real_sound_pack_entries_accept_browser_filenames(tmp_path, dest, browser_name):
    asset = next(a for a in download_assets.ASSETS if a.dest == dest)
    downloads = tmp_path / "_downloads"
    downloads.mkdir()
    (downloads / browser_name).write_bytes(_zip_bytes({"Audio/a.ogg": bytes(200_000)}))
    found, problem = download_assets.locate(asset, tmp_path)
    assert problem is None and found.name == browser_name
    other = next(a for a in download_assets.ASSETS if a.url is None and a.dest != dest)
    assert download_assets.locate(other, tmp_path)[1] is not None  # ไม่จับชุดเสียงผิดชุด


def test_manual_asset_is_never_fetched(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("ต้องไม่ต่อ network สำหรับ asset ที่โหลดเอง")

    monkeypatch.setattr(download_assets, "fetch_file", no_network)
    manual = next(a for a in download_assets.ASSETS if a.url is None)
    status, detail = download_assets.process(manual, tmp_path, force=True, check_only=False)
    assert status == "missing" and "เบราว์เซอร์" in detail


def test_check_only_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_assets, "fetch_file", lambda *a, **k: pytest.fail("ต้องไม่โหลดตอน --check")
    )
    assert download_assets.main(["--check", "--assets-dir", str(tmp_path)]) == 1


def test_download_flow_with_fake_network(tmp_path, monkeypatch, capsys):
    """จำลอง network: ลิงก์ตรงโหลดได้ครบ เหลือแค่ชุดเสียงที่ต้องโหลดเอง → exit 1 พร้อมคำแนะนำ"""

    def fake_fetch(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".task":
            dest.write_bytes(_zip_bytes({"m.tflite": bytes(1_100_000)}, prefix=b"\x00\x00"))
        elif dest.suffix == ".ttf":
            dest.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 20_000)
        else:
            dest.write_text("SIL OPEN FONT LICENSE " * 100, encoding="utf-8")

    monkeypatch.setattr(download_assets, "fetch_file", fake_fetch)
    assert download_assets.main(["--assets-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "kenney.nl/assets/interface-sounds" in output
    assert (tmp_path / "models" / "gesture_recognizer.task").is_file()
    assert (tmp_path / "fonts" / "Orbitron.ttf").is_file()

    # วาง zip ชุดเสียงแล้วตรวจอีกครั้ง → ครบ
    downloads = tmp_path / "_downloads"
    downloads.mkdir()
    for name in ("kenney_interface-sounds.zip", "kenney_sci-fi-sounds.zip"):
        (downloads / name).write_bytes(_zip_bytes({"Audio/a.ogg": bytes(200_000)}))
    assert download_assets.main(["--check", "--assets-dir", str(tmp_path)]) == 0


def test_corrupt_download_is_removed(tmp_path, monkeypatch):
    def html_instead_of_model(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>error</html>" * 100_000)

    monkeypatch.setattr(download_assets, "fetch_file", html_instead_of_model)
    model = download_assets.ASSETS[0]
    status, _ = download_assets.process(model, tmp_path, force=False, check_only=False)
    assert status == "missing"
    assert not (tmp_path / model.dest).exists()


# ---- convert_sounds: audio processing ----------------------------------- #


def _tone(ms: float, amplitude: float = 0.5, freq: float = 880.0) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000.0)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10), np.zeros(10)], axis=1)
    mono = convert_sounds.to_mono(stereo)
    assert mono.shape == (10,) and mono.dtype == np.float32
    assert np.allclose(mono, 0.5)
    assert convert_sounds.to_mono(np.ones(4)).shape == (4,)


def test_trim_leading_silence_removes_quiet_start():
    silence = np.zeros(int(SR * 0.050), dtype=np.float32)
    trimmed = convert_sounds.trim_leading_silence(np.concatenate([silence, _tone(100)]), SR)
    removed_ms = (len(silence) + int(SR * 0.1) - len(trimmed)) * 1000 / SR
    assert 45 <= removed_ms <= 50  # เหลือไว้ ~2ms ก่อนเสียงเริ่ม


def test_trim_leading_silence_keeps_all_silent_input():
    silent = np.zeros(100, dtype=np.float32)
    assert len(convert_sounds.trim_leading_silence(silent, SR)) == 100


def test_limit_duration_cuts_and_fades_out():
    out = convert_sounds.limit_duration(_tone(500), SR, max_ms=200)
    assert len(out) == int(SR * 0.2)
    assert abs(out[-1]) < 1e-3  # จบที่เงียบ ไม่มีเสียงคลิก
    assert np.max(np.abs(out[: len(out) // 2])) == pytest.approx(0.5, abs=0.01)


def test_limit_duration_leaves_short_sound_untouched():
    tone = _tone(100)
    out = convert_sounds.limit_duration(tone, SR, max_ms=200)
    assert out is tone


def test_normalize_peak():
    out = convert_sounds.normalize_peak(_tone(50, amplitude=0.1), peak=0.7)
    assert np.max(np.abs(out)) == pytest.approx(0.7, abs=1e-3)
    silent = np.zeros(10, dtype=np.float32)
    assert np.all(convert_sounds.normalize_peak(silent) == 0)


# ---- convert_sounds: เลือกไฟล์ -------------------------------------------- #


def test_choose_prefers_exact_then_pattern_order():
    role = convert_sounds.Role("click", ("click_002.ogg", "click_*.ogg", "select_*.ogg"))
    assert convert_sounds.choose(role, ["click_001.ogg", "click_002.ogg"]) == "click_002.ogg"
    assert convert_sounds.choose(role, ["click_005.ogg", "click_003.ogg"]) == "click_003.ogg"
    assert convert_sounds.choose(role, ["select_001.ogg", "error_001.ogg"]) == "select_001.ogg"
    assert convert_sounds.choose(role, ["error_001.ogg"]) is None


def test_parse_overrides():
    assert convert_sounds.parse_overrides(["click=select_003.ogg", " boot = x.ogg "]) == {
        "click": "select_003.ogg",
        "boot": "x.ogg",
    }
    with pytest.raises(ValueError, match="ROLE=FILE"):
        convert_sounds.parse_overrides(["click"])
    with pytest.raises(ValueError, match="ไม่รู้จัก role"):
        convert_sounds.parse_overrides(["beep=a.ogg"])


def test_every_role_has_candidates_and_boot_is_the_only_long_one():
    names = [role.name for role in convert_sounds.ROLES]
    assert names == ["boot", "click", "release", "arm", "disarm", "swipe", "minimize", "error"]
    for role in convert_sounds.ROLES:
        assert role.candidates
        assert (role.max_ms > convert_sounds.SHORT_MS) == (role.name == "boot")


# ---- convert_sounds: ทั้ง flow ด้วยชุดเสียงปลอม ----------------------------- #


def _ogg(samples: np.ndarray) -> bytes:
    sf = pytest.importorskip("soundfile")
    buffer = io.BytesIO()
    sf.write(buffer, samples, SR, format="OGG", subtype="VORBIS")
    return buffer.getvalue()


@pytest.fixture
def fake_packs(tmp_path):
    """zip สองไฟล์หน้าตาเหมือนชุดเสียงที่โหลดมา (OGG stereo อยู่ในโฟลเดอร์ย่อย)"""
    downloads = tmp_path / "_downloads"
    downloads.mkdir()
    stereo_long = np.stack([_tone(600), _tone(600)], axis=1)
    padded = np.concatenate([np.zeros(int(SR * 0.05), dtype=np.float32), _tone(400, 0.2)])
    interface = {
        f"Audio/{name}": _ogg(padded)
        for name in (
            "click_002.ogg", "drop_002.ogg", "confirmation_001.ogg", "close_002.ogg",
            "scroll_003.ogg", "minimize_004.ogg", "error_007.ogg",
        )
    }  # fmt: skip
    interface["License.txt"] = b"CC0"
    (downloads / "kenney_interface-sounds.zip").write_bytes(_zip_bytes(interface))
    (downloads / "kenney_sci-fi-sounds.zip").write_bytes(
        _zip_bytes({"Audio/computerNoise_000.ogg": _ogg(np.concatenate([stereo_long] * 4))})
    )
    return downloads


def test_convert_end_to_end(fake_packs, tmp_path, capsys):
    sf = pytest.importorskip("soundfile")
    sounds = tmp_path / "sounds"
    code = convert_sounds.main(["--downloads-dir", str(fake_packs), "--sounds-dir", str(sounds)])
    output = capsys.readouterr().out
    assert code == 0, output

    assert sorted(p.name for p in sounds.iterdir()) == sorted(
        f"{name}.wav" for name in convert_sounds.ROLE_NAMES
    )
    for name in convert_sounds.ROLE_NAMES:
        info = sf.info(sounds / f"{name}.wav")
        assert (info.format, info.subtype, info.channels) == ("WAV", "PCM_16", 1)
        limit_ms = convert_sounds.BOOT_MS if name == "boot" else convert_sounds.SHORT_MS
        assert info.duration * 1000 <= limit_ms + 1
        samples, _ = sf.read(sounds / f"{name}.wav", dtype="float32")
        assert np.max(np.abs(samples)) == pytest.approx(convert_sounds.PEAK, abs=0.02)
        assert np.max(np.abs(samples[: SR // 100])) > 0.05  # ดังภายใน 10ms แรก

    assert "error.wav ← error_007.ogg" in output and "(ตัวสำรอง)" in output


def test_convert_reports_missing_role(fake_packs, tmp_path, capsys):
    (fake_packs / "kenney_sci-fi-sounds.zip").unlink()  # ไม่มีเสียง boot
    sounds = tmp_path / "sounds"
    code = convert_sounds.main(["--downloads-dir", str(fake_packs), "--sounds-dir", str(sounds)])
    output = capsys.readouterr().out
    assert code == 1
    assert "boot" in output and "--map" in output
    assert (sounds / "click.wav").is_file()  # เสียงอื่นยังถูกแปลง


def test_convert_with_override(fake_packs, tmp_path, capsys):
    sounds = tmp_path / "sounds"
    code = convert_sounds.main(
        ["--downloads-dir", str(fake_packs), "--sounds-dir", str(sounds),
         "--map", "click=DROP_002.ogg"]
    )  # fmt: skip
    assert code == 0
    assert "click.wav ← drop_002.ogg" in capsys.readouterr().out


def test_list_shows_available_files(fake_packs, capsys):
    assert convert_sounds.main(["--downloads-dir", str(fake_packs), "--list"]) == 0
    output = capsys.readouterr().out
    assert "click_002.ogg" in output and "computernoise_000.ogg" in output


def test_no_packs_gives_instructions(tmp_path, capsys):
    assert convert_sounds.main(["--downloads-dir", str(tmp_path / "none")]) == 1
    assert "download_assets.py" in capsys.readouterr().out


def test_convert_adds_random_variants_when_the_pack_has_them(fake_packs, tmp_path, capsys):
    """M7: เสียงสลับ <ชื่อ>_2.wav, _3.wav จากไฟล์ในชุดเดียวกัน (ไม่มีก็ข้าม ไม่ถือว่าขาด)"""
    pytest.importorskip("soundfile")
    pack = fake_packs / "kenney_interface-sounds.zip"
    with zipfile.ZipFile(pack, "a") as archive:
        tone = np.concatenate([np.zeros(10, dtype=np.float32), _tone(500, 0.15)])
        archive.writestr("Audio/click_001.ogg", _ogg(tone))
        archive.writestr("Audio/click_003.ogg", _ogg(tone))
        archive.writestr("Audio/scroll_005.ogg", _ogg(tone))
    sounds = tmp_path / "sounds"
    code = convert_sounds.main(["--downloads-dir", str(fake_packs), "--sounds-dir", str(sounds)])
    assert code == 0, capsys.readouterr().out
    names = {p.name for p in sounds.iterdir()}
    assert {"click_2.wav", "click_3.wav", "swipe_3.wav"} <= names
    assert "swipe_2.wav" not in names  # scroll_001 ไม่มีในชุด: ข้ามไป

    from aether.sound import variant_paths

    assert [p.name for p in variant_paths(sounds, "click")] == [
        "click.wav",
        "click_2.wav",
        "click_3.wav",
    ]
    assert [p.name for p in variant_paths(sounds, "swipe")] == ["swipe.wav", "swipe_3.wav"]

"""รายละเอียดของธาตุ: การจัดเรียงอิเล็กตรอน, ชั้นพลังงาน (แบบจำลองของโบร์), สถานะ, บล็อก, คำอธิบายภาษาไทย (pure)

การจัดเรียงอิเล็กตรอนใช้ลำดับ Madelung (n + l แล้ว n) และแก้ธาตุที่เป็นข้อยกเว้นจริง (Cr, Cu, Pd, Au …) ตามค่าที่วัดได้
— ธาตุหนักยิ่งยวด (Z ≥ 104) เป็นค่าคาดการณ์ตามลำดับปกติ
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chemistry import CATEGORIES, Element, by_symbol

__all__ = [
    "CATEGORY_NOTES",
    "ELEMENT_NOTES",
    "AtomInfo",
    "atom_info",
    "configuration",
    "configuration_text",
    "shells",
]

_L = "spdf"
_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
_ORDER = sorted(
    ((n, l) for n in range(1, 8) for l in range(min(n, 4))),
    key=lambda s: (s[0] + s[1], s[0]),
)
_NOBLE = (2, 10, 18, 36, 54, 86, 118)
_NOBLE_SYMBOL = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}
# ข้อยกเว้นของลำดับ Madelung (อิเล็กตรอนวงนอกต่อจากแก๊สมีตระกูลตัวก่อนหน้า)
_EXCEPTIONS = {
    24: "3d5 4s1",
    29: "3d10 4s1",
    41: "4d4 5s1",
    42: "4d5 5s1",
    44: "4d7 5s1",
    45: "4d8 5s1",
    46: "4d10",
    47: "4d10 5s1",
    57: "5d1 6s2",
    58: "4f1 5d1 6s2",
    64: "4f7 5d1 6s2",
    78: "4f14 5d9 6s1",
    79: "4f14 5d10 6s1",
    89: "6d1 7s2",
    90: "6d2 7s2",
    91: "5f2 6d1 7s2",
    92: "5f3 6d1 7s2",
    93: "5f4 6d1 7s2",
    96: "5f7 6d1 7s2",
    103: "5f14 7s2 7p1",
}
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

Subshell = tuple[int, int, int]  # (n, l, จำนวนอิเล็กตรอน)


def _parse(text: str) -> list[Subshell]:
    out = []
    for token in text.split():
        match = re.fullmatch(r"(\d)([spdf])(\d+)", token)
        if match is None:
            raise ValueError(token)
        out.append((int(match.group(1)), _L.index(match.group(2)), int(match.group(3))))
    return out


def configuration(number: int) -> list[Subshell]:
    """subshell ที่มีอิเล็กตรอน เรียงตาม (n, l)"""
    if number in _EXCEPTIONS:
        core = max(n for n in _NOBLE if n < number)
        subshells = configuration(core) + _parse(_EXCEPTIONS[number])
    else:
        subshells, left = [], number
        for n, l in _ORDER:
            if left <= 0:
                break
            count = min(_CAPACITY[l], left)
            subshells.append((n, l, count))
            left -= count
    return sorted(subshells, key=lambda s: (s[0], s[1]))


def configuration_text(number: int) -> str:
    """แบบย่อด้วยแก๊สมีตระกูล เช่น 26 → ``"[Ar] 3d⁶ 4s²"``"""
    core = max((n for n in _NOBLE if n < number), default=0)
    inner = {(n, l) for n, l, _ in configuration(core)} if core else set()
    parts = [f"[{_NOBLE_SYMBOL[core]}]"] if core else []
    for n, l, count in configuration(number):
        if (n, l) in inner and (n, l, count) in configuration(core):
            continue
        parts.append(f"{n}{_L[l]}{str(count).translate(_SUPERSCRIPT)}")
    return " ".join(parts)


def shells(number: int) -> list[int]:
    """จำนวนอิเล็กตรอนในแต่ละชั้น (K, L, M, …) — แบบจำลองของโบร์"""
    counts = [0] * 7
    for n, _, count in configuration(number):
        counts[n - 1] += count
    while counts and counts[-1] == 0:
        counts.pop()
    return counts


_GASES = {"H", "N", "O", "F", "Cl", "He", "Ne", "Ar", "Kr", "Xe", "Rn"}
_LIQUIDS = {"Br", "Hg"}

CATEGORY_NOTES = {
    "alkali": "โลหะนิ่ม ตัดด้วยมีดได้ ว่องไวมาก ทำปฏิกิริยากับน้ำรุนแรง มีอิเล็กตรอนวงนอกสุด 1 ตัว",
    "alkaline": "โลหะว่องไวรองจากแอลคาไล มีอิเล็กตรอนวงนอกสุด 2 ตัว พบมากในหินและกระดูก",
    "transition": "โลหะแข็ง นำไฟฟ้าดี จุดหลอมเหลวสูง สารประกอบมักมีสีสัน",
    "post": "โลหะค่อนข้างนิ่ม จุดหลอมเหลวต่ำกว่าโลหะทรานซิชัน",
    "metalloid": "มีสมบัติกึ่งโลหะกึ่งอโลหะ หลายตัวเป็นสารกึ่งตัวนำในชิปอิเล็กทรอนิกส์",
    "nonmetal": "ไม่นำไฟฟ้า (ยกเว้นแกรไฟต์) เป็นส่วนประกอบหลักของสิ่งมีชีวิต",
    "halogen": "อโลหะที่ว่องไวมาก รับอิเล็กตรอนเพิ่ม 1 ตัวแล้วเสถียร รวมกับโลหะกลายเป็นเกลือ",
    "noble": "แก๊สเฉื่อย อิเล็กตรอนวงนอกสุดเต็ม จึงแทบไม่ทำปฏิกิริยากับอะไรเลย",
    "lanthanide": "ธาตุหายาก ใช้ในแม่เหล็กแรงสูง จอภาพ เลเซอร์ และแบตเตอรี่",
    "actinide": "กัมมันตรังสีทุกตัว บางตัวเป็นเชื้อเพลิงนิวเคลียร์ หลายตัวมนุษย์สร้างขึ้น",
}

ELEMENT_NOTES = {
    "H": "ธาตุที่เบาที่สุดและมีมากที่สุดในจักรวาล เป็นเชื้อเพลิงของดวงอาทิตย์ (ฟิวชัน)",
    "He": "เบากว่าอากาศ ใช้เติมลูกโป่งและหล่อเย็นเครื่อง MRI สูดแล้วเสียงแหลม",
    "Li": "โลหะที่เบาที่สุด หัวใจของแบตเตอรี่ลิเทียมไอออนในมือถือและรถไฟฟ้า",
    "Be": "เบาแต่แข็งแรง ใช้ทำกระจกของกล้องโทรทรรศน์อวกาศเจมส์ เว็บบ์ (ฝุ่นเป็นพิษ)",
    "B": "ใช้ทำแก้วทนความร้อน (บอโรซิลิเกต) และบอแรกซ์",
    "C": "พื้นฐานของสิ่งมีชีวิต อยู่ได้หลายรูปแบบ เช่น เพชร แกรไฟต์ (ไส้ดินสอ) และกราฟีน",
    "N": "78% ของอากาศ ไนโตรเจนเหลว (−196 °C) แช่แข็งของได้ในพริบตา",
    "O": "จำเป็นต่อการหายใจ และเป็นธาตุที่มีมากที่สุดในเปลือกโลก",
    "F": "อโลหะที่ว่องไวที่สุด ทำปฏิกิริยากับแทบทุกอย่าง ฟลูออไรด์ช่วยป้องกันฟันผุ",
    "Ne": "เรืองแสงสีส้มแดงในป้ายไฟนีออน",
    "Na": "โลหะนิ่มตัดด้วยมีดได้ โยนลงน้ำแล้ววิ่งพล่าน ลุกไหม้ และระเบิด",
    "Mg": "เผาแล้วให้แสงขาวจ้า ใช้ในพลุและโลหะผสมน้ำหนักเบา",
    "Al": "โลหะที่มีมากที่สุดในเปลือกโลก ใช้ทำกระป๋อง ฟอยล์ และเครื่องบิน",
    "Si": "หัวใจของชิปคอมพิวเตอร์และแผงโซลาร์เซลล์",
    "P": "ฟอสฟอรัสขาวลุกติดไฟเองในอากาศ ฟอสฟอรัสแดงอยู่ที่ข้างกล่องไม้ขีดไฟ",
    "S": "สีเหลืองสด เผาแล้วกลิ่นฉุน ใช้ทำดินปืนและยางรถยนต์",
    "Cl": "ใช้ฆ่าเชื้อในน้ำประปาและสระว่ายน้ำ",
    "Ar": "แก๊สเฉื่อยเกือบ 1% ของอากาศ ใช้ในหลอดไฟและงานเชื่อม",
    "K": "โยนลงน้ำแล้วลุกเป็นไฟสีม่วง จำเป็นต่อการทำงานของเส้นประสาท กล้วยมีมาก",
    "Ca": "ส่วนประกอบหลักของกระดูกและฟัน",
    "Ti": "แข็งแรงเท่าเหล็กแต่เบากว่ามาก ไม่เป็นสนิม ใช้ทำข้อต่อเทียมและเครื่องบิน",
    "Cr": "ทำให้สแตนเลสไม่เป็นสนิม และชุบโครเมียมให้เงาวาว",
    "Fe": "โลหะที่ใช้มากที่สุดในโลก (เหล็กกล้า) และอยู่ในเลือดเพื่อขนส่งออกซิเจน",
    "Co": "ให้สีน้ำเงินโคบอลต์ในเครื่องเคลือบ และใช้ในแบตเตอรี่",
    "Ni": "ใช้ทำเหรียญและสแตนเลส",
    "Cu": "นำไฟฟ้าดีมาก ใช้ทำสายไฟ ทิ้งไว้นานเกิดสนิมสีเขียวแบบเทพีเสรีภาพ",
    "Zn": "เคลือบเหล็กกันสนิม เช่นสังกะสีมุงหลังคา",
    "Ga": "ละลายในมือได้ (จุดหลอมเหลวราว 30 °C)",
    "Ge": "กึ่งโลหะที่ใช้ทำทรานซิสเตอร์ยุคแรก",
    "As": "สารหนู พิษร้ายแรงที่โด่งดังในประวัติศาสตร์",
    "Br": "หนึ่งในสองธาตุที่เป็นของเหลวที่อุณหภูมิห้อง สีน้ำตาลแดง ระเหยเป็นไอพิษ",
    "Kr": "แก๊สเฉื่อยที่ใช้ในหลอดไฟบางชนิด",
    "Sr": "ให้เปลวไฟสีแดงสดในดอกไม้ไฟ",
    "Ag": "นำไฟฟ้าและความร้อนดีที่สุดในบรรดาโลหะ",
    "Sn": "ใช้เคลือบกระป๋องและทำตะกั่วบัดกรี",
    "I": "จำเป็นต่อต่อมไทรอยด์ จึงเติมในเกลือไอโอดีน",
    "Xe": "ใช้ในไฟหน้ารถซีนอนและเครื่องยนต์ไอออนของยานอวกาศ",
    "Cs": "นาฬิกาอะตอมใช้ซีเซียมกำหนดความยาวหนึ่งวินาที โยนลงน้ำระเบิดรุนแรงมาก",
    "Ba": "ใช้ในพลุให้ไฟสีเขียว และกลืนก่อนเอกซเรย์ทางเดินอาหาร (แบเรียมซัลเฟต)",
    "Nd": "ทำแม่เหล็กที่แรงที่สุด ใช้ในหูฟังและมอเตอร์รถไฟฟ้า",
    "W": "จุดหลอมเหลวสูงที่สุดในบรรดาโลหะ (3,422 °C) ใช้ทำไส้หลอดไฟ",
    "Pt": "โลหะมีค่า ใช้ในเครื่องกรองไอเสียรถยนต์",
    "Au": "ไม่เป็นสนิม ไม่หมอง ใช้ทำเครื่องประดับและหน้าสัมผัสในวงจรอิเล็กทรอนิกส์",
    "Hg": "โลหะชนิดเดียวที่เป็นของเหลวที่อุณหภูมิห้อง ไอระเหยเป็นพิษ",
    "Pb": "หนักและนิ่ม กันรังสีได้ดี แต่เป็นพิษต่อระบบประสาท",
    "Bi": "เกิดผลึกสีรุ้งเป็นขั้นบันไดสวยงาม",
    "Po": "กัมมันตรังสีรุนแรงมาก มารี กูรีค้นพบและตั้งชื่อตามโปแลนด์",
    "Rn": "แก๊สกัมมันตรังสีที่อาจสะสมในห้องใต้ดิน",
    "Ra": "เรืองแสงในที่มืด เคยใช้ทาหน้าปัดนาฬิกาก่อนรู้ว่าอันตราย",
    "U": "เชื้อเพลิงของโรงไฟฟ้านิวเคลียร์",
    "Pu": "Pu-238 ใช้เป็นแบตเตอรี่นิวเคลียร์ของยานอวกาศที่ไปไกลจากดวงอาทิตย์",
    "Am": "ใช้ในเครื่องตรวจจับควันในบ้าน",
    "Es": "ค้นพบในเศษซากการทดสอบระเบิดไฮโดรเจนครั้งแรก ปี 1952",
    "Fm": "ค้นพบในเศษซากการทดสอบระเบิดไฮโดรเจนครั้งแรก ปี 1952",
    "Og": "ธาตุที่หนักที่สุดที่รู้จัก สร้างได้ทีละไม่กี่อะตอมและสลายในเสี้ยววินาที",
}


@dataclass(frozen=True, slots=True)
class AtomInfo:
    element: Element
    protons: int
    neutrons: int
    electrons: int
    shells: tuple[int, ...]
    configuration: str
    state: str  # "solid" | "liquid" | "gas" | "unknown"
    block: str  # "s" | "p" | "d" | "f"
    valence: int | None  # อิเล็กตรอนวงนอก (ธาตุหมู่หลัก) — None = แปรผัน (โลหะทรานซิชัน/f)
    note: str  # คำอธิบายภาษาไทย

    @property
    def state_thai(self) -> str:
        return {"solid": "ของแข็ง", "liquid": "ของเหลว", "gas": "แก๊ส"}.get(self.state, "ไม่ทราบ")


def _block(element: Element) -> str:
    if element.group is None:
        return "f"
    if element.group <= 2 or element.symbol == "He":
        return "s"
    return "d" if element.group <= 12 else "p"


def atom_info(element: Element | str) -> AtomInfo:
    if isinstance(element, str):
        element = by_symbol(element)
    z = element.number
    if element.number >= 104:
        state = "unknown"
    elif element.symbol in _GASES:
        state = "gas"
    elif element.symbol in _LIQUIDS:
        state = "liquid"
    else:
        state = "solid"
    group = element.group
    if element.symbol == "He":
        valence: int | None = 2
    elif group is not None and group <= 2:
        valence = group
    elif group is not None and group >= 13:
        valence = group - 10
    else:
        valence = None
    note = ELEMENT_NOTES.get(element.symbol) or CATEGORY_NOTES[element.category]
    return AtomInfo(
        element,
        z,
        max(round(element.mass) - z, 0),
        z,
        tuple(shells(z)),
        configuration_text(z),
        state,
        _block(element),
        valence,
        note,
    )


assert set(CATEGORY_NOTES) == set(CATEGORIES)

"""โมเลกุล 3 มิติของสาร (pure — ไม่มี Qt) สำหรับเครื่องมือทดลอง "Molecule"

- สารในตัวแอปมีโครงสร้างที่เขียนไว้เป็น SMILES (ภาษาบรรยายโมเลกุลมาตรฐาน) — test ตรวจว่านับอะตอมแล้วตรงสูตรทุกตัว
- สารอื่น (สูตรจาก Claude, ของในขวดที่ยังไม่ตรงสูตรใด) ใช้ตัวต่อโครงแบบประมาณจากวาเลนซ์ → บอกว่า "โครงสร้างโดยประมาณ"
- แบ่งชนิด: โมเลกุลโคเวเลนต์ / ไอออนิก (โลหะ + อโลหะที่อิเล็กโตรเนกาติวิตีต่างกันมาก) / โลหะ (ทะเลอิเล็กตรอน) /
  โครงผลึกร่างตาข่าย (เพชร, SiO₂, SiC …) / อะตอมเดี่ยว (แก๊สมีตระกูล)
- จัดวาง 3 มิติแบบ VSEPR อย่างง่าย: พันธะเป็นสปริง, คู่อิเล็กตรอน (พันธะ + คู่โดดเดี่ยว) รอบอะตอมกลางผลักกัน
  → H₂O งอ, NH₃ พีระมิด, CH₄ ทรงสี่หน้า, CO₂ เส้นตรง โดยไม่ต้องใส่พิกัดเอง
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .chemistry import by_symbol, formula_text, hill_formula, parse_formula

__all__ = [
    "MolAtom",
    "MolBond",
    "Structure",
    "build_formula",
    "electronegativity",
    "is_metal",
    "parse_smiles",
    "structure_for",
    "vsepr_shape",
]

# --------------------------------------------------------------------------- #
# สมบัติของธาตุที่ต้องใช้
# --------------------------------------------------------------------------- #

# อิเล็กโตรเนกาติวิตีแบบพอลิง (ธาตุที่พบบ่อย)
# fmt: off
_EN = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
    "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
    "Zr": 1.33, "Mo": 2.16, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.10, "I": 2.66, "Xe": 2.60, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "W": 2.36,
    "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "U": 1.38,
}
# รัศมีโคเวเลนต์ (Å)
_RADIUS = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Ag": 1.45, "Sn": 1.39, "I": 1.39,
    "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "W": 1.62, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Pb": 1.46, "U": 1.96,
}
# วาเลนซ์ที่ใช้ได้ (จำนวนพันธะ) เรียงจากที่พบบ่อยสุด
_VALENCES = {
    "H": (1,), "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "F": (1,), "Si": (4,),
    "P": (3, 5), "S": (2, 4, 6), "Cl": (1, 3, 5, 7), "Br": (1, 3, 5, 7), "I": (1, 3, 5, 7),
    "Se": (2, 4, 6), "As": (3, 5), "Ge": (4,), "Te": (2, 4, 6), "Xe": (0, 2, 4, 6),
    "He": (0,), "Ne": (0,), "Ar": (0,), "Kr": (0, 2), "Rn": (0,),
}
# fmt: on
_METAL_CATEGORIES = {"alkali", "alkaline", "transition", "post", "lanthanide", "actinide"}
_NETWORK_ELEMENTS = {"C", "Si", "Ge", "B"}


def is_metal(symbol: str) -> bool:
    return by_symbol(symbol).category in _METAL_CATEGORIES


def electronegativity(symbol: str) -> float:
    return _EN.get(symbol, 1.3 if is_metal(symbol) else 2.5)


def radius(symbol: str) -> float:
    return _RADIUS.get(symbol, 1.4)


def valence_electrons(symbol: str) -> int | None:
    """อิเล็กตรอนวงนอกของธาตุหมู่หลัก (โลหะทรานซิชัน/f = ``None`` ไม่คิดคู่โดดเดี่ยว)"""
    element = by_symbol(symbol)
    if symbol == "He":
        return 2
    group = element.group
    if group is None or 3 <= group <= 12:
        return None
    return group if group <= 2 else group - 10


def _ion_charge(symbol: str) -> int:
    """ประจุไอออนที่พบบ่อย (โลหะ +, อโลหะ −)"""
    element = by_symbol(symbol)
    if element.category == "alkali":
        return 1
    if element.category == "alkaline":
        return 2
    if symbol in ("Al", "Ga", "Fe", "Cr", "Co", "Sc", "La", "Bi"):
        return 3
    if is_metal(symbol):
        return 2
    electrons = valence_electrons(symbol)
    return -(8 - electrons) if electrons is not None and electrons >= 5 else -1


# --------------------------------------------------------------------------- #
# โครงสร้าง
# --------------------------------------------------------------------------- #


@dataclass
class MolAtom:
    symbol: str
    charge: int = 0
    lone_pairs: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    fragment: int = 0

    @property
    def radius(self) -> float:
        return radius(self.symbol)


@dataclass(frozen=True)
class MolBond:
    a: int
    b: int
    order: int  # 1, 2, 3


@dataclass
class Structure:
    kind: str  # "molecule" | "ionic" | "metallic" | "network" | "atomic" | "reaction"
    atoms: list[MolAtom] = field(default_factory=list)
    bonds: list[MolBond] = field(default_factory=list)
    title: str = ""
    shape: str | None = None
    angle: str | None = None
    note: str = ""
    approximate: bool = False
    labels: list[tuple[float, float, float, str]] = field(default_factory=list)  # ป้ายในอวกาศ 3 มิติ

    def counts(self) -> Counter[str]:
        return Counter(a.symbol for a in self.atoms)

    def neighbors(self, index: int) -> list[int]:
        return [b.b if b.a == index else b.a for b in self.bonds if index in (b.a, b.b)]

    def angle_deg(self, a: int, center: int, b: int) -> float:
        pa, pc, pb = self.atoms[a], self.atoms[center], self.atoms[b]
        u = (pa.x - pc.x, pa.y - pc.y, pa.z - pc.z)
        v = (pb.x - pc.x, pb.y - pc.y, pb.z - pc.z)
        dot = sum(i * j for i, j in zip(u, v, strict=True))
        norm = math.dist((0, 0, 0), u) * math.dist((0, 0, 0), v) or 1.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm))))

    def radius(self) -> float:
        """รัศมีรวมรอบจุดศูนย์กลาง (ใช้ซูมให้พอดีจอ)"""
        if not self.atoms:
            return 1.0
        return max(math.dist((0, 0, 0), (a.x, a.y, a.z)) + a.radius for a in self.atoms)


# --------------------------------------------------------------------------- #
# SMILES (ชุดย่อย): อะตอม B C N O P S F Cl Br I, [วงเล็บเหลี่ยม+ประจุ], = # พันธะ, ( ) กิ่ง, เลขวง, . แยกชิ้น
# --------------------------------------------------------------------------- #

_SMILES_TOKEN = re.compile(r"\[([^\]]+)\]|Br|Cl|[BCNOPSFI]|[=#\-]|\(|\)|\.|\d")
_BRACKET = re.compile(r"^([A-Z][a-z]?)(H(\d*))?([+-]\d*|\++|-+)?$")


def parse_smiles(smiles: str) -> tuple[list[MolAtom], list[MolBond]]:
    """แปลง SMILES เป็นอะตอม + พันธะ (เติม H ที่ละไว้ตามวาเลนซ์ปกติ)"""
    atoms: list[MolAtom] = []
    explicit_h: list[int | None] = []  # None = อะตอมนอกวงเล็บ (เติม H อัตโนมัติ)
    bonds: dict[tuple[int, int], int] = {}
    stack: list[int | None] = []
    previous: int | None = None
    order = 1
    rings: dict[str, tuple[int, int]] = {}
    position = 0
    while position < len(smiles):
        match = _SMILES_TOKEN.match(smiles, position)
        if match is None:
            raise ValueError(f"อ่าน SMILES {smiles!r} ไม่ได้ที่ตำแหน่ง {position}")
        token = match.group(0)
        position = match.end()
        if token in "=#-":
            order = {"-": 1, "=": 2, "#": 3}[token]
            continue
        if token == "(":
            stack.append(previous)
            continue
        if token == ")":
            previous = stack.pop()
            continue
        if token == ".":
            previous = None
            continue
        if token.isdigit():
            if token in rings:
                other, ring_order = rings.pop(token)
                assert previous is not None
                bonds[(other, previous)] = max(order, ring_order)
            else:
                assert previous is not None
                rings[token] = (previous, order)
            order = 1
            continue
        if match.group(1) is not None:
            bracket = _BRACKET.match(match.group(1))
            if bracket is None:
                raise ValueError(f"อ่านอะตอม [{match.group(1)}] ไม่ได้")
            symbol, has_h, h_count, charge_text = bracket.groups()
            hydrogens = (int(h_count) if h_count else 1) if has_h else 0
            charge = _charge(charge_text)
            atoms.append(MolAtom(symbol, charge))
            explicit_h.append(hydrogens)
        else:
            atoms.append(MolAtom(token))
            explicit_h.append(None)
        index = len(atoms) - 1
        if previous is not None:
            bonds[(previous, index)] = order
        previous, order = index, 1
    if rings:
        raise ValueError(f"วงไม่ปิดใน {smiles!r}")
    bond_list = [MolBond(a, b, o) for (a, b), o in bonds.items()]
    for index in range(len(atoms)):  # เติม H
        used = sum(b.order for b in bond_list if index in (b.a, b.b))
        count = explicit_h[index]
        if count is None:
            options = _VALENCES.get(atoms[index].symbol, (0,))
            fit = next((v for v in options if v >= used), used)
            count = fit - used
        for _ in range(count):
            atoms.append(MolAtom("H"))
            bond_list.append(MolBond(index, len(atoms) - 1, 1))
    return atoms, bond_list


def _charge(text: str | None) -> int:
    if not text:
        return 0
    sign = 1 if text[0] == "+" else -1
    if len(text) > 1 and text[1:].isdigit():
        return sign * int(text[1:])
    return sign * len(text)


# --------------------------------------------------------------------------- #
# โครงสร้างที่เขียนไว้ (คีย์ = สูตรตามที่คนคุ้น — แปลงเป็น Hill ตอนค้น)
# --------------------------------------------------------------------------- #

# fmt: off
_SMILES = {
    "H2": "[H][H]", "O2": "O=O", "N2": "N#N", "Cl2": "ClCl", "F2": "FF", "Br2": "BrBr", "I2": "II",
    "O3": "[O-][O+]=O", "H2O": "O", "H2O2": "OO", "CO2": "O=C=O", "H2CO3": "OC(=O)O",
    "NaCl": "[Na+].[Cl-]", "KCl": "[K+].[Cl-]", "NaF": "[Na+].[F-]", "NaHCO3": "[Na+].OC(=O)[O-]",
    "CaCO3": "[Ca+2].[O-]C(=O)[O-]", "Ca(OH)2": "[Ca+2].[OH-].[OH-]",
    "CaCl2": "[Cl-].[Ca+2].[Cl-]", "Fe2O3": "[O-2].[Fe+3].[O-2].[Fe+3].[O-2]",
    "CuO": "[Cu+2].[O-2]", "ZnO": "[Zn+2].[O-2]", "TiO2": "[O-2].[Ti+4].[O-2]",
    "NH4Cl": "[NH4+].[Cl-]", "C2H5OH": "CCO", "CH3COOH": "CC(=O)O", "C3H6O": "CC(=O)C",
    "CH4N2O": "NC(=O)N", "C6H12O6": "OCC(O)C(O)C(O)C(O)C=O",
    "C6H8O7": "OC(=O)CC(O)(CC(=O)O)C(=O)O", "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "C2H4": "C=C", "C8H10N4O2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C8H8O3": "COC1=C(C=CC(=C1)C=O)O", "NaC5H8NO4": "[Na+].NC(CCC(=O)[O-])C(=O)O",
    "SF6": "FS(F)(F)(F)(F)F", "N2O": "[N-]=[N+]=O", "MgO": "[Mg+2].[O-2]",
    "Al2O3": "[O-2].[Al+3].[O-2].[Al+3].[O-2]",
    "Fe3O4": "[O-2].[Fe+2].[O-2].[Fe+3].[O-2].[Fe+3].[O-2]", "FeS2": "[Fe+2].[S-][S-]",
    "LiCoO2": "[Li+].[O-2].[Co+3].[O-2]", "KO2": "[K+].[O-][O]", "AgCl": "[Ag+].[Cl-]",
    "CuSO4": "[Cu+2].[O-]S(=O)(=O)[O-]", "XeF4": "F[Xe](F)(F)F", "CH4": "C", "C3H8": "CCC",
    "C2H2": "C#C", "KNO3": "[K+].[O-][N+](=O)[O-]", "KClO3": "[K+].[O-]Cl(=O)=O",
    "NH4NO3": "[NH4+].[O-][N+](=O)[O-]", "NI3": "IN(I)I",
    "C7H5N3O6": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
    "C3H5N3O9": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "N2H4": "NN",
    "NaN3": "[Na+].[N-]=[N+]=[N-]", "CO": "[C-]#[O+]", "HCN": "C#N", "COCl2": "O=C(Cl)Cl",
    "H2S": "S", "PH3": "P", "NH3": "N", "NO2": "O=N[O]", "SO2": "O=S=O", "CH3OH": "CO",
    "CH2O": "C=O", "C6H6": "C1=CC=CC=C1", "CHCl3": "ClC(Cl)Cl", "CCl4": "ClC(Cl)(Cl)Cl",
    "HgS": "[Hg+2].[S-2]", "UF6": "F[U](F)(F)(F)(F)F", "HCl": "Cl", "HF": "F",
    "HNO3": "O[N+](=O)[O-]", "H2SO4": "OS(=O)(=O)O", "NaOH": "[Na+].[OH-]",
    "NaClO": "[Na+].[O-]Cl", "CaO": "[Ca+2].[O-2]",
    # สารที่อยู่ในสมการของปฏิกิริยา
    "LiOH": "[Li+].[OH-]", "KOH": "[K+].[OH-]", "CsOH": "[Cs+].[OH-]",
    "CH3COONa": "[Na+].CC(=O)[O-]", "ZnCl2": "[Cl-].[Zn+2].[Cl-]",
    "Cu(NO3)2": "[O-][N+](=O)[O-].[Cu+2].[O-][N+](=O)[O-]", "CaC2": "[Ca+2].[C-]#[C-]",
    # รูปปกติของธาตุ
    "P4": "P12P3P1P23", "S8": "S1SSSSSSS1",
}
# fmt: on
STRUCTURES = {hill_formula(parse_formula(formula)): smiles for formula, smiles in _SMILES.items()}
_ELEMENT_FORMS = {
    "H": "H2", "N": "N2", "O": "O2", "F": "F2", "Cl": "Cl2", "Br": "Br2", "I": "I2",
    "P": "P4", "S": "S8",
}  # fmt: skip
# ผลึกร่างตาข่าย (ไม่ใช่โมเลกุลเดี่ยว)
_NETWORKS = {
    "SiO2": ("Si", "O"), "SiC": ("Si", "C"), "GaAs": ("Ga", "As"), "WC": ("W", "C"),
}  # fmt: skip
_NETWORK_KEYS = {hill_formula(parse_formula(f)): pair for f, pair in _NETWORKS.items()}


# --------------------------------------------------------------------------- #
# ตัวต่อโครงแบบประมาณ (สูตรที่ไม่ได้เขียนไว้)
# --------------------------------------------------------------------------- #


def _capacity(symbol: str) -> int:
    if symbol in _VALENCES:
        return _VALENCES[symbol][0]
    return 2


def _expand(symbol: str, current: int) -> int | None:
    options = [v for v in _VALENCES.get(symbol, ()) if v > current]
    return options[0] if options else None


def _covalent(symbols: list[str]) -> tuple[list[MolAtom], list[MolBond], bool]:
    """ต่ออะตอมเป็นโมเลกุลเดียวตามวาเลนซ์ คืน (อะตอม, พันธะ, ครบวาเลนซ์ไหม)"""
    order = sorted(
        range(len(symbols)), key=lambda i: (-_capacity(symbols[i]), electronegativity(symbols[i]))
    )
    atoms = [MolAtom(symbols[i]) for i in order]
    cap = [_capacity(a.symbol) for a in atoms]
    used = [0] * len(atoms)
    bonds: dict[tuple[int, int], int] = {}

    def bond(i: int, j: int) -> None:
        key = (min(i, j), max(i, j))
        bonds[key] = bonds.get(key, 0) + 1
        used[i] += 1
        used[j] += 1

    def room(i: int) -> int:
        return cap[i] - used[i]

    heavy = [i for i, c in enumerate(cap) if c >= 2]
    terminal = [i for i, c in enumerate(cap) if c < 2]
    for position, i in enumerate(heavy[1:], start=1):
        placed = heavy[:position]
        j = max(placed, key=lambda k: (room(k), -k))
        if room(j) <= 0:
            grown = _expand(atoms[j].symbol, cap[j])
            if grown is not None:
                cap[j] = grown
        bond(i, j)
    for t in terminal:
        if heavy:
            j = max(heavy, key=lambda k: (room(k), -k))
            if room(j) <= 0:
                expandable = [k for k in heavy if _expand(atoms[k].symbol, cap[k]) is not None]
                if expandable:
                    j = expandable[0]
                    cap[j] = _expand(atoms[j].symbol, cap[j]) or cap[j]
            bond(t, j)
        else:  # ไม่มีอะตอมกลาง: จับคู่กันเอง (H₂, HCl)
            partner = next((k for k in terminal if k != t and used[k] == 0 and used[t] == 0), None)
            if partner is not None:
                bond(t, partner)
    changed = True
    while changed:  # ยังมีวาเลนซ์เหลือทั้งสองฝั่ง: เพิ่มเป็นพันธะคู่/สาม
        changed = False
        for (i, j), n in sorted(bonds.items(), key=lambda kv: -(room(kv[0][0]) + room(kv[0][1]))):
            if n < 3 and room(i) > 0 and room(j) > 0:
                bonds[(i, j)] = n + 1
                used[i] += 1
                used[j] += 1
                changed = True
    complete = all(room(i) == 0 for i in range(len(atoms)))
    return atoms, [MolBond(i, j, n) for (i, j), n in bonds.items()], complete


def build_formula(counts: Counter[str]) -> Structure:
    """โครงสร้างจากจำนวนอะตอมอย่างเดียว (ไม่มีในรายการที่เขียนไว้)"""
    symbols = [s for s, n in sorted(counts.items()) for _ in range(n)]
    metals = [s for s in symbols if is_metal(s)]
    others = [s for s in symbols if not is_metal(s)]
    if not others:
        return _metallic(counts)
    if (
        metals
        and max(electronegativity(s) for s in others) - min(electronegativity(s) for s in metals)
        >= 1.5
    ):
        return _ionic(metals, others)
    atoms, bonds, complete = _covalent(symbols)
    structure = Structure("molecule", atoms, bonds, approximate=True)
    structure.note = "โครงสร้างโดยประมาณจากวาเลนซ์ (อาจมีไอโซเมอร์อื่น)" + (
        "" if complete else " · มีอิเล็กตรอนเดี่ยว/วาเลนซ์ไม่ครบ"
    )
    return structure


def _ionic(metals: list[str], others: list[str]) -> Structure:
    atoms: list[MolAtom] = []
    bonds: list[MolBond] = []
    positive = 0
    for symbol in metals:
        charge = _ion_charge(symbol)
        positive += charge
        atoms.append(MolAtom(symbol, charge))
    if len(set(others)) == 1:  # ไอออนลบอะตอมเดียว เช่น Cl⁻, O²⁻
        for symbol in others:
            atoms.append(MolAtom(symbol, _ion_charge(symbol)))
    else:  # ไอออนลบหลายอะตอม (CO₃²⁻, OH⁻ …) — ต่อเป็นกลุ่มเดียว ประจุอยู่ที่อะตอมที่ EN สูงสุด
        group, group_bonds, _complete = _covalent(others)
        offset = len(atoms)
        target = max(range(len(group)), key=lambda i: electronegativity(group[i].symbol))
        group[target].charge = -positive
        atoms.extend(group)
        bonds.extend(MolBond(b.a + offset, b.b + offset, b.order) for b in group_bonds)
    structure = Structure("ionic", atoms, bonds, approximate=True)
    structure.note = "สารประกอบไอออนิก (ประจุโดยประมาณ) — ในผลึกจริงไอออนเรียงซ้ำกันเป็นโครงผลึก"
    return structure


# --------------------------------------------------------------------------- #
# โครงผลึก: โลหะ / ร่างตาข่าย / เกลือแบบหินเกลือ
# --------------------------------------------------------------------------- #

_DIAMOND_A = 3.57


def _diamond_sites() -> list[tuple[tuple[float, float, float], bool]]:
    """ตำแหน่งในหนึ่งเซลล์ของโครงสร้างเพชร: (พิกัด 0–1, เป็นจุดของโครงตาข่าย FCC ไหม)"""
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    sites: set[tuple[float, float, float]] = set()
    for x, y, z in fcc:
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    p = (x + dx, y + dy, z + dz)
                    if all(0 <= c <= 1 for c in p):
                        sites.add(p)
    out = [(p, True) for p in sorted(sites)]
    out += [((x + 0.25, y + 0.25, z + 0.25), False) for x, y, z in fcc]
    return out


def _network(first: str, second: str | None) -> Structure:
    sites = _diamond_sites()
    atoms: list[MolAtom] = []
    for (x, y, z), lattice in sites:
        symbol = first if second is None or second == "O" or lattice else second
        atoms.append(
            MolAtom(
                symbol, x=(x - 0.5) * _DIAMOND_A, y=(y - 0.5) * _DIAMOND_A, z=(z - 0.5) * _DIAMOND_A
            )
        )
    bonds = []
    limit = _DIAMOND_A * math.sqrt(3) / 4 * 1.05
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            if math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) <= limit:
                bonds.append(MolBond(i, j, 1))
    if second == "O":  # SiO₂: ออกซิเจนเชื่อมระหว่างซิลิคอนทุกคู่ (คริสโตบาไลต์แบบอุดมคติ)
        bridged: list[MolBond] = []
        for bond in bonds:
            a, b = atoms[bond.a], atoms[bond.b]
            atoms.append(MolAtom("O", x=(a.x + b.x) / 2, y=(a.y + b.y) / 2, z=(a.z + b.z) / 2))
            middle = len(atoms) - 1
            bridged += [MolBond(bond.a, middle, 1), MolBond(middle, bond.b, 1)]
        bonds = bridged
    return Structure(
        "network",
        atoms,
        bonds,
        note="ผลึกร่างตาข่าย: อะตอมต่อกันด้วยพันธะโคเวเลนต์ไปทุกทิศ ไม่มีโมเลกุลแยกเป็นหน่วย (แสดงหนึ่งเซลล์)",
    )


def _grid(symbols: list[str], spacing: float, size: int = 3) -> list[MolAtom]:
    atoms = []
    middle = (size - 1) / 2
    index = 0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                atoms.append(
                    MolAtom(
                        symbols[index % len(symbols)],
                        x=(i - middle) * spacing,
                        y=(j - middle) * spacing,
                        z=(k - middle) * spacing,
                    )
                )
                index += 1
    return atoms


def _metallic(counts: Counter[str]) -> Structure:
    symbols = [s for s, n in sorted(counts.items(), key=lambda kv: -kv[1]) for _ in range(n)] or [
        "Fe"
    ]
    atoms = _grid(symbols, 2.6)
    return Structure(
        "metallic",
        atoms,
        [],
        note='พันธะโลหะ: ไอออนบวกเรียงเป็นผลึก อิเล็กตรอนวงนอกเคลื่อนที่อิสระเป็น "ทะเลอิเล็กตรอน" — นำไฟฟ้าได้',
    )


def _rock_salt(cation: MolAtom, anion: MolAtom) -> Structure:
    atoms = []
    for atom in _grid(["+", "-"], 2.8):
        source = cation if atom.symbol == "+" else anion
        atoms.append(MolAtom(source.symbol, source.charge, x=atom.x, y=atom.y, z=atom.z))
    return Structure(
        "ionic",
        atoms,
        [],
        note="ผลึกไอออนิกแบบหินเกลือ: ไอออนบวก-ลบสลับกันทั้งสามแกน แรงดึงดูดไฟฟ้ายึดไว้ (แสดง 3×3×3)",
    )


# --------------------------------------------------------------------------- #
# จัดวาง 3 มิติ (VSEPR อย่างง่าย)
# --------------------------------------------------------------------------- #

BOND_SHORTEN = {1: 1.0, 2: 0.87, 3: 0.78}
DOMAIN_K = 4.0  # ค่าจากการลองให้ได้มุมตามตำรา: CO₂ 180°, BF₃ 120°, CH₄ 109.5°, NH₃ ≈ 107°, H₂O ≈ 104°
LONE_WEIGHT = 1.3  # คู่โดดเดี่ยวผลักแรงกว่าคู่พันธะ (VSEPR)
RELAX_DT = 0.08
RELAX_DAMPING = 0.85


def _fragments(atoms: list[MolAtom], bonds: list[MolBond]) -> list[list[int]]:
    parent = list(range(len(atoms)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for bond in bonds:
        parent[find(bond.a)] = find(bond.b)
    groups: dict[int, list[int]] = {}
    for i in range(len(atoms)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _lone_pairs(structure: Structure) -> None:
    for index, atom in enumerate(structure.atoms):
        electrons = valence_electrons(atom.symbol)
        if electrons is None or atom.symbol == "H":
            atom.lone_pairs = 0
            continue
        used = sum(b.order for b in structure.bonds if index in (b.a, b.b))
        atom.lone_pairs = max(0, (electrons - atom.charge - used) // 2)


def _relax(structure: Structure, members: list[int], rng: random.Random, steps: int) -> None:
    atoms = structure.atoms
    local = {index: k for k, index in enumerate(members)}
    n = len(members)
    bonds = [(local[b.a], local[b.b], b) for b in structure.bonds if b.a in local]
    pos = _spectral_start(n, [(i, j) for i, j, _b in bonds], rng)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, j, _bond in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    # คู่โดดเดี่ยวเป็นจุดล่องหนผูกกับอะตอมกลาง (เฉพาะอะตอมที่มีเพื่อนบ้าน ≥ 1)
    lone: list[tuple[int, list[float]]] = []
    for k, index in enumerate(members):
        if neighbors[k]:
            for _ in range(atoms[index].lone_pairs):
                lone.append(
                    (
                        k,
                        [
                            pos[k][0] + rng.gauss(0, 0.3),
                            pos[k][1] + rng.gauss(0, 0.3),
                            pos[k][2] + rng.gauss(0, 0.3),
                        ],
                    )
                )
    vel = [[0.0, 0.0, 0.0] for _ in range(n)]
    lone_vel = [[0.0, 0.0, 0.0] for _ in lone]
    domains_max = max(
        (len(neighbors[k]) + sum(1 for c, _ in lone if c == k) for k in range(n)), default=0
    )
    if domains_max >= 5:  # 5–6 คู่อิเล็กตรอน (SF₆, XeF₄) ลู่เข้าช้ากว่า
        steps = max(steps, 1000)
    dt = RELAX_DT
    rings = len(bonds) >= n  # มีวง (พันธะ ≥ จำนวนอะตอม)
    flat_steps = steps // 2
    for _step in range(steps):
        force = [[0.0, 0.0, 0.0] for _ in range(n)]
        lone_force = [[0.0, 0.0, 0.0] for _ in lone]
        for i, j, bond in bonds:  # สปริงของพันธะ
            rest = (atoms[members[i]].radius + atoms[members[j]].radius) * BOND_SHORTEN[bond.order]
            _spring(pos[i], pos[j], rest, 6.0, force[i], force[j])
        for k, (center, p) in enumerate(lone):
            _spring(pos[center], p, 0.55, 6.0, force[center], lone_force[k])
        for center in range(n):  # คู่อิเล็กตรอนรอบอะตอมกลางผลักกัน (ตามมุม)
            domains: list[tuple[list[float], list[float], float]] = [
                (pos[j], force[j], 1.0) for j in neighbors[center]
            ]
            domains += [
                (p, lone_force[k], LONE_WEIGHT) for k, (c, p) in enumerate(lone) if c == center
            ]
            if len(domains) < 2:
                continue
            origin = pos[center]
            for a in range(len(domains)):
                for b in range(a + 1, len(domains)):
                    pa, fa, wa = domains[a]
                    pb, fb, wb = domains[b]
                    ua = _unit([pa[0] - origin[0], pa[1] - origin[1], pa[2] - origin[2]])
                    ub = _unit([pb[0] - origin[0], pb[1] - origin[1], pb[2] - origin[2]])
                    d = [ua[0] - ub[0], ua[1] - ub[1], ua[2] - ub[2]]
                    length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) + 1e-3
                    strength = DOMAIN_K * wa * wb / (length**2)
                    for axis in range(3):
                        push = d[axis] / length * strength
                        fa[axis] += push
                        fb[axis] -= push
        for i in range(n):  # อะตอมที่ไม่ได้ต่อกันผลักกันเบาๆ (กันทับ)
            for j in range(i + 1, n):
                d = [pos[i][0] - pos[j][0], pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]]
                dist2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 1e-4
                if dist2 > 9.0:
                    continue
                strength = 0.25 / dist2
                dist = math.sqrt(dist2)
                for axis in range(3):
                    force[i][axis] += d[axis] / dist * strength
                    force[j][axis] -= d[axis] / dist * strength
        for items, velocities, forces in (
            (pos, vel, force),
            ([p for _, p in lone], lone_vel, lone_force),
        ):
            for p, v, f in zip(items, velocities, forces, strict=True):
                for axis in range(3):
                    v[axis] = (v[axis] + f[axis] * dt) * RELAX_DAMPING
                    p[axis] += v[axis] * dt
        if rings and _step < flat_steps:  # ช่วงแรกแบนราบ: วงแหวนไม่บิดพันกัน แล้วค่อยปล่อยเป็น 3 มิติ
            for p in pos:
                p[2] *= 0.5
        elif rings and _step == flat_steps:
            for p in pos:
                p[2] += rng.gauss(0, 0.08)
    cx = sum(p[0] for p in pos) / n
    cy = sum(p[1] for p in pos) / n
    cz = sum(p[2] for p in pos) / n
    for k, index in enumerate(members):
        atoms[index].x, atoms[index].y, atoms[index].z = (
            pos[k][0] - cx,
            pos[k][1] - cy,
            pos[k][2] - cz,
        )


def _spectral_start(n: int, edges: list[tuple[int, int]], rng: random.Random) -> list[list[float]]:
    """ตำแหน่งเริ่มจาก eigenvector ของ graph Laplacian — วงแหวนออกมาเป็นรูปหลายเหลี่ยมไม่พันกัน
    (เริ่มแบบสุ่มแล้ววงของคาเฟอีน/เบนซีนบิดพันจนคลายไม่ออก)"""
    if n < 3:
        return [[rng.gauss(0, 1.0), rng.gauss(0, 1.0), rng.gauss(0, 1.0)] for _ in range(n)]
    import numpy as np

    laplacian = np.zeros((n, n))
    for i, j in edges:
        laplacian[i, j] = laplacian[j, i] = -1.0
        laplacian[i, i] += 1.0
        laplacian[j, j] += 1.0
    _values, vectors = np.linalg.eigh(laplacian)
    columns = [vectors[:, k] if k < n else np.zeros(n) for k in (1, 2, 3)]
    coords = np.stack(columns, axis=1)
    coords[:, 2] *= 0.3
    lengths = [np.linalg.norm(coords[i] - coords[j]) for i, j in edges] or [1.0]
    scale = 1.3 / max(float(np.mean(lengths)), 1e-6)
    return [
        [float(x) * scale + rng.gauss(0, 0.05), float(y) * scale + rng.gauss(0, 0.05),
         float(z) * scale + rng.gauss(0, 0.15)]
        for x, y, z in coords
    ]  # fmt: skip


def _spring(
    a: list[float], b: list[float], rest: float, k: float, fa: list[float], fb: list[float]
) -> None:
    d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    dist = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) + 1e-6
    pull = k * (dist - rest)
    for axis in range(3):
        fa[axis] += d[axis] / dist * pull
        fb[axis] -= d[axis] / dist * pull


def _unit(v: list[float]) -> list[float]:
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-9
    return [v[0] / length, v[1] / length, v[2] / length]


def _arrange(
    structure: Structure, groups: list[list[int]], gap: float = 1.6
) -> list[tuple[float, list[int]]]:
    """วางชิ้นส่วน (โมเลกุล/ไอออน) เรียงกันตามแกน x — คืน (ตำแหน่งกลาง, สมาชิก)"""
    sizes = []
    for members in groups:
        extent = max(
            (
                math.dist(
                    (0, 0, 0), (structure.atoms[i].x, structure.atoms[i].y, structure.atoms[i].z)
                )
                + structure.atoms[i].radius
            )
            for i in members
        )
        sizes.append(extent)
    total = sum(2 * s for s in sizes) + gap * (len(groups) - 1)
    x = -total / 2
    placed = []
    for members, size in zip(groups, sizes, strict=True):
        center = x + size
        for i in members:
            structure.atoms[i].x += center
        placed.append((center, members))
        x += 2 * size + gap
    return placed


def layout(structure: Structure, *, seed: int = 3, steps: int = 320) -> Structure:
    """จัดตำแหน่ง 3 มิติ (ครั้งเดียวตอนสร้าง) — โครงผลึกมีตำแหน่งอยู่แล้ว ไม่ต้องจัด"""
    if structure.kind in ("metallic", "network") or (
        structure.kind == "ionic" and not structure.bonds and len(structure.atoms) == 27
    ):
        return structure
    _lone_pairs(structure)
    rng = random.Random(seed)
    groups = _fragments(structure.atoms, structure.bonds)
    for members in groups:
        if len(members) == 1:
            atom = structure.atoms[members[0]]
            atom.x = atom.y = atom.z = 0.0
        else:
            _relax(structure, members, rng, steps)
    if len(groups) > 1 and not structure.labels:
        _arrange(structure, groups)
    return structure


# --------------------------------------------------------------------------- #
# รูปร่างโมเลกุล (VSEPR)
# --------------------------------------------------------------------------- #

_SHAPES = {
    (2, 0): ("linear · เส้นตรง", "180°"),
    (2, 1): ("bent · มุมงอ", "<120°"),
    (2, 2): ("bent · มุมงอ", "<109.5° (น้ำ ≈ 104.5°)"),
    (2, 3): ("linear · เส้นตรง", "180°"),
    (3, 0): ("trigonal planar · สามเหลี่ยมแบนราบ", "120°"),
    (3, 1): ("trigonal pyramidal · พีระมิดฐานสามเหลี่ยม", "<109.5° (NH₃ ≈ 107°)"),
    (3, 2): ("T-shaped · รูปตัว T", "<90°"),
    (4, 0): ("tetrahedral · ทรงสี่หน้า", "109.5°"),
    (4, 1): ("seesaw · ไม้กระดก", "<90°, <120°"),
    (4, 2): ("square planar · สี่เหลี่ยมแบนราบ", "90°"),
    (5, 0): ("trigonal bipyramidal · พีระมิดคู่ฐานสามเหลี่ยม", "90°, 120°"),
    (5, 1): ("square pyramidal · พีระมิดฐานสี่เหลี่ยม", "<90°"),
    (6, 0): ("octahedral · ทรงแปดหน้า", "90°"),
}


def vsepr_shape(structure: Structure) -> tuple[str | None, str | None, int | None]:
    """(ชื่อรูปร่าง, มุมพันธะตามทฤษฎี, อะตอมกลาง) — เฉพาะโมเลกุลที่มีอะตอมกลางเดียว"""
    if structure.kind != "molecule":
        return None, None, None
    groups = _fragments(structure.atoms, structure.bonds)
    if len(groups) != 1:
        return None, None, None
    if len(structure.atoms) == 2:
        return "linear · เส้นตรง (โมเลกุลอะตอมคู่)", "—", None
    centers = [i for i in range(len(structure.atoms)) if len(structure.neighbors(i)) >= 2]
    if len(centers) != 1:
        return None, None, None
    center = centers[0]
    key = (len(structure.neighbors(center)), structure.atoms[center].lone_pairs)
    shape = _SHAPES.get(key)
    return (shape[0], shape[1], center) if shape else (None, None, center)


# --------------------------------------------------------------------------- #
# ทางเข้า: สาร → โครงสร้าง
# --------------------------------------------------------------------------- #

_SPECIES = re.compile(r"(\d*)\s*([A-Z][A-Za-z0-9()]*)")
MAX_REACTION_ATOMS = 48


def _from_key(key: str, counts: Counter[str]) -> Structure:
    if key in _NETWORK_KEYS:
        first, second = _NETWORK_KEYS[key]
        return _network(first, second)
    smiles = STRUCTURES.get(key)
    if smiles is None:
        return build_formula(counts)
    atoms, bonds = parse_smiles(smiles)
    ionic = any(a.charge for a in atoms) and any(is_metal(a.symbol) for a in atoms)
    if ionic and not bonds and len(atoms) == 2 and atoms[0].charge == -atoms[1].charge:
        cation, anion = sorted(atoms, key=lambda a: -a.charge)
        if abs(cation.charge) <= 2:
            return _rock_salt(cation, anion)
    structure = Structure("ionic" if ionic else "molecule", atoms, bonds)
    if ionic:
        structure.note = "สารประกอบไอออนิก: แสดงหนึ่งหน่วยสูตร — ในผลึกจริงไอออนเรียงซ้ำกันเป็นโครงผลึก"
    return structure


def _element_structure(symbol: str) -> Structure:
    element = by_symbol(symbol)
    if element.category == "noble":
        return Structure(
            "atomic", [MolAtom(symbol)], [], note="แก๊สมีตระกูล: อยู่เป็นอะตอมเดี่ยว ไม่สร้างพันธะ"
        )
    if is_metal(symbol):
        return _metallic(Counter({symbol: 1}))
    if symbol in _NETWORK_ELEMENTS:
        structure = _network(symbol, None)
        if symbol == "C":
            structure.note = (
                "เพชร: คาร์บอนแต่ละอะตอมต่อกับอีก 4 อะตอมเป็นทรงสี่หน้า (แกรไฟต์จะเรียงเป็นแผ่นหกเหลี่ยมแทน)"
            )
        return structure
    form = _ELEMENT_FORMS.get(symbol)
    if form is not None:
        key = hill_formula(parse_formula(form))
        structure = _from_key(key, parse_formula(form))
        structure.note = f"ในธรรมชาติ{element.thai or element.name}อยู่เป็นโมเลกุล {formula_text(form)}"
        return structure
    return Structure("atomic", [MolAtom(symbol)], [], note="แสดงเป็นอะตอมเดี่ยว")


def _reaction(equation: str, side: int) -> Structure:
    parts = equation.split("→")
    text = parts[min(side, len(parts) - 1)]
    combined = Structure("reaction")
    groups: list[tuple[list[int], str]] = []
    for coefficient, formula in _SPECIES.findall(text):
        copies = int(coefficient or 1)
        counts = parse_formula(formula)
        if len(counts) == 1 and sum(counts.values()) == 1:
            symbol = next(iter(counts))
            piece = (
                Structure("metallic", [MolAtom(symbol) for _ in range(min(copies, 4))], [])
                if is_metal(symbol)
                else _element_structure(symbol)
            )
            copies = 1 if is_metal(symbol) else copies
        else:
            piece = _from_key(hill_formula(counts), counts)
            if piece.kind == "ionic" and len(piece.atoms) == 27:  # ไม่ใช้โครงผลึกเต็มในสมการ
                smiles = STRUCTURES.get(hill_formula(counts))
                atoms, bonds = parse_smiles(smiles) if smiles else (piece.atoms[:2], [])
                piece = Structure("ionic", atoms, bonds)
        label = (
            f"{coefficient}{formula_text(formula)}"
            if coefficient and coefficient != "1"
            else formula_text(formula)
        )
        if len(combined.atoms) + len(piece.atoms) * copies > MAX_REACTION_ATOMS:
            copies = 1
        members: list[int] = []
        for _ in range(copies):
            offset = len(combined.atoms)
            combined.atoms.extend(
                MolAtom(a.symbol, a.charge, x=a.x, y=a.y, z=a.z) for a in piece.atoms
            )
            combined.bonds.extend(MolBond(b.a + offset, b.b + offset, b.order) for b in piece.bonds)
            members.extend(range(offset, len(combined.atoms)))
        groups.append((members, label))
    _lone_pairs(combined)
    rng = random.Random(5)
    for members in _fragments(combined.atoms, combined.bonds):
        if len(members) > 1 and any(
            combined.bonds[k].a in members for k in range(len(combined.bonds))
        ):
            _relax(combined, members, rng, 260)
    # จัดแต่ละ "สาร" เป็นกลุ่ม แล้วเรียงกลุ่มตามสมการ
    for members, _label in groups:
        fragments = [f for f in _fragments(combined.atoms, combined.bonds) if f[0] in members]
        if len(fragments) > 1:
            sub = Structure("reaction", combined.atoms, combined.bonds)
            _arrange(sub, fragments, gap=0.9)
            cx = sum(combined.atoms[i].x for i in members) / len(members)
            for i in members:
                combined.atoms[i].x -= cx
    placed = _arrange(combined, [m for m, _ in groups], gap=2.4)
    combined.labels = [
        (center, max(combined.atoms[i].radius for i in members) + 2.2, 0.0, label)
        for (center, members), (_m, label) in zip(placed, groups, strict=True)
    ]
    combined.note = "ก่อนเกิดปฏิกิริยา (สารตั้งต้น)" if side == 0 else "หลังเกิดปฏิกิริยา (ผลิตภัณฑ์)"
    combined.title = text.strip()
    return combined


def structure_for(subject, *, side: int = 0) -> Structure:
    """โครงสร้าง 3 มิติของ :class:`aether.studio.science.Subject` — ``side`` = 0 ก่อน / 1 หลังปฏิกิริยา"""
    if subject.is_reaction:
        return _reaction(subject.compound.equation, side)
    if subject.element is not None:
        structure = _element_structure(subject.element.symbol)
    else:
        structure = _from_key(subject.key, subject.counts)
    structure.title = subject.display
    layout(structure)
    shape, angle, _center = vsepr_shape(structure)
    structure.shape, structure.angle = shape, angle
    return structure

"""ห้องทดลองเคมีใน Studio: ข้อมูลธาตุ, สูตรเคมี, ขวดผสม, การหยิบ/ปล่อยด้วยมือ และหน้า Elements"""

from __future__ import annotations

from collections import Counter

import pytest

from aether.config import Config
from aether.studio.chemistry import (
    CATEGORIES,
    COMPOUNDS,
    ELEMENTS,
    MAX_ATOMS,
    ChemLab,
    Flask,
    by_symbol,
    formula_text,
    hill_formula,
    molar_mass,
    parse_formula,
)
from aether.studio.input import StudioHand
from aether.studio.window import StudioWindow


def hand(x=0.5, y=0.5, *, pinch=False, gesture="None", side="right") -> StudioHand:
    return StudioHand(side, x, y, pinch, 1.0 if pinch else 0.0, gesture, (True,) * 5)


# ---- ตารางธาตุ ---------------------------------------------------------------- #


def test_periodic_table_has_all_118_elements_in_unique_cells():
    assert [e.number for e in ELEMENTS] == list(range(1, 119))
    assert len({e.symbol for e in ELEMENTS}) == 118
    assert len({(e.row, e.column) for e in ELEMENTS}) == 118
    assert Counter(e.category for e in ELEMENTS) == {
        "transition": 38,
        "lanthanide": 15,
        "actinide": 15,
        "post": 12,
        "nonmetal": 7,
        "noble": 7,
        "alkali": 6,
        "alkaline": 6,
        "metalloid": 6,
        "halogen": 6,
    }
    assert set(CATEGORIES) == {e.category for e in ELEMENTS}


@pytest.mark.parametrize(
    ("symbol", "period", "group"),
    [
        ("H", 1, 1),
        ("He", 1, 18),
        ("B", 2, 13),
        ("Ar", 3, 18),
        ("K", 4, 1),
        ("Fe", 4, 8),
        ("Ag", 5, 11),
        ("Cs", 6, 1),
        ("Hf", 6, 4),
        ("Au", 6, 11),
        ("Rn", 6, 18),
        ("Rf", 7, 4),
        ("Og", 7, 18),
        ("La", 6, None),
        ("Lr", 7, None),
    ],
)
def test_elements_sit_in_the_right_period_and_group(symbol, period, group):
    element = by_symbol(symbol)
    assert (element.period, element.group) == (period, group)
    if group is not None:
        assert (element.row, element.column) == (period - 1, group - 1)


def test_f_block_rows_and_masses():
    lanthanides = [e for e in ELEMENTS if e.row == 7]
    actinides = [e for e in ELEMENTS if e.row == 8]
    assert [e.symbol for e in (lanthanides[0], lanthanides[-1])] == ["La", "Lu"]
    assert [e.symbol for e in (actinides[0], actinides[-1])] == ["Ac", "Lr"]
    assert [e.column for e in lanthanides] == list(range(2, 17))
    assert by_symbol("C").mass == pytest.approx(12.011)
    assert by_symbol("Tc").mass_text == "[98]" and not by_symbol("Tc").stable_mass
    assert by_symbol("U").mass_text == "238.03"
    assert by_symbol("Fe").thai == "เหล็ก" and by_symbol("Os").thai == ""


# ---- สูตรเคมี ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("formula", "counts"),
    [
        ("H2O", {"H": 2, "O": 1}),
        ("NaHCO3", {"Na": 1, "H": 1, "C": 1, "O": 3}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("CH3COOH", {"C": 2, "H": 4, "O": 2}),
        ("Al2(SO4)3", {"Al": 2, "S": 3, "O": 12}),
    ],
)
def test_parse_formula(formula, counts):
    assert parse_formula(formula) == Counter(counts)


@pytest.mark.parametrize("bad", ["Xx2", "H2(O", "H2O)", "h2o", "H2O+"])
def test_parse_formula_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_formula_helpers():
    assert formula_text("C6H12O6") == "C₆H₁₂O₆"
    assert hill_formula({"O": 2, "C": 1}) == "CO2"
    assert hill_formula({"H": 4, "C": 1}) == "CH4"
    assert hill_formula({"Na": 1, "Cl": 1}) == "ClNa"  # ไม่มี C: เรียงตามตัวอักษร
    assert molar_mass(parse_formula("H2O")) == pytest.approx(18.015, abs=0.01)
    assert molar_mass(parse_formula("NaCl")) == pytest.approx(58.44, abs=0.01)


def test_compounds_are_valid_and_distinct():
    seen = set()
    for compound in COMPOUNDS:
        key = tuple(sorted(compound.counts.items()))
        assert key not in seen, compound.formula  # สูตรซ้ำ = แยกไม่ออกว่าผสมได้สารไหน
        seen.add(key)
        assert compound.atoms <= MAX_ATOMS
        assert compound.thai and compound.fact and compound.color.startswith("#")
    assert len(COMPOUNDS) >= 40


# ---- ขวดผสม -------------------------------------------------------------------- #


def test_flask_matches_exact_and_multiple_recipes():
    flask = Flask()
    for symbol in "HHO":
        flask.add(symbol)
    compound, multiple = flask.match()
    assert (compound.formula, multiple) == ("H2O", 1)
    for symbol in "OHH":
        flask.add(symbol)
    compound, multiple = flask.match()
    assert (compound.formula, multiple) == ("H2O", 2)
    flask.add("H")
    assert flask.match() is None and flask.formula() == "H₅O₂"
    assert flask.undo() == "H"
    flask.clear()
    assert flask.match() is None and flask.atoms == []


def test_flask_order_does_not_matter_and_capacity_is_limited():
    flask = Flask()
    for symbol in ("O", "Cl", "Na"):
        flask.add(symbol)
    assert flask.match()[0].formula == "NaClO"
    assert not flask.add("Zz")
    small = Flask(capacity=2)
    assert small.add("H") and small.add("H") and not small.add("H") and small.full


def test_flask_hints_list_what_is_missing():
    flask = Flask()
    flask.add("C")
    hints = flask.hints()
    assert hints[0].compound.formula == "CO" and hints[0].text == "+O → CO"
    assert [h.compound.formula for h in hints[1:]] == ["SiC", "WC"]  # ขาดอีกหนึ่งอะตอมเหมือนกัน
    more = {h.compound.formula: h.text for h in flask.hints(limit=10)}
    assert more["CO2"] == "+O×2 → CO₂"
    flask.add("Xe")
    assert flask.hints() == []  # ไม่มีสารไหนมีทั้ง C และ Xe
    assert Flask().hints() == []


# ---- มือ: หยิบ ลาก ปล่อย -------------------------------------------------------- #


def table_and_flask(x: float, y: float):
    """ตารางจำลอง: ครึ่งบนแบ่งเป็นช่อง H/O/Na/Cl ตามแนวนอน, ขวดอยู่ครึ่งล่าง"""
    if y < 0.5:
        return by_symbol(("H", "O", "Na", "Cl")[min(int(x * 4), 3)])
    return None


def in_flask(x: float, y: float) -> bool:
    return y > 0.7


def run(lab: ChemLab, hands, now: float):
    return lab.update(hands, now, cell_at=table_and_flask, in_flask=in_flask)


def carry(lab: ChemLab, x: float, now: float) -> list:
    events = []
    events += run(lab, (hand(x, 0.2),), now)  # ชี้ที่ช่อง
    events += run(lab, (hand(x, 0.2, pinch=True),), now + 0.05)  # จีบ = หยิบ
    events += run(lab, (hand(0.5, 0.8, pinch=True),), now + 0.3)  # ลากไปที่ขวด
    events += run(lab, (hand(0.5, 0.8),), now + 0.4)  # ปล่อย
    return events


def test_pinch_picks_an_atom_and_releasing_over_the_flask_adds_it():
    lab = ChemLab()
    run(lab, (hand(0.1, 0.2),), 0.0)
    assert lab.hover.symbol == "H"
    events = carry(lab, 0.1, 0.0)
    assert [e.kind for e in events] == ["pick", "add"] and lab.flask.atoms == ["H"]
    carry(lab, 0.1, 1.0)
    events = carry(lab, 0.3, 2.0)
    kinds = [e.kind for e in events]
    assert kinds == ["pick", "add", "match"]
    assert events[-1].compound.formula == "H2O" and lab.last_match[0].name == "Water"


def test_releasing_away_from_the_flask_drops_the_atom():
    lab = ChemLab()
    run(lab, (hand(0.6, 0.2, pinch=True),), 0.0)
    assert lab.held == {"right": "Na"}
    events = run(lab, (hand(0.6, 0.6),), 0.1)
    assert [e.kind for e in events] == ["drop"] and lab.flask.atoms == []
    run(lab, (hand(0.6, 0.2, pinch=True),), 0.2)
    assert [e.kind for e in run(lab, (), 0.3)] == ["drop"]  # มือหายระหว่างถือ


def test_two_hands_carry_two_atoms():
    lab = ChemLab()
    run(lab, (hand(0.6, 0.2, pinch=True), hand(0.9, 0.2, pinch=True, side="left")), 0.0)
    assert lab.held == {"right": "Na", "left": "Cl"}
    events = run(lab, (hand(0.5, 0.8), hand(0.5, 0.8, side="left")), 0.2)
    assert [e.kind for e in events] == ["add", "add", "match"]
    assert events[-1].compound.formula == "NaCl" and events[-1].compound.energetic


def test_hold_gestures_remove_empty_and_toggle_the_table():
    lab = ChemLab()
    for symbol in "HHO":
        lab.flask.add(symbol)

    def hold(gesture: str, start: float) -> list[str]:
        kinds, now = [], start
        while now < start + 1.5:
            kinds += [e.kind for e in run(lab, (hand(0.5, 0.6, gesture=gesture),), now)]
            now += 0.05
        run(lab, (hand(0.5, 0.6),), now)
        return kinds

    # เอา O ออกจาก H₂O → เหลือ H₂ ซึ่งเป็นสาร (ก๊าซไฮโดรเจน) พอดี
    assert hold("Victory", 0.0) == ["undo", "match"] and lab.flask.atoms == ["H", "H"]
    assert lab.last_match[0].formula == "H2"
    assert hold("Thumb_Up", 2.0) == ["table"]
    assert hold("Closed_Fist", 4.0) == ["empty"] and lab.flask.atoms == []
    assert hold("Closed_Fist", 6.0) == []  # ขวดว่างแล้ว: ไม่มีอะไรให้เท


# ---- หน้า Elements ------------------------------------------------------------- #


@pytest.fixture
def studio(qapp):
    clock = {"t": 0.0}
    window = StudioWindow(Config(), audio=False, clock=lambda: clock["t"])
    window.resize(1180, 780)
    window.show()
    window.set_page(window.pages.index(window.elements))
    yield window, clock
    window.close()


def test_elements_page_geometry_and_group_pages(studio):
    window, _ = studio
    page = window.elements
    assert window.current is page and "flick" in window.hint.text()
    for element in (by_symbol("H"), by_symbol("Og"), by_symbol("La"), by_symbol("Lr")):
        rect = page.cell_rect(element)
        x, y = rect.center().x() / page.width(), rect.center().y() / page.height()
        assert page.cell_at(x, y) is element
    assert page.content_rect().right() < page.right_rect().left()
    flask = page.flask_rect()
    assert page.in_flask(flask.center().x() / page.width(), flask.center().y() / page.height())
    page.toggle_table()  # 👍 / ปุ่ม: ไปหน้าหมวดล่าสุด (เริ่มที่โลหะแอลคาไล)
    assert page.pages[page.page_index][0] == "alkali" and not page.table_open
    assert page.cell_rect(by_symbol("H")).isEmpty()  # H ไม่ใช่แอลคาไล: ไม่อยู่ในหน้านี้
    lithium = page.cell_rect(by_symbol("Li"))
    assert (
        page.cell_at(
            lithium.center().x() / page.width(), lithium.center().y() / page.height()
        ).symbol
        == "Li"
    )
    page.toggle_table()
    assert page.table_open
    window.grab()


# ---- รายละเอียดธาตุ / ข้อมูลชุดใหม่ ------------------------------------------------- #


@pytest.mark.parametrize(
    ("symbol", "config", "shells"),
    [
        ("H", "1s¹", [1]),
        ("O", "[He] 2s² 2p⁴", [2, 6]),
        ("Na", "[Ne] 3s¹", [2, 8, 1]),
        ("Fe", "[Ar] 3d⁶ 4s²", [2, 8, 14, 2]),
        ("Cr", "[Ar] 3d⁵ 4s¹", [2, 8, 13, 1]),
        ("Cu", "[Ar] 3d¹⁰ 4s¹", [2, 8, 18, 1]),
        ("Pd", "[Kr] 4d¹⁰", [2, 8, 18, 18]),
        ("Au", "[Xe] 4f¹⁴ 5d¹⁰ 6s¹", [2, 8, 18, 32, 18, 1]),
        ("U", "[Rn] 5f³ 6d¹ 7s²", [2, 8, 18, 32, 21, 9, 2]),
        ("Og", "[Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁶", [2, 8, 18, 32, 32, 18, 8]),
    ],
)
def test_electron_configuration(symbol, config, shells):
    from aether.studio.atoms import configuration_text
    from aether.studio.atoms import shells as shell_counts

    number = by_symbol(symbol).number
    assert configuration_text(number) == config and shell_counts(number) == shells


def test_every_element_has_details():
    from aether.studio.atoms import atom_info, shells

    for element in ELEMENTS:
        info = atom_info(element)
        assert sum(shells(element.number)) == element.number == info.electrons == info.protons
        assert info.neutrons >= 0 and info.note and info.block in "spdf"
    assert atom_info("Fe").neutrons == 30 and atom_info("Fe").valence is None
    assert atom_info("Cl").valence == 7 and atom_info("Cl").state == "gas"
    assert atom_info("Hg").state == "liquid" and atom_info("Og").state == "unknown"
    assert atom_info("He").valence == 2 and atom_info("He").block == "s"
    assert (
        atom_info("Na").block == "s"
        and atom_info("Fe").block == "d"
        and atom_info("U").block == "f"
    )


def test_compound_kinds_dangers_and_reactions():
    from aether.studio.chemistry import DANGER_LEVELS, KINDS

    for compound in COMPOUNDS:
        assert compound.kind in KINDS and 0 <= compound.danger < len(DANGER_LEVELS)
        assert compound.is_reaction == (compound.kind == "reaction")
        if compound.is_reaction:
            assert "→" in compound.equation
            left = compound.equation.split("→")[0]
            reactants = Counter()
            for term in left.split("+"):
                term = term.strip()
                count = int(term[0]) if term[0].isdigit() else 1
                for symbol, n in parse_formula(term.lstrip("0123456789")).items():
                    reactants[symbol] += n * count
            assert reactants == compound.counts, compound.name  # อะตอมของสารตั้งต้นตรงกับสูตรในขวด
    assert any(c.danger == 3 and c.kind == "explosive" for c in COMPOUNDS)
    water = next(c for c in COMPOUNDS if c.formula == "H2O")
    assert water.recipe == "H×2 + O"
    assert formula_text("2Al + Fe2O3 → Al2O3 + 2Fe") == "2Al + Fe₂O₃ → Al₂O₃ + 2Fe"


def test_exact_match_wins_over_multiples():
    flask = Flask()
    for symbol in "C" * 6 + "H" * 12 + "O" * 6:
        flask.add(symbol)
    assert flask.match()[0].name == "Glucose"  # ไม่ใช่ 6 × ฟอร์มาลดีไฮด์
    flask.clear()
    for symbol in "C" * 6 + "H" * 6:
        flask.add(symbol)
    assert flask.match()[0].name == "Benzene"  # ไม่ใช่ 3 × อะเซทิลีน
    flask.clear()
    for symbol in "O" * 6:
        flask.add(symbol)
    assert flask.match() == (next(c for c in COMPOUNDS if c.formula == "O3"), 2)


def test_swipe_detector_rules():
    from aether.studio.input import SwipeDetector

    swipe = SwipeDetector()

    def palm(vx: float, vy: float = 0.0, side: str = "right", gesture: str = "Open_Palm"):
        return StudioHand(side, 0.5, 0.5, False, 0.0, gesture, (True,) * 5, vx, vy)

    assert swipe.update(palm(-2.5), 0.0) == -1
    assert swipe.update(palm(-2.5), 0.3) == 0  # cooldown
    assert swipe.update(palm(2.5), 1.0) == 1
    assert swipe.update(palm(1.0), 2.0) == 0  # ช้าเกินไป
    assert swipe.update(palm(2.0, 2.0), 3.0) == 0  # ทแยง ไม่ใช่ปัดแนวนอน
    assert swipe.update(palm(3.0, side="mouse"), 4.0) == 0  # เมาส์ไม่นับ
    fist = StudioHand("right", 0.5, 0.5, False, 0.0, "Closed_Fist", (False,) * 5, 3.0, 0.0)
    assert swipe.update(fist, 5.0) == 0

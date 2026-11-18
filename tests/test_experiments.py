"""เครื่องมือทดลองของ Studio: ข้อมูลสาร/สถานะ/อากาศ/แรงโน้มถ่วง, โมเลกุล 3 มิติ (SMILES + VSEPR),
จำลองสถานะสสาร และหน้า Experiments ที่เชื่อมกับหน้า Elements"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from aether.config import Config
from aether.studio import molecule as mol
from aether.studio import science as sci
from aether.studio.chemistry import COMPOUNDS, by_symbol, hill_formula, parse_formula
from aether.studio.experiments_page import DROP_HEIGHT_M, TOOLS
from aether.studio.input import StudioHand
from aether.studio.matter import MatterSim, slider_from_temperature, temperature_from_slider
from aether.studio.science import Subject
from aether.studio.window import StudioWindow


def compound(formula: str):
    return next(c for c in COMPOUNDS if c.formula == formula)


def reaction(name: str):
    return next(c for c in COMPOUNDS if c.name == name)


# ---- ข้อมูลสาร ------------------------------------------------------------------- #


def test_phases_from_real_melting_and_boiling_points():
    water = Subject.from_compound(compound("H2O"))
    assert [sci.phase_at(water, t) for t in (-10, 25, 150)] == ["solid", "liquid", "gas"]
    dry_ice = Subject.from_compound(compound("CO2"))
    assert sci.phase_at(dry_ice, -100) == "solid" and sci.phase_at(dry_ice, -50) == "gas"
    helium = Subject.from_element(by_symbol("He"))
    assert sci.phase_at(helium, -270) == "liquid" and sci.phase_at(helium, -273) == "liquid"
    gold = Subject.from_element(by_symbol("Au"))  # ไม่มีจุดเดือดในข้อมูล: บอกได้แค่เหลว
    assert sci.phase_at(gold, 2000) == "liquid" and sci.phase_at(gold, 500) == "solid"
    caffeine = Subject.from_compound(compound("C8H10N4O2"))
    assert sci.phase_at(caffeine, 25) is None and sci.room_state(caffeine) == "unknown"
    assert sci.room_state(Subject.from_element(by_symbol("Og"))) == "unknown"
    hydrogen_gas = Subject.from_compound(compound("H2"))  # H₂ จากขวด ใช้ข้อมูลของธาตุ H
    assert sci.phase_at(hydrogen_gas, -255) == "liquid"
    assert sci.room_state(Subject.from_element(by_symbol("Br"))) == "liquid"


def test_subject_basics():
    water = Subject.from_formula("H2O", "Water", "น้ำ")
    assert water.kind == "compound" and water.key == "H2O" and water.main_symbol == "O"
    assert water.molar_mass == pytest.approx(18.015, abs=0.01)
    assert Subject.from_element(by_symbol("O")).molar_mass == pytest.approx(31.998, abs=0.01)
    mix = Subject.from_counts({"C": 2, "H": 6, "Xe": 0})
    assert mix.kind == "mix" and mix.formula == "C2H6"
    assert Subject.from_formula("Fe").kind == "element"
    sodium = Subject.from_compound(reaction("Sodium + water"))
    assert sodium.is_reaction and sodium.chip == "Sodium + water"


def test_air_balloon_and_density():
    assert sci.AIR_MOLAR_MASS == pytest.approx(28.96, abs=0.05)
    verdict = {
        f: sci.balloon_for(Subject.from_formula(f)).verdict for f in ("He", "CH4", "CO2", "Ne")
    }
    assert verdict == {"He": "rises", "CH4": "rises", "CO2": "sinks", "Ne": "rises"}
    assert sci.balloon_for(Subject.from_element(by_symbol("Cl"))).verdict == "sinks"
    assert sci.balloon_for(Subject.from_compound(compound("N2"))).verdict == "floats"
    water = sci.balloon_for(Subject.from_compound(compound("H2O")))
    assert water.verdict == "not_gas" and not water.gas and "ของเหลว" in water.text
    helium = sci.balloon_for(Subject.from_formula("He"))
    assert helium.ratio == pytest.approx(7.2, abs=0.1) and "7.2" in helium.text
    co2 = Subject.from_compound(compound("CO2"))
    assert sci.density(co2) == pytest.approx(44.01 / 22.414 * 273.15 / 298.15 / 1000, rel=1e-3)
    assert sci.diffusion_speed(Subject.from_formula("He")) > 2.5


def test_gravity_weights_and_fall():
    iron = Subject.from_element(by_symbol("Fe"))
    assert sci.cube_mass_kg(iron) == pytest.approx(7.87)
    assert sci.weight_newton(7.87, 9.81) == pytest.approx(77.2, abs=0.1)
    assert sci.fall_time(2.0, 9.81) == pytest.approx(0.639, abs=0.001)
    assert sci.fall_time(2.0, 1.62) > sci.fall_time(2.0, 24.79)
    assert sci.cube_mass_kg(Subject.from_compound(compound("C8H10N4O2"))) is None
    assert sci.cube_mass_kg(Subject.from_formula("He")) < 0.001  # กล่อง 1 ลิตรของแก๊ส
    assert [p[0] for p in sci.PLANETS][4] == "Earth"


# ---- โมเลกุล --------------------------------------------------------------------- #


def test_every_curated_structure_matches_its_formula():
    for key, smiles in mol.STRUCTURES.items():
        atoms, _bonds = mol.parse_smiles(smiles)
        assert hill_formula(Counter(a.symbol for a in atoms)) == key, smiles


def test_every_built_in_compound_builds():
    for item in COMPOUNDS:
        structure = mol.structure_for(Subject.from_compound(item))
        assert structure.atoms
        if structure.kind in ("network", "metallic") or len(structure.atoms) == 27:
            continue  # โครงผลึก: แสดงหลายหน่วยสูตร
        assert hill_formula(structure.counts()) == hill_formula(item.counts), item.formula
    after = mol.structure_for(Subject.from_compound(reaction("Sodium + water")), side=1)
    assert after.kind == "reaction" and [label for *_xyz, label in after.labels] == ["2NaOH", "H₂"]


def angles(structure, center):
    pairs = itertools.combinations(structure.neighbors(center), 2)
    return sorted(structure.angle_deg(a, center, b) for a, b in pairs)


@pytest.mark.parametrize(
    ("formula", "shape", "low", "high"),
    [
        ("H2O", "bent", 100, 110),
        ("NH3", "trigonal pyramidal", 102, 110),
        ("CH4", "tetrahedral", 105, 114),
        ("CO2", "linear", 170, 180.1),
        ("SO2", "bent", 105, 122),
    ],
)
def test_vsepr_shapes(formula, shape, low, high):
    structure = mol.structure_for(Subject.from_formula(formula))
    assert structure.shape.startswith(shape)
    _shape, _angle, center = mol.vsepr_shape(structure)
    found = angles(structure, center)
    assert low <= found[0] and found[-1] <= high, found


def test_six_domains_and_planar_rings():
    sf6 = mol.structure_for(Subject.from_compound(compound("SF6")))
    assert sf6.shape.startswith("octahedral")
    center = next(i for i, a in enumerate(sf6.atoms) if a.symbol == "S")
    found = angles(sf6, center)
    assert 84 <= found[0] and found[-1] >= 174
    xef4 = mol.structure_for(Subject.from_compound(compound("XeF4")))
    assert xef4.shape.startswith("square planar")
    assert xef4.atoms[next(i for i, a in enumerate(xef4.atoms) if a.symbol == "Xe")].lone_pairs == 2
    benzene = mol.structure_for(Subject.from_compound(compound("C6H6")))
    carbons = [i for i, a in enumerate(benzene.atoms) if a.symbol == "C"]
    points = np.array(
        [[benzene.atoms[i].x, benzene.atoms[i].y, benzene.atoms[i].z] for i in carbons]
    )
    points -= points.mean(0)
    assert np.linalg.svd(points, compute_uv=False)[-1] / math.sqrt(6) < 0.08  # แบนราบ
    for c in carbons:
        assert all(115 <= a <= 125 for a in angles(benzene, c))


def test_bond_types_and_lattices():
    salt = mol.structure_for(Subject.from_compound(compound("NaCl")))
    assert salt.kind == "ionic" and len(salt.atoms) == 27
    by_position = {(round(a.x, 1), round(a.y, 1), round(a.z, 1)): a for a in salt.atoms}
    center = by_position[(0.0, 0.0, 0.0)]
    neighbour = by_position[(2.8, 0.0, 0.0)]
    assert center.charge == -neighbour.charge  # สลับบวก-ลบ
    chalk = mol.structure_for(Subject.from_compound(compound("CaCO3")))
    assert chalk.kind == "ionic" and sum(a.charge for a in chalk.atoms) == 0
    assert mol.structure_for(Subject.from_element(by_symbol("Fe"))).kind == "metallic"
    assert mol.structure_for(Subject.from_element(by_symbol("Ne"))).kind == "atomic"
    diamond = mol.structure_for(Subject.from_element(by_symbol("C")))
    assert diamond.kind == "network"
    inner = [i for i in range(len(diamond.atoms)) if len(diamond.neighbors(i)) == 4]
    assert len(inner) == 4  # อะตอมในเซลล์ต่อ 4 พันธะ (ทรงสี่หน้า)
    quartz = mol.structure_for(Subject.from_compound(compound("SiO2")))
    oxygens = [i for i, a in enumerate(quartz.atoms) if a.symbol == "O"]
    assert oxygens and all(
        {quartz.atoms[n].symbol for n in quartz.neighbors(o)} == {"Si"}
        and len(quartz.neighbors(o)) == 2
        for o in oxygens
    )
    oxygen = mol.structure_for(Subject.from_element(by_symbol("O")))
    assert [b.order for b in oxygen.bonds] == [2] and "O₂" in oxygen.note
    nitrogen = mol.structure_for(Subject.from_element(by_symbol("N")))
    assert [b.order for b in nitrogen.bonds] == [3]
    phosphorus = mol.structure_for(Subject.from_element(by_symbol("P")))
    assert len(phosphorus.atoms) == 4 and all(len(phosphorus.neighbors(i)) == 3 for i in range(4))
    water = mol.structure_for(Subject.from_formula("H2O"))
    assert next(a for a in water.atoms if a.symbol == "O").lone_pairs == 2


def test_flask_mixes_get_approximate_structures():
    ethane = mol.structure_for(Subject.from_counts(parse_formula("C2H6")))
    assert ethane.approximate and hill_formula(ethane.counts()) == "C2H6"
    cc = [b for b in ethane.bonds if {ethane.atoms[b.a].symbol, ethane.atoms[b.b].symbol} == {"C"}]
    assert len(cc) == 1 and cc[0].order == 1
    ethyne = mol.build_formula(Counter({"C": 2, "H": 2}))
    assert max(b.order for b in ethyne.bonds) == 3  # วาเลนซ์ที่เหลือกลายเป็นพันธะสาม
    magnesium = mol.structure_for(Subject.from_counts(parse_formula("MgCl2")))
    assert magnesium.kind == "ionic" and {a.charge for a in magnesium.atoms} == {2, -1}
    sulfate = mol.build_formula(Counter({"Na": 2, "S": 1, "O": 4}))
    assert sulfate.kind == "ionic" and sum(a.charge for a in sulfate.atoms) == 0
    assert mol.structure_for(Subject.from_counts(parse_formula("Fe3Al"))).kind == "metallic"
    with pytest.raises(ValueError):
        mol.parse_smiles("C1CC")


# ---- สถานะสสาร ---------------------------------------------------------------------- #


def test_matter_sim_behaves_like_each_phase():
    sim = MatterSim()

    def settle(phase, celsius, seconds=6.0):
        sim.set_state(phase, celsius)
        for _ in range(int(seconds * 60)):
            sim.step(1 / 60)

    settle("solid", -20)
    solid_spread = sim.spread()
    assert sim.mean_offset() < 0.02
    settle("gas", 150)
    assert sim.spread() > 0.5 > solid_spread
    hot = sim.speed
    sim.set_state("gas", 1000)
    assert sim.speed > hot  # ร้อนขึ้น = เร็วขึ้น
    settle("liquid", 50)
    assert sim.spread() < 0.4  # ไหลลงก้นภาชนะ
    assert all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in sim.particles)
    sim.set_state(None, 80)
    assert sim.phase == "liquid" and sim.celsius == 80
    for celsius in (-200, 0, 100, 1500):
        assert temperature_from_slider(slider_from_temperature(celsius)) == pytest.approx(
            celsius, abs=0.01
        )


# ---- หน้า Experiments --------------------------------------------------------------- #


@pytest.fixture
def lab(qapp):
    clock = {"t": 0.0}
    window = StudioWindow(Config(), audio=False, clock=lambda: clock["t"])
    window.resize(1180, 780)
    window.show()
    page = window.experiments
    window.set_page(window.pages.index(page))
    yield window, page, clock
    window.close()


def run(page, clock, seconds: float) -> None:
    for _ in range(int(seconds * 60)):
        clock["t"] += 1 / 60
        page.tick(clock["t"])


def hand(page, point: QPointF, *, pinch=False, gesture="None", vx=0.0) -> StudioHand:
    return StudioHand(
        "right", point.x() / page.width(), point.y() / page.height(), pinch, 1.0 if pinch else 0.0,
        gesture, (True,) * 5, vx,
    )  # fmt: skip


def test_page_renders_every_tool_for_many_substances(lab):
    window, page, clock = lab
    assert len(window.pages) == 7 and window.current is page
    subjects = [
        Subject.from_compound(compound("H2O")),
        Subject.from_compound(compound("NaCl")),
        Subject.from_compound(reaction("Thermite")),
        Subject.from_element(by_symbol("Fe")),
        Subject.from_element(by_symbol("He")),
        Subject.from_counts(parse_formula("C3H4")),
    ]
    for key, _label in TOOLS:
        page.set_tool(key)
        for subject in subjects:
            page.set_subject(subject)
            run(page, clock, 0.2)
            window.grab()
    page.set_subject(subjects[2])
    page.set_tool("molecule")
    page.toggle_side()
    assert page.side == 1 and page.structure().note.startswith("หลัง")


def test_pinch_drag_rotates_heats_lifts_and_drops(lab):
    window, page, clock = lab
    view = page.view_rect()
    page.set_tool("molecule")
    yaw = page.yaw
    start = view.center()
    page.set_hands((hand(page, start, pinch=True),), clock["t"])
    page.set_hands((hand(page, start + QPointF(120, 0), pinch=True),), clock["t"])
    page.set_hands((hand(page, start + QPointF(120, 0)),), clock["t"])
    assert page.yaw > yaw + 0.4
    page.set_tool("matter")
    page.set_subject(Subject.from_compound(compound("H2O")))
    top = QPointF(view.center().x(), view.top() + view.height() * 0.1)
    page.set_hands((hand(page, top, pinch=True),), clock["t"])
    assert page.celsius > 1000 and page.matter.phase == "gas"
    bottom = QPointF(view.center().x(), view.bottom() - 5)
    page.set_hands((hand(page, bottom, pinch=True),), clock["t"])
    assert page.celsius < -200 and page.matter.phase == "solid"
    page.set_hands((hand(page, bottom),), clock["t"])
    page.preset("boil")
    assert page.celsius > 100 and page.matter.phase == "gas"
    page.preset("room")
    assert page.celsius == 25.0
    page.set_tool("air")
    page.set_subject(Subject.from_formula("He"))
    start_y = page.balloon_y
    run(page, clock, 1.5)
    assert page.balloon_y < start_y  # ฮีเลียมลอยขึ้น
    page.set_subject(Subject.from_compound(compound("CO2")))
    run(page, clock, 1.5)
    assert page.balloon_y > 0.55  # CO₂ จมลง
    page.set_tool("gravity")
    page.set_subject(Subject.from_element(by_symbol("Fe")))
    moon = next(i for i, p in enumerate(sci.PLANETS) if p[0] == "Moon")
    _rect, action = next((r, a) for r, a in page.zones() if r == page._planet_rects()[moon][0])
    action()
    assert page.planet == moon
    cube = page._cube_rect().center()
    page.set_hands((hand(page, cube, pinch=True),), clock["t"])
    page.set_hands((hand(page, cube + QPointF(0, 40), pinch=True),), clock["t"])
    lifted = page.cube_height
    assert 0.0 < lifted < DROP_HEIGHT_M
    page.set_hands((hand(page, cube + QPointF(0, 40)),), clock["t"])  # ปล่อย = ตก
    assert page.falling_since is not None
    run(page, clock, 3.0)
    assert page.cube_height == 0.0 and page.last_fall == pytest.approx(sci.fall_time(lifted, 1.62))
    window.grab()


def test_zones_swipe_and_wheel(lab):
    _window, page, clock = lab
    _tab, action = page.zones()[2]  # แท็บที่ 3 = Air
    action()
    assert page.tool == "air"
    for step in range(12):  # แบมือสะบัดไปซ้าย = เครื่องมือถัดไป
        point = QPointF(page.width() * (0.8 - step * 0.05), page.height() * 0.5)
        page.set_hands((hand(page, point, gesture="Open_Palm", vx=-2.4),), clock["t"] + step * 0.03)
    assert page.tool == "gravity"
    subjects = page.subject_choices()
    assert (
        any(s.key == "H2O" for s in subjects) and subjects[0].element is not None
    )  # ธาตุจาก Elements


def test_elements_and_experiments_link_both_ways(lab):
    window, page, clock = lab
    elements = window.elements
    window.set_page(window.pages.index(elements))
    for symbol in ("Na", "Cl"):
        elements.drop_atom(symbol)
    assert elements.experiment_subject().key == "ClNa"
    window.grab()  # ชิป "Experiment with NaCl ›"
    _rect, action = next((r, a) for r, a in elements.zones() if r == elements.experiment_rect())
    action()
    assert window.current is page and page.subject.compound is compound("NaCl")
    assert page.subject_choices()[0].key == "ClNa"  # สารที่เพิ่งผสมได้มาก่อน
    page.set_subject(Subject.from_compound(compound("CO2")))
    page.mix()  # กลับไปผสมในขวด
    assert window.current is elements
    for step in range(30):
        elements.tick(clock["t"] + 1 + step)
    assert elements.lab.last_match[0] is compound("CO2")
    elements.empty()
    for symbol in ("C", "H", "H"):
        elements.drop_atom(symbol)
    assert elements.experiment_subject().kind == "mix"
    elements.open_experiment()
    assert page.subject.kind == "mix" and page.structure().approximate
    element = Subject.from_element(by_symbol("Fe"))
    page.set_subject(element)
    window.mix_in_flask(element)
    assert elements.focus is by_symbol("Fe")

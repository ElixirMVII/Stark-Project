from __future__ import annotations

import itertools
import math
import random
import statistics

import pytest

from aether.filters import OneEuroFilter, OneEuroFilter2D, smoothing_factor

DT = 1 / 30


def _run(filt: OneEuroFilter, values: list[float], dt: float = DT) -> list[float]:
    return [filt(v, i * dt) for i, v in enumerate(values)]


def test_smoothing_factor_matches_paper_formula():
    # alpha = 1 / (1 + tau/Te), tau = 1 / (2π fc)
    assert smoothing_factor(1.0, DT) == pytest.approx(1 / (1 + (1 / (2 * math.pi)) / DT))
    assert 0 < smoothing_factor(0.1, DT) < smoothing_factor(10.0, DT) < 1


def test_first_sample_passes_through():
    assert OneEuroFilter()(123.4, 0.0) == 123.4


def test_constant_signal_is_unchanged():
    out = _run(OneEuroFilter(), [5.0] * 50)
    assert out == pytest.approx([5.0] * 50)


def test_converges_to_step():
    out = _run(OneEuroFilter(min_cutoff=1.0, beta=0.0), [0.0] + [100.0] * 150)
    assert 0 < out[1] < 100  # ไม่กระโดดทันที
    assert out[-1] == pytest.approx(100.0, abs=0.5)
    assert all(b >= a for a, b in itertools.pairwise(out))  # ไม่ overshoot


def test_jitter_at_rest_is_reduced():
    """มืออยู่นิ่ง: noise ต้องถูกกดลงมาก (เคอร์เซอร์ไม่สั่น)"""
    rng = random.Random(7)
    noisy = [500.0 + rng.uniform(-4.0, 4.0) for _ in range(300)]
    out = _run(OneEuroFilter(min_cutoff=1.0, beta=0.02), noisy)
    assert statistics.pstdev(out[60:]) < 0.5 * statistics.pstdev(noisy[60:])
    assert statistics.fmean(out[60:]) == pytest.approx(500.0, abs=1.0)


def test_beta_reduces_lag_on_fast_motion():
    """ขยับเร็ว: beta ทำให้ตามทันกว่า (lag น้อยกว่า) เทียบกับ beta = 0"""
    ramp = [i * 40.0 for i in range(60)]  # 1200 px/s
    lag_plain = ramp[-1] - _run(OneEuroFilter(1.0, 0.0), ramp)[-1]
    lag_adaptive = ramp[-1] - _run(OneEuroFilter(1.0, 0.02), ramp)[-1]
    assert lag_adaptive < 0.25 * lag_plain
    assert lag_adaptive >= 0


def test_higher_min_cutoff_smooths_less():
    rng = random.Random(3)
    noisy = [rng.uniform(-1.0, 1.0) for _ in range(300)]
    smooth = statistics.pstdev(_run(OneEuroFilter(min_cutoff=0.5), noisy))
    loose = statistics.pstdev(_run(OneEuroFilter(min_cutoff=5.0), noisy))
    assert smooth < loose


def test_non_increasing_timestamp_returns_previous_value():
    filt = OneEuroFilter()
    filt(10.0, 1.0)
    second = filt(20.0, 1.1)
    assert filt(999.0, 1.1) == second  # timestamp ซ้ำ
    assert filt(999.0, 0.5) == second  # timestamp ถอยหลัง


def test_reset_forgets_history():
    filt = OneEuroFilter()
    _run(filt, [0.0] * 10)
    filt.reset()
    assert filt(77.0, 100.0) == 77.0


@pytest.mark.parametrize(
    "kwargs", [{"min_cutoff": 0}, {"d_cutoff": -1.0}, {"beta": -0.1}, {"min_cutoff": -2}]
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        OneEuroFilter(**kwargs)


def test_2d_filters_axes_independently():
    filt = OneEuroFilter2D(1.0, 0.0)
    assert filt(10.0, 20.0, 0.0) == (10.0, 20.0)
    x, y = filt(10.0, 120.0, DT)
    assert x == pytest.approx(10.0)
    assert 20.0 < y < 120.0
    filt.reset()
    assert filt(1.0, 2.0, 5.0) == (1.0, 2.0)

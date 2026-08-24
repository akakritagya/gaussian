import pytest

from gaussian import Gaussian


def test_solves_simple_system() -> None:
    ge = Gaussian(["x + 2y = 5", "3x - y = 4"])
    ge.solve()
    assert ge.solution == pytest.approx({"x": 13 / 7, "y": 11 / 7})


def test_solves_3x3_system() -> None:
    ge = Gaussian(["x + y + z = 6", "2x + 3y + z = 11", "x - y + 2z = 5"])
    ge.solve()
    assert ge.solution == pytest.approx({"x": 1.0, "y": 2.0, "z": 3.0})


def test_residual_near_zero_after_solve() -> None:
    ge = Gaussian(["x + y = 2", "x - y = 0"])
    ge.solve()
    assert ge.residual() == pytest.approx(0.0, abs=1e-9)


def test_residual_raises_before_solve() -> None:
    ge = Gaussian(["x + y = 2", "x - y = 0"])
    with pytest.raises(RuntimeError):
        ge.residual()


def test_non_square_system_raises() -> None:
    with pytest.raises(ValueError, match="requires a square system"):
        Gaussian(["x + y = 2"])

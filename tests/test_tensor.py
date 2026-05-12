from __future__ import annotations

import pytest

from tensorlite import Tensor


# ===================================================================================
#                              Testing Forward Pass
# ===================================================================================
def test_tensor_plus_tensor():
    a = Tensor(1.5)
    b = Tensor(3.5)

    assert (a + b).data == pytest.approx(5.0)


@pytest.fixture
def dummy_value():
    class NotATensor:
        def __init__(self, value):
            self.value = value

        def __repr__(self) -> str:
            return f"NotATensor({self.value})"

    return NotATensor(1.0)


def test_tensor_plus_not_tensor(dummy_value):
    a = Tensor(1.0)
    with pytest.raises(TypeError):
        a + dummy_value  # type: ignore


def test_tensor_plus_float():
    a = Tensor(1.0)
    assert (a + 1.0).data == pytest.approx(2.0)
    assert (2.0 + a).data == pytest.approx(3.0)


def test_tensor_plus_int():
    a = Tensor(1.0)
    assert (a + 1).data == pytest.approx(2.0)
    assert (2 + a).data == pytest.approx(3.0)


def test_tensor_times_tensor():
    a = Tensor(1.0)
    b = Tensor(2.0)
    assert (a * b).data == pytest.approx(2.0)


def test_tensor_times_not_tensor(dummy_value):
    a = Tensor(1.0)
    with pytest.raises(TypeError):
        a * dummy_value  # type: ignore


def test_tensor_times_float():
    a = Tensor(2.0)
    assert (a * 1.0).data == pytest.approx(2.0)
    assert (2.0 * a).data == pytest.approx(4.0)


def test_tensor_times_int():
    a = Tensor(2.0)
    assert (a * 1).data == pytest.approx(2.0)
    assert (2 * a).data == pytest.approx(4.0)


# ===================================================================================
#                              Testing Backward Pass
# ===================================================================================
def test_backward_addition():
    a = Tensor(1.0)
    b = Tensor(1.0)
    c = a + b
    c.backward()
    assert a.grad == pytest.approx(1.0)


def test_backward_multiple_addition():
    a = Tensor(1.0)
    b = Tensor(2.0)
    c = (a + b) + (a + b + b)
    c.backward()
    assert a.grad == pytest.approx(2.0)
    assert b.grad == pytest.approx(3.0)


def test_backward_multiplication():
    a = Tensor(1.0)
    b = Tensor(2.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(2.0)
    assert b.grad == pytest.approx(1.0)


def test_backward_complex_function():
    x = [Tensor(3.0), Tensor(2.5)]
    w = [Tensor(1.0), Tensor(3.5)]
    c = (x[0] * w[0]) + (x[1] * w[1])
    L = c * c
    L.backward()
    assert x[0].grad == pytest.approx(2 * c.data * w[0].data)
    assert w[0].grad == pytest.approx(2 * c.data * x[0].data)
    assert x[1].grad == pytest.approx(2 * c.data * w[1].data)
    assert w[1].grad == pytest.approx(2 * c.data * x[1].data)

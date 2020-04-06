import numpy as np
from skipi.function import Function

from ..helper import assert_equal


def test_complex_conjugate():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.exp(1j*x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: np.exp(-1j*x))

    assert_equal(f1.conj(), f2)

def test_absolute_value():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.exp(1j * x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: 1)

    f3 = Function(f1.get_domain(), lambda x: -x)
    f4 = Function(f1.get_domain(), lambda x: x)

    assert_equal(f1.abs(), f2)
    assert_equal(f3.abs(), f4)

def test_log():
    f1 = Function(np.linspace(-5, 5, 100), lambda x: np.exp(0.5*x))
    f2 = Function(np.linspace(-5, 5, 100), lambda x: 0.5*x)

    assert_equal(f1.log(), f2)

def test_log10():
    f1 = Function(np.linspace(-5, 5, 100), lambda x: 10**(0.3*x))
    f2 = Function(np.linspace(-5, 5, 100), lambda x: 0.3*x)

    assert_equal(f1.log10(), f2)
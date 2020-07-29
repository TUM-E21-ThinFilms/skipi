import numpy as np
from skipi.function import PiecewiseFunction, Function, StitchedFunction

from ..helper import assert_equal

def test_absolute_value():
    fl = Function(np.linspace(-10, 0, 50), lambda x: -x)
    fr = Function(np.linspace(0, 10, 100), lambda x: x)

    f = PiecewiseFunction.from_function(np.linspace(-10, 10, 200), fl, lambda x: x <= 0, fr)

    fabs = Function(np.linspace(-10, 10, 200), lambda x: abs(x))

    assert_equal(f, fabs)

def test_absolute_value_stiched():
    fl = Function(np.linspace(-10, 0, 50), lambda x: -x)
    fr = Function(np.linspace(0, 10, 100), lambda x: x)
    f = StitchedFunction.from_functions(fl, fr)

    fabs = Function(np.linspace(-10, 10, 200), lambda x: abs(x))

    assert_equal(f, fabs)

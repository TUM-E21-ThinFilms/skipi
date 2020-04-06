import numpy as np
from skipi.function import Function, Derivative
from ..helper import assert_equal

def test_scale_domain():
    x_domain = np.linspace(0, 1, 100)
    f = Function.to_function(x_domain, lambda x: x**2)
    f_ref = Function(np.linspace(0, 10, 100), lambda x: x**2/100)

    assert_equal(f.scale_domain(10), f_ref)
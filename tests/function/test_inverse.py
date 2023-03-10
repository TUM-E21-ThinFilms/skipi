import numpy as np
from skipi.domain import Domain

from skipi.function import Function, InverseFunction


def test_inverse():
    f = Function.to_function(Domain(-5, 5, 101), lambda x: x**2)
    fInv = InverseFunction.from_function(f, at=-0.1)
    fInv.plot(show=True)

test_inverse()
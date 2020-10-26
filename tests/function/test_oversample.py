import numpy as np

from skipi.function import Function
from skipi.domain import Domain

from ..helper import assert_equal


def test_oversample_does_not_change_function():
    f = Function(Domain(-10, 10, 100), lambda x: x ** 2)
    fexact = Function(Domain(-10, 10, 1000), lambda x: x ** 2)

    fnew = f.oversample(10)

    assert_equal(fexact, fnew)


def test_oversample_domain():
    f = Function(Domain.linear(-10, 10, 100), lambda x: x ** 2)
    #randint = np.random.randint(1, 100)
    randint=2
    fnew = f.oversample(randint)

    np.testing.assert_array_equal(fnew.get_domain(), Domain(-10, 10, randint * 100 + 1).get())

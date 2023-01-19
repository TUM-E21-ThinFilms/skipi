import numpy as np
from skipi.domain import Domain

from skipi.function import Function, ExtendableFunction
from ..helper import assert_equal

def test_extend_left():
    f1 = Function.to_function(Domain(0, 1, 100), lambda x: 10*x)
    f2 = Function.to_function(Domain(-1, 0, 100), lambda x: -10*x)

    f2e = ExtendableFunction.from_function(f2)

    f = f2e.extend(f1)
    f_true = Function.to_function(Domain(-1, 1, 201), lambda x: 10*x if x >= 0 else -10*x)

    #f_true = lambda x: abs(10*x)
    f.plot(show=True)
    #print(f.get_domain())
    #print(list(zip(f.get_domain(), f.eval())))
    assert_equal(f, f_true)
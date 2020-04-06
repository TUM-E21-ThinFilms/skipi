import numpy as np
from skipi.function import Function, Integral, UnevenlySpacedFunction

from ..helper import assert_equal, randspace

def test_integration():
    x_domain = np.linspace(0, 10, 100)
    f = Function(x_domain, lambda x: 6 * x)
    F = Integral.from_function(f)
    F2 = Function(x_domain, lambda x: 3 * x ** 2)

    assert_equal(F, F2)

def test_integration_random_domain():
    x_domain = randspace(0, 10, 1000)
    f = UnevenlySpacedFunction(x_domain, lambda x: 6 * x)
    F = Integral.from_function(f)
    F2 = UnevenlySpacedFunction(x_domain, lambda x: 3 * x ** 2)

    assert_equal(F, F2, TOL=1e-5)

def abstract_test_integration(domain):
    x_domain = domain
    f = Function(x_domain, lambda x: 6 * x)
    F = Integral.from_function(f)
    F2 = Function(x_domain, lambda x: 3 * x ** 2)

    assert_equal(F, F2)


def test_linearity():
    x_domain = np.linspace(0, 10, 1000)
    f1 = Function(x_domain, lambda x: 1)
    f2 = Function(x_domain, lambda x: 2 * x)
    f = f1 + f2
    F = Integral.from_function(f)
    f3 = Function(x_domain, lambda x: x + x ** 2)

    assert_equal(F, f3)

def test_intergation_bounds():
    x_domain = np.linspace(-0.7, 1, 1001)
    f = Function(x_domain, lambda x: 2*x+1)
    F = Integral.from_function(f)
    F2 = Function(x_domain, lambda x: x**2 +x - (x_domain[0]**2+x_domain[0]))
    assert_equal(F, F2)

    # Note here: since we're not starting the integration from exactly 0, but
    # from close to zero, we have to add this region to the integral via C ...
    f3 = f.vremesh((0, None))
    F3 = Integral.from_function(f.vremesh((0, None)), C=(f3.get_domain()[0]**2 + f3.get_domain()[0]))


    assert_equal(F3, Function(F3.get_domain(), lambda x: x**2+x))

    F = Integral.from_function(f, x0=0)
    F4 = Function(x_domain, lambda x: x ** 2+x)


    assert_equal(F, F4, TOL=1e-6)

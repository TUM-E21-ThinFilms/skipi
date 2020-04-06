from skipi.function import Function, Integral


def Lp(f: Function, p=2, domain=None):
    r"""
    Computes the Lp-norm of a function.

    The Lp-Norm of a function is defined as
    ::math..
        (\int |f(x)|^{p} dx)^(1/p)
    for any :math:`1 \leq p \le \infty` and any function f which p-th power is lebesgue integrable

    :param f:
    :param p:
    :param domain:
    :return:
    """
    assert 1 <= p

    if domain is None:
        domain = f.get_domain()

    return Integral.integrate((f.abs()) ** p) ** (1 / p)

def sup(f: Function, domain=None):
    if domain is None:
        domain = f.get_domain()

    return f(domain).max()
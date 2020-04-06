import numpy

from scipy.integrate import trapz

from skipi.function import Function, evaluate


class FourierTransform(Function):
    @classmethod
    def to_function(cls, domain, feval, frequency_domain, evenly_spaced=True):

        w = numpy.array(frequency_domain).reshape((len(frequency_domain), 1))
        feval = evaluate(domain, feval)

        t_domain = numpy.array(domain).reshape((1, len(domain)))

        integrand = feval * numpy.exp(- 1j * numpy.dot(w, t_domain))

        if evenly_spaced:
            F = trapz(integrand, dx=cls.get_dx(domain))
        else:
            F = trapz(integrand, x=domain)

        return Function.to_function(frequency_domain, F)

    @classmethod
    def from_function(cls, frequency_domain, fun: Function):
        return cls.to_function(fun.get_domain(), fun.get_function(), frequency_domain, fun.is_evenly_spaced())


class InverseFourierTransform(Function):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain, evenly_spaced=True):
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)
        integrand = feval * numpy.exp(1j * numpy.dot(w, domain))
        if evenly_spaced:
            F = 1 / (2 * numpy.pi) * trapz(integrand, dx=cls.get_dx(frequency_domain))
        else:
            F = 1 / (2 * numpy.pi) * trapz(integrand, x=frequency_domain)

        return Function.to_function(x_domain, F)

    @classmethod
    def from_function(cls, x_domain, fun: Function):
        return cls.to_function(fun.get_domain(), fun.get_function(), x_domain, fun.is_evenly_spaced())


class InverseCosineTransform(InverseFourierTransform):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain):
        dx = cls.get_dx(frequency_domain)
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)

        F = 1 / (numpy.pi) * trapz(feval * numpy.cos(numpy.dot(w, domain)), dx=dx)
        return Function.to_function(x_domain, F)


class CosineTransform(FourierTransform):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain):
        dx = cls.get_dx(frequency_domain)
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)

        F = trapz(feval * numpy.cos(numpy.dot(w, domain)), dx=dx)
        return Function.to_function(x_domain, F)


def fourier_matrix(t_space, f_space):
    # Important, otherwise t_space changes outside the function
    t_space = numpy.array(t_space)

    dt = t_space[1] - t_space[0]

    if dt == 0:
        raise RuntimeError("Given t_space has an incorrect format")

    f = numpy.array(f_space).reshape((len(f_space), 1))
    t = numpy.array(t_space).reshape((1, len(t_space)))

    f_t_matrix = numpy.dot(f, t)
    e_matrix = numpy.exp(-1j * f_t_matrix)

    # this is kinda the weighting of the trapezoidal integration rule
    e_matrix[:, 0] *= 0.5
    e_matrix[:, -1] *= 0.5

    return e_matrix * dt


def invfourier_matrix(f_space, t_space):
    r"""
    Returns a matrix representing a inverse fourier transform.

    Let :math:`R \colon f_space \to RR` (RR being the real numbers) be a function. The inverse fourier
    transform is then defined as
    ..math::
        F^{-1}[R](t) = \int_{RR}{e^{itf} R(f) df}

    This can be represented by a matrix multiplication. For this, assume you want to evaluate the inverse
    fourier transform :math:`F^-1[R] at t \in t_space` and you know the function R at :math:`f \in f_space`.
    Then this function returns a matrix A with the following properties
    ..math::
        F^{-1}[R] = A * R

    with * denoting the usual matrix-vector-product (i.e. numpy.dot) and the resulting vector will be
    evaluated exactly at :math:`t \in t_space`, i.e. :math:`(A*R)[i] = F^{-1}[R](t_space[i])`

    ..note::
        More details in the case of reflectometry: The resulting matrix A looks like the following
        with k (wavevector transfer) being f and x (depth) being t:

                k
            ----------▶
            |    ikx
          x |   e
            |
            ▼

        Thus :math:`A * R(k) = F[R](x) = V(x)`

    :Example:
    >>> x_space = numpy.linspace(0, 200, 200)
    >>> k_space = numpy.linspace(-0.1, 0.1, 400)
    >>> # R ... being the reflection evaluated at k_space, i.e. R = R_function(k_space)
    >>> A = invfourier_matrix(k_space, x_space)
    >>> V = numpy.dot(A, R) # V being evaluated at x_space, i.e. V = V_function(x_space)

    with R, V being the reflection and potential function, respectively.

    :param f_space: frequency space. The space where the function to do the inverse fourier transform is
    known. Has to be equidistantly spaced.
    :param t_space: time space. The space where the inverse fourier transform shall be evaluated.
    :return: A matrix representing the inverse fourier transform
    :raises:
        RuntimeError: If the spacing of f_space is zero, i.e. delta f_space = 0, df = 0.
    """

    # Important, otherwise f_space changes outside the function
    f_space = numpy.array(f_space)

    df = f_space[1] - f_space[0]
    if df == 0:
        raise RuntimeError("Given f_space has an incorrect format")

    f = numpy.array(f_space).reshape((1, len(f_space)))
    t = numpy.array(t_space).reshape((len(t_space), 1))

    t_f_matrix = numpy.dot(t, f)
    e_matrix = numpy.exp(1j * t_f_matrix)

    # this is kinda the weighting of the trapezoidal integration rule
    e_matrix[:, 0] *= 0.5
    e_matrix[:, -1] *= 0.5

    return 1 / (2 * numpy.pi) * e_matrix * df

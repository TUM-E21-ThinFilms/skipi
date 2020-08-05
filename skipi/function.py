import numpy
import scipy.interpolate

from typing import Callable

from scipy.integrate import trapz

from skipi.util import vslice
from skipi.domain import Domain

FUNCTION_INTERPOLATION_TYPE = 'linear'


class Function(object):
    """
    A mathematical function

    A function is in principle just a relation on a domain and the relation operation. Thus, every function
    here needs a domain (mesh/grid) together with a callable object (relation).

    Functions support the add, sub, mul, div and power operators:

    :Example:
    >>> f, g = Function(), Function
    >>> f + g, f + 3
    >>> f - g, f - 3
    >>> f * g, g * 3
    >>> f / g, f / 3
    >>> f ** g, f ** 3

    Composition is also possible:
    :Example:
    >>> f.apply(g) == g(f) # Use this if g is a build-in function, like abs
    >>> f.composeWith(g) == f(g)
    >>> g.composeWith(f) == g(f) # This is only possible if g is a Function

    Plotting is done
    :Example:
    >>> f.plot() # plots f on the whole domain (f.get_domain())
    >>> g.plot(domain, show=True) # plots g on domain
    """

    def __init__(self, domain, function_callable: Callable):
        """
        Creates a mathematical function based on the given domain and callable object.

        A function always needs a domain and a relation, i.e. f: X -> C
        with X being the domain, and C being the complex numbers.

        :Example:
        >>> f = Function(range(0, 10), lambda x: x**2)
        >>> g = Function(numpy.linspace(0, 10, 1000), lambda x: x**2)
        >>> h = Function(numpy.linspace(-10, 10, 200), abs)

        Function f and g have the same relation, however different domains. Function h is an example to use
        in-build function definitions.

        :param domain: list of points where the function is defined, equidistantly spaced!
        :param function_callable: callable function to evaluate this Function.
        """
        # if not self._is_evenly_spaced_domain(domain):
        #    raise RuntimeWarning("Given domain is not equidistantly spaced")

        if not isinstance(domain, numpy.ndarray):
            self._dom = numpy.array(domain)
        else:
            self._dom = domain

        if not callable(function_callable):
            raise RuntimeError("function must be callable")

        if isinstance(function_callable, Function):
            function_callable = function_callable.get_function()

        self._f = function_callable

    @classmethod
    def _is_evenly_spaced_domain(cls, domain):
        """
        Checks whether the given domain (list) is evenly (equdistantly) spaced.

        :param domain: numpy.array. domain to check
        :return: boolean
        """

        diff = numpy.diff(domain)

        if numpy.all(numpy.isclose(diff - diff[0], numpy.zeros(len(diff)))):
            return True

        return False

    def is_complex(self):
        return numpy.any(numpy.iscomplex(self.eval()))

    def is_evenly_spaced(self):
        return True

    def copy(self):
        """
        Copies and returns the copied function
        :return:
        """
        return Function(self._dom, self._f)

    def transform(self, transformation: Callable[[complex, complex], complex]):
        """
        Transforms the function f based on the given transformation and returns a new Function F via:

            F.domain = f.domain
            F(x) = transformation(x, f(x)) for x in f.domain

        The transformation has to accept two parameters: x and f(x)

        :Example:
        >>> # take the square of a function
        >>> transformation = lambda x, fx: fx**2
        >>> # ignoring the previous function and just return a straight line with slope 1.
        >>> transformation = lambda x, fx: x
        >>> # Scaling by x**2
        >>> transformation = lambda x, fx: x**2 * fx

        :param transformation: callable
        :return: Function
        """

        if not callable(transformation):
            raise RuntimeError("Transformation has to be callable")

        if not transformation.__code__.co_argcount == 2:
            raise RuntimeError("Transformation has to accept two parameters: x and f(x)")

        return Function.to_function(self._dom,
                                    [transformation(x, fx) for (x, fx) in zip(self._dom, self.eval())])

    def reinterpolate(self, interpolation_kind=None):
        """
        Uses the internal callable function, to interpolate it on the given domain.

        Useful after applying different functions to it, to increase the performance.

        :return:
        """
        return Function(self._dom, to_function(self._dom, self._f, interpolation=interpolation_kind))

    def shift(self, offset, domain=False):
        """
        Shifts the function to the right by offset.

        If domain is True, it additionally shifts the domain.

        :param offset:
        :param domain:
        :return:
        """
        dom = self._dom
        if domain is True:
            dom = self._dom + offset

        f = self._f
        return Function(dom, lambda x: f(x - offset))

    def scale_domain(self, factor):
        dom = factor * self._dom
        f = self._f
        return Function(dom, lambda x: f(x / factor))

    def apply(self, function: Callable):
        """
        Applies a function to Function. (Composition).

        In mathematical terms, let g be function, and f being the called Function. Then this method computes
        f.apply(g)(x) = g(f(x))

        :Example:
        >>> f = Function()
        >>> g = lambda x...
        >>> f.apply(g) # g(f(x))

        :param function: Callable function
        :return:
        """

        f = self._f
        return Function(self._dom, lambda x: function(f(x)))

    def composeWith(self, function: Callable):
        """
        Composition of two functions, similar to apply. However, the composition is the other way round.

        In mathematical terms, let g be function, and f being the called Function. Then this method computes
        f.composeWith(g) = f(g(x))

        :Example:
        >>> f = Function()
        >>> g = lambda x:
        >>> f.composeWith(g) # f(g(x))
        :param function:
        :return:
        """

        f = self._f
        return Function(self._dom, lambda x: f(function(x)))

    def conj(self):
        """
        Computes the complex conjugate and returns it.
        :return:
        """
        return self.apply(numpy.conj)

    def abs(self):
        """
        Computes the absolute value and returns it.
        :return:
        """
        return self.apply(abs)

    def log(self):
        """
        Computes the natural logarithm and returns it.
        :return:
        """
        return self.apply(numpy.log)

    def log10(self):
        """
        Computes the logarithm (base 10) and returns it.
        :return:
        """
        return self.apply(numpy.log10)

    def max(self):
        """
        Computes the maximum value and returns it.
        :return:
        """
        return numpy.max(self.eval())

    def min(self):
        """
        Computes the minimum value and returns it.
        :return:
        """
        return numpy.min(self.eval())

    def argmax(self):
        """
        Computes the argument which attains the maximum value
        :return:
        """
        return self.get_domain()[numpy.argmax(self.eval())]

    def argmin(self):
        """
        Computes the argument which attains the minimum value
        :return:
        """
        return self.get_domain()[numpy.argmin(self.eval())]

    def get_domain(self):
        return self._dom

    def eval(self):
        return self(self.get_domain())

    @classmethod
    def get_dx(cls, domain):
        if len(domain) < 2:
            return 0

        # Assuming equidistantly spaced domain
        return domain[1] - domain[0]

    def get_function(self):
        return self._f

    def __call__(self, x):
        return self._f(x)

    @classmethod
    def to_function(cls, domain, feval, **kwargs):
        return cls(domain, to_function(domain, feval, **kwargs))

    def remesh(self, new_mesh, reevaluate=False, **kwargs):
        """
        Remeshes the function using the new_mesh

        Note that this will only change the domain, nothing else will change (the callable function
        is preserved)

        :param new_mesh: The new mesh (i.e. linspace from numpy)
        :param reevaluate: If True, the function will be evaluated on the new mesh, and interpolated (using
        the default interpolation kind). kwargs will be directly passed to to_function (to change the
        interpolation kind).
        :return:
        """
        if reevaluate:
            return Function(new_mesh, to_function(new_mesh, self._f(new_mesh), **kwargs))

        return Function(new_mesh, self._f)

    def oversample(self, n):
        """
        Oversamples/Interpolates the function on a equidistant grid.
        The number of grid points is determined by the old grid times n.

        :param n: Grid-point factor
        :return:
        """
        if n <= 0:
            raise RuntimeError("The oversampling-factor n has to be a positive integer")

        new_mesh = numpy.linspace(self._dom.min(), self._dom.max(), int(n) * len(self._dom) + 1)
        return self.remesh(new_mesh)

    def vremesh(self, *selectors, dstart=0, dstop=0):
        """
        Remeshes the grid/domain using vslice.

        Particularly useful if you want to restrict you function

        :Example:
        >>> f.vremesh((None, None)) # does nothing in principle
        >>> f.vremesh((0, None)) # remeshes from 0 to the end of domain
        >>> f.vremesh((None, 0)) # remeshes from the start of the domain till 0

        >>> f = Function(np.linspace(-1, 1, 100), numpy.sin)
        >>> g = f.vremesh((-0.1, 0.1)) # == Function(np.linspace(-0.1, 0.1, 10), numpy.sin)

        >>> h = f.vremesh((-1.0, -0.1), (0.1, 1.0)) # remeshes the function on ([-1, -0.1] union [0.1, 1.0])
        >>> f == g + h

        :param selectors:
        :param dstart:
        :param dstop:
        :return:
        """

        return self.remesh(vslice(self.get_domain(), *selectors, dstart=dstart, dstop=dstop))

    @classmethod
    def from_function(cls, fun: 'Function'):
        return cls.to_function(fun.get_domain(), fun.get_function())

    @staticmethod
    def _is_number(other):
        return (isinstance(other, int) or
                isinstance(other, float) or
                isinstance(other, numpy.complex) or
                isinstance(other, numpy.float) or
                (isinstance(other, numpy.ndarray) and other.size == 1))

    @staticmethod
    def _unknown_type(other):
        raise RuntimeError("Unknown type of other")

    def __add__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) + other.get_function()(x))
        if callable(other):
            return Function(self._dom, lambda x: self._f(x) + other(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) + other)

        self._unknown_type(other)

    def __sub__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) - other.get_function()(x))
        if callable(other):
            return Function(self._dom, lambda x: self._f(x) - other(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) - other)

        self._unknown_type(other)

    def __pow__(self, power):
        if isinstance(power, Function):
            return Function(self._dom, lambda x: self._f(x) ** power.get_function()(x))
        if callable(power):
            return Function(self._dom, lambda x: self._f(x) ** power(x))
        if self._is_number(power):
            return Function(self._dom, lambda x: self._f(x) ** power)

        self._unknown_type(power)

    def __mul__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) * other.get_function()(x))
        if callable(other):
            return Function(self._dom, lambda x: self._f(x) * other(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) * other)

        self._unknown_type(other)

    def __truediv__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) / other.get_function()(x))
        if callable(other):
            return Function(self._dom, lambda x: self._f(x) / other(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) / other)

        self._unknown_type(other)

    def __neg__(self):
        f = self._f
        return Function(self._dom, lambda x: -f(x))

    def plot(self, plot_space=None, show=False, real=True, **kwargs):
        import pylab
        if plot_space is None:
            plot_space = self.get_domain()

        feval = self._f(plot_space)

        lbl_re = {}
        lbl_im = {}

        try:
            lbl = kwargs.pop("label")
            if not lbl is None:
                lbl_re["label"] = lbl
                if not real:
                    lbl_re["label"] = lbl + ' (Re)'
                    lbl_im["label"] = lbl + ' (Im)'

        except KeyError:
            lbl = None

        pylab.plot(plot_space, feval.real, **kwargs, **lbl_re)

        if not real:
            pylab.plot(plot_space, feval.imag, **kwargs, **lbl_im)

        if not lbl is None:
            pylab.legend()

        if show:
            pylab.show()

    def show(self):
        self.plot(show=True)

    @property
    def real(self):
        return Function(self._dom, lambda x: self._f(x).real)

    @property
    def imag(self):
        return Function(self._dom, lambda x: self._f(x).imag)

    def real_imag(self):
        return self.real, self.imag

    def find_zeros(self):
        f0 = self._f(self._dom[0])
        roots = []
        for el in self._dom:
            fn = self._f(el)
            if (f0.real * fn.real) < 0:
                # there was a change in sign.
                roots.append(el)
                f0 = fn
        return roots


class NullFunction(Function):
    def __init__(self, domain):
        super(NullFunction, self).__init__(domain, lambda x: 0)


class ComplexFunction(Function):
    @classmethod
    def to_function(cls, domain, real_part, imaginary_part, **kwargs):
        return Function.to_function(domain, real_part + 1j * imaginary_part, **kwargs)

    @classmethod
    def from_function(cls, real_part: Function, imaginary_part: Function):
        return real_part + imaginary_part * 1j


class UnevenlySpacedFunction(Function):
    @classmethod
    def _is_evenly_spaced_domain(cls, domain):
        return True

    def is_evenly_spaced(self):
        return False

    def get_dx(self, domain):
        return numpy.diff(domain)


class Integral(Function):
    @classmethod
    def to_function(cls, domain, feval, C=0, evenly_spaced=True, **kwargs):
        r"""
        Returns the integral function starting from the first element of domain, i.e.
        ::math..
            F(x) = \int_{x0}^{x} f(z) dz + C

        where x0 = domain[0] and f is the given function (feval).
        :param domain:
        :param feval:
        :param C: integral constant (can be arbitrary)
        :param evenly_spaced: Whether the domain is evenly spaced or not
        :return:
        """
        dx = cls.get_dx(domain)
        if evenly_spaced:
            Feval = scipy.integrate.cumtrapz(y=evaluate(domain, feval), dx=dx, initial=0) + C
            return Function.to_function(domain, Feval, **kwargs)
        else:
            Feval = scipy.integrate.cumtrapz(y=evaluate(domain, feval), x=domain, initial=0) + C
            return UnevenlySpacedFunction.to_function(domain, Feval, **kwargs)

    @classmethod
    def from_function(cls, fun: Function, x0=None, C=0):
        if x0 is None:
            return cls.to_function(fun.get_domain(), fun, C=C, evenly_spaced=fun.is_evenly_spaced())
        else:
            F = cls.from_function(fun)
            return F - F(x0)

    @classmethod
    def integrate(cls, fun: Function, x0=None, x1=None):
        r"""
        Calculates the definite integral of the Function fun.

        If x0 or x1 are given, the function is re-meshed at these points, and thus this function returns
        ::math..
            \int_{x_0}^{x_1} f(x) dx

        If x0 and x1 are both None, the integral is evaluated over the whole domain
        x0, x1 = domain[0], domain[-1], i.e.
        ::math..
            \int_{x_0}^{x_1} f(x) dx = \int f(x) dx

        :param fun:
        :param x0: lower bound of the integral limit or None
        :param x1: upper bound of the integral limit or None
        :return: definite integral value (Not a function!)
        """
        if not any(numpy.array([x0, x1]) is None):
            fun = fun.vremesh((x0, x1))

        dx = cls.get_dx(fun.get_domain())
        if fun.is_evenly_spaced():
            return scipy.integrate.trapz(fun.eval(), dx=dx)
        else:
            return scipy.integrate.trapz(fun.eval(), x=fun.get_domain())


# Just renaming
class Antiderivative(Integral):
    pass


class Derivative(Function):
    @classmethod
    def to_function(cls, domain, feval, **kwargs):
        feval = evaluate(domain, feval)
        fprime = numpy.gradient(feval, cls.get_dx(domain), edge_order=2)
        return Function.to_function(domain, fprime, **kwargs)

    @classmethod
    def from_function(cls, fun: Function):
        return cls.to_function(fun.get_domain(), fun)


class PiecewiseFunction(Function):
    @classmethod
    def from_function(cls, domain, f: Function, conditional: Callable[..., bool], f_otherwise: Function):
        def feval(x):
            conds = conditional(x)
            nconds = numpy.logical_not(conds)

            fe = numpy.zeros(x.shape, dtype=complex)
            fe[conds] = f(x[conds])
            fe[nconds] = f_otherwise(x[nconds])
            return fe

        return Function(domain, feval)


class StitchedFunction(Function):
    @classmethod
    def from_functions(cls, left: Function, right: Function, grid=None):

        if grid is None:
            grid = Domain.coarse_grid

        if callable(grid):
            grid = grid([left.get_domain(), right.get_domain()])

        right_domain = min(right.get_domain())

        conditional = lambda x: x < right_domain

        return PiecewiseFunction.from_function(grid, left, conditional, right)


def evaluate(domain, function):
    """
    Evaluates a function on its domain.

    If function is callable, it's simply evaluated using the callable
    If function is a numpy array, then its simply returned (assuming it was already evaluated elsewhere)


    :param domain: numpy.array
    :param function: callable/np.array
    :raise RuntimeError: Unknown type of function given
    :return:
    """
    if callable(function):
        return numpy.array([function(x) for x in domain])
    elif isinstance(function, numpy.ndarray) and len(domain) == len(function):
        return function
    elif isinstance(function, list) and len(domain) == len(function):
        return numpy.array(function)
    else:
        raise RuntimeError("Cannot evaluate, unknown type")


def set_interpolation_type(interpolation_type):
    """
    Sets the interpolation type used for all Functions

    :param interpolation_type: "linear", "cubic", "quadratic", etc.. see scipy.interpolation.interp1d
    :return: previous interpolation type
    """
    global FUNCTION_INTERPOLATION_TYPE
    previous_type = FUNCTION_INTERPOLATION_TYPE
    FUNCTION_INTERPOLATION_TYPE = interpolation_type
    return previous_type


def to_function(x_space, feval, interpolation=None, to_zero=True):
    """
    Returns an interpolated function using x and f(x).

    :param x_space: domain of the function
    :param feval: evaluated function at f(x) for each x/ or callable function
    :param interpolation: Type of interpolation, see scipy.interp1d
    :param to_zero: the returned function will evaluate to zero (or nan) outside the domain
    :return: Callable function
    """
    if interpolation is None:
        global FUNCTION_INTERPOLATION_TYPE
        interpolation = FUNCTION_INTERPOLATION_TYPE

    if callable(feval):
        feval = numpy.array([feval(x) for x in x_space])

    if len(x_space) == 0:
        return lambda x: 0

    feval = numpy.array(feval)

    if to_zero:
        fill = (0, 0)
    else:
        fill = numpy.nan

    real = scipy.interpolate.interp1d(x_space, feval.real, fill_value=fill, bounds_error=False,
                                      kind=interpolation)

    if numpy.any(numpy.iscomplex(feval)):
        imag = scipy.interpolate.interp1d(x_space, feval.imag, fill_value=fill, bounds_error=False,
                                          kind=interpolation)

        return lambda x: real(x) + 1j * imag(x)

    return real


class FunctionFileLoader:
    """
    Simple class to write a function to disk and read a function from disk.

    Uses the numpy.savetxt/loadtxt methods.
    """

    def __init__(self, file):
        self._file = file

    def exists(self):
        from os import path

        return path.exists(self._file)

    def from_file(self):
        """
        Loads a function from file and returns its object.

        This can read files of the row-form:
            - x f(x).real f(x).imag
            - x f(x).real

        :return: Function
        """

        data = numpy.loadtxt(self._file)
        try:
            x, freal, fimag = data.T
        except:
            try:
                x, freal = data.T
                fimag = None
            except:
                raise RuntimeError("Unknown function file type")

        if fimag is None:
            return Function.to_function(x, freal)
        else:
            feval = freal + 1j * fimag
            return Function.to_function(x, feval)

    def to_file(self, function: Function, header=''):
        """
        Saves a given function to disk.

        It will save the file in such a way that it is readable by from_file.

        :param function: Function to save to disk
        :param header: Header string attached at the beginning of the file, will be added as a comment
        :return: None
        """
        domain = function.get_domain()
        feval = function.eval()

        if function.is_complex():
            data = numpy.array([domain, feval.real, feval.imag])
        else:
            data = numpy.array([domain, feval])

        numpy.savetxt(self._file, data.T, header=header)


class AutomorphDecorator(object):
    """
    Use this class to create a function object which changes when you apply specific methods.

    Usually, a function object is kept immutable, i.e. calling transform/apply just returns a new function
    and the old function is not changing.

    To avoid this behaviour (i.e. the function is actually changing) you can use this decorator. So calling
    i.e. transform/apply does change the function.

    We get this behaviour by this proxy class. This class keeps a reference to a function object and every
    time a new function is created, the reference is updated. From the outside, it looks like it's changing.

    Note that this class does not act like a Function class, it's just forwarding method calls to the
    internal function. Thus, use this class only in special cases and avoid it when possible.
    """
    def __init__(self, f: Function):
        self._f = f
        # Methods that change the internal function
        self.morph_methods = ['reparametrize', 'transform', 'reinterpolate', 'shift', 'scale_domain', 'apply',
                           'composeWith', 'vremesh', 'oversample', 'remesh', 'log10', 'log', 'abs', 'conj']

    def __getattr__(self, method):
        if method in self.morph_methods:
            def wrapped(*args, **kwargs):
                ret = getattr(self._f, method)(*args, **kwargs)
                if isinstance(ret, Function):
                    self._f = ret
                return ret

            return wrapped

        return getattr(self._f, method)
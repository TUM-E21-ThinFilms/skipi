import numpy
import scipy.interpolate

from typing import Callable

from scipy.integrate import trapz

from skipi.util import is_number
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

    Plotting is done via

    :Example:
    >>> f.plot() # plots f on the whole domain (f.get_domain())
    >>> g.plot(domain, show=True) # plots g on domain
    """

    def __init__(self, domain: Domain, fun_callable: Callable, dy: 'Function' = None, dx: 'Function' = None):
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
        :param fun_callable: callable function to evaluate this Function.
        """

        if isinstance(domain, Domain):
            self._domain = domain
        else:
            self._domain = Domain.from_domain(domain)

        if not callable(fun_callable):
            raise RuntimeError("function must be callable")

        if isinstance(fun_callable, Function):
            fun_callable = fun_callable.get_function()

        self._f = fun_callable
        self._dy = dy
        self._dx = dx

    def __reduce__(self):
        dy = self._dy.eval() if self._dy is not None else None
        dx = self._dx.eval() if self._dx is not None else None
        args = (self._domain, self.eval(), dy, dx)
        return (Function.__unpickle__, args)

    @staticmethod
    def __unpickle__(dom, f, dy, dx):
        f = Function.to_function(dom, f)
        dy = Function.to_function(dom, dy) if dy is not None else None
        dx = Function.to_function(dom, dx) if dx is not None else None
        return Function(dom, f, dy, dx)

    def is_complex(self):
        return numpy.any(numpy.iscomplex(self.eval()))

    def is_evenly_spaced(self):
        return True

    def copy(self):
        """
        Copies and returns the copied function
        :return:
        """
        return self.__class__(self._domain, self._f)

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

        return self.__class__.to_function(self._domain,
                                          [transformation(x, fx) for (x, fx) in
                                           zip(self.get_domain(), self.eval())])

    def reinterpolate(self, interpolation_kind=None):
        """
        Uses the internal callable function, to interpolate it on the given domain.

        Useful after applying different functions to it, to increase the performance.

        :return:
        """

        dx = self._dx.reinterpolate(interpolation_kind) if self._dx else None
        dy = self._dy.reinterpolate(interpolation_kind) if self._dy else None

        return self.__class__(self._domain,
                              to_function(self.get_domain(), self._f, interpolation=interpolation_kind),
                              dx=dx, dy=dy)

    def shift(self, offset, domain=False):
        """
        Shifts the function to the right by offset.

        If domain is True, it additionally shifts the domain.

        :param offset:
        :param domain:
        :return:
        """
        dom = self._domain
        if domain is True:
            dom = self._domain.shift(offset)

        dx = self._dx.shift(offset, domain) if self._dx else None
        dy = self._dy.shift(offset, domain) if self._dy else None

        f = self._f
        return self.__class__(dom, lambda x: f(x - offset), dx=dx, dy=dy)

    def scale_domain(self, factor):
        """
        Scales the domain by the factor (and the function accordingly)

        :param factor: Scaling factor, should not be zero
        :return:
        """
        f = self._f

        dx = self._dx.scale_domain(factor) if self._dx else None
        dy = self._dy.scale_domain(factor) if self._dy else None

        return self.__class__(self._domain.scale(factor), lambda x: f(x / factor), dx=dx, dy=dy)

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
        return self.__class__(self._domain, lambda x: function(f(x)))

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
        return self.__class__(self._domain, lambda x: f(function(x)))

    def flip(self, x0=None):
        """
        "Flips"/"mirrors" the function the function at x0.

        If x0 is None, then it will be chosen such that the most right data point in the domain
        will be mapped to the most left x point (and vice versa).

        The domain will not be adjusted by default!

        :param x0: mirror axis
        :return:
        """
        if x0 is None:
            x0 = self._domain.min() + self._domain.max()

        dx = self._dx.flip(x0) if self._dx else None
        dy = self._dy.flip(x0) if self._dy else None

        f = self._f
        return self.__class__(self._domain, lambda x: f(x0 - x), dx=dx, dy=dy)

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
        return self.apply(numpy.abs)

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
        return self._domain.get()

    def get_dom(self):
        return self._domain

    def eval(self):
        return self(self.get_domain())

    @classmethod
    def get_dx(cls, domain):
        """
        Returns the spacing in the domain, called dx.
        Not to be mixed up with dx being the error in the x variable.

        TODO: Should be moved to Domain-class
        :param domain:
        :return:
        """
        if len(domain) < 2:
            return 0

        return Domain.get_dx(domain)

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
        dx, dy = None, None

        if self._dx is not None:
            dx = self._dx.remesh(new_mesh, reevaluate=reevaluate, **kwargs)
        if self._dy is not None:
            dy = self._dy.remesh(new_mesh, reevaluate=reevaluate, **kwargs)

        if reevaluate:
            return self.__class__(new_mesh, to_function(new_mesh, self._f, **kwargs), dy=dy, dx=dx)

        return self.__class__(new_mesh, self._f, dy=dy, dx=dx)

    def oversample(self, n):
        """
        Oversamples/Interpolates the function on a equidistant grid.
        The number of grid points is determined by the old grid times n.

        :param n: Grid-point factor
        :return:
        """
        if n <= 0:
            raise RuntimeError("The oversampling-factor n has to be a positive integer")

        if n == 1:
            return self

        return self.remesh(self._domain.oversample(n))

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
        return self.remesh(self._domain.vremesh(*selectors, dstart=dstart, dstop=dstop))

    @classmethod
    def from_function(cls, fun: 'Function'):
        return cls.to_function(fun.get_dom(), fun.get_function())

    @staticmethod
    def _unknown_type(other):
        raise RuntimeError("Unknown type of other")

    def __add__(self, other):
        if isinstance(other, Function):
            return self.__class__(self._domain, lambda x: self._f(x) + other.get_function()(x))
        if callable(other):
            return self.__class__(self._domain, lambda x: self._f(x) + other(x))
        if is_number(other):
            return self.__class__(self._domain, lambda x: self._f(x) + other)

        self._unknown_type(other)

    def __sub__(self, other):
        if isinstance(other, Function):
            return self.__class__(self._domain, lambda x: self._f(x) - other.get_function()(x))
        if callable(other):
            return self.__class__(self._domain, lambda x: self._f(x) - other(x))
        if is_number(other):
            return self.__class__(self._domain, lambda x: self._f(x) - other)

        self._unknown_type(other)

    def __pow__(self, power):
        if isinstance(power, Function):
            return self.__class__(self._domain, lambda x: self._f(x) ** power.get_function()(x))
        if callable(power):
            return self.__class__(self._domain, lambda x: self._f(x) ** power(x))
        if is_number(power):
            return self.__class__(self._domain, lambda x: self._f(x) ** power)

        self._unknown_type(power)

    def __mul__(self, other):
        if isinstance(other, Function):
            return self.__class__(self._domain, lambda x: self._f(x) * other.get_function()(x))
        if callable(other):
            return self.__class__(self._domain, lambda x: self._f(x) * other(x))
        if is_number(other):
            return self.__class__(self._domain, lambda x: self._f(x) * other)

        self._unknown_type(other)

    def __truediv__(self, other):
        if isinstance(other, Function):
            return self.__class__(self._domain, lambda x: self._f(x) / other.get_function()(x))
        if callable(other):
            return self.__class__(self._domain, lambda x: self._f(x) / other(x))
        if is_number(other):
            return self.__class__(self._domain, lambda x: self._f(x) / other)

        self._unknown_type(other)

    def __neg__(self):
        f = self._f
        return self.__class__(self._domain, lambda x: -f(x))

    def plot(self, plot_space=None, show=False, real=True, **kwargs):
        import pylab
        if plot_space is None:
            plot_space = self.get_domain()

        plot_function = pylab.plot
        feval = self._f(plot_space)
        dfeval = self._dy.eval() if self._dy is not None else None

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

        if dfeval is not None:
            plot_function = pylab.errorbar
            kwargs['yerr'] = dfeval.real

        plot_function(plot_space, feval.real, **kwargs, **lbl_re)

        if not real:
            if dfeval is not None:
                kwargs['yerr'] = dfeval.imag
            plot_function(plot_space, feval.imag, **kwargs, **lbl_im)

        if not lbl is None:
            pylab.legend()

        if show:
            pylab.show()

    def show(self):
        self.plot(show=True)

    @property
    def real(self):
        return self.__class__(self._domain, lambda x: self._f(x).real)

    @property
    def imag(self):
        return self.__class__(self._domain, lambda x: self._f(x).imag)

    @property
    def dy(self):
        return self._dy

    @property
    def dx(self):
        return self._dx

    @dy.setter
    def dy(self, dy):
        assert isinstance(dy, Function) or dy is None
        self._dy = dy

    @dx.setter
    def dx(self, dx):
        assert isinstance(dx, Function) or dx is None
        self._dx = dx

    def set_dy(self, dy: 'Function'):
        self._dy = dy

    def set_dx(self, dx: 'Function'):
        self._dx = dx

    def real_imag(self):
        return self.real, self.imag

    def lp_metric(self, p=2):
        return numpy.sum(numpy.power(self.eval(), p))


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


class Integral(Function):
    @classmethod
    def to_function(cls, domain, feval, C=0, **kwargs):
        r"""
        Returns the integral function starting from the first element of domain, i.e.
        ::math..
            F(x) = \int_{x0}^{x} f(z) dz + C

        where x0 = domain[0] and f is the given function (feval).
        :param domain:
        :param feval:
        :param C: integral constant (can be arbitrary)
        :return:
        """
        dx = Domain.get_dx(domain)

        Feval = scipy.integrate.cumtrapz(y=evaluate(domain, feval), dx=dx, initial=0) + C
        return Function.to_function(domain, Feval, **kwargs)

    @classmethod
    def from_function(cls, fun: Function, x0=None, C=0):
        if x0 is None:
            return cls.to_function(fun.get_domain(), fun, C=C)
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

        dx = Domain.get_dx(fun.get_dom())
        return scipy.integrate.trapz(fun.eval(), dx=dx)


# Just renaming
class Antiderivative(Integral):
    pass


class Derivative(Function):
    @classmethod
    def to_function(cls, domain, feval, **kwargs):
        feval = evaluate(domain, feval)
        fprime = numpy.gradient(feval, Domain.get_dx(domain), edge_order=2)
        return Function.to_function(domain, fprime, **kwargs)

    @classmethod
    def from_function(cls, fun: Function):
        return cls.to_function(fun.get_dom(), fun)


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
        domain = Domain.from_domains([left.get_dom(), right.get_dom()], grid)

        right_domain = right.get_dom().min()

        conditional = lambda x: x < right_domain

        return PiecewiseFunction.from_function(domain, left, conditional, right)


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
    if isinstance(domain, Domain):
        domain = domain.get()

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

    x_space = Domain.as_array(x_space)

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

    def __init__(self, file, prefix=''):
        import os
        self._file = os.path.join(prefix, file)

    def exists(self):
        from os import path

        return path.exists(self._file)

    def from_file(self, has_errors=False):
        """
        Loads a function from file and returns its object.

        This can read files of the row-form:
            - x f(x).real f(x).imag
            - X f(x).real df(x).real f(x).imag df(x).imag
            - x f(x).real
            - x f(x).real df(x).real

        :return: Function
        """

        data = numpy.loadtxt(self._file)

        l = len(data.T)

        if l == 2:
            x, freal = data.T
            return Function.to_function(x, freal)
        elif l == 3:
            if has_errors:
                x, f, df = data.T
                f = Function.to_function(x, f)
                f.set_dy(Function.to_function(x, df))
                return f
            else:
                x, freal, fimag = data.T
                return ComplexFunction.to_function(x, freal, fimag)

        elif l == 5:
            x, freal, dfreal, fimag, dfimag = data.T
            f = ComplexFunction.to_function(x, freal, fimag)
            f.set_dy(ComplexFunction.to_function(x, dfreal, dfimag))
            return f

        raise RuntimeError("Unknown function file type")

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
        dy = function.dy

        if function.is_complex():
            if dy is not None:
                dyeval = dy.eval()
                data = numpy.array([domain, feval.real, dyeval.real, feval.imag, dyeval.imag])
                pre_header = ["x", "Re f(x)", "Re df(x)", "Im f(x)", "Im df(x)"]
            else:
                data = numpy.array([domain, feval.real, feval.imag])
                pre_header = ["x", "Re f(x)", "Im f(x)"]
        else:
            if dy is not None:
                pre_header = ["x", "f(x)", "df(x)"]
                data = numpy.array([domain, feval, dy.eval()])
            else:
                pre_header = ["x", "f(x)"]
                data = numpy.array([domain, feval])

        header = "\t".join(pre_header) + "\n" + header

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

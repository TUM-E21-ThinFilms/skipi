from skipi.function import Function, NullFunction


class ErrorFunction(object):
    def __init__(self, f: Function, x_error: Function = None, y_error: Function = None):

        self._f = f

        # error in x and y
        domain = f.get_domain()
        if x_error is None:
            x_error = NullFunction(domain)
        if y_error is None:
            y_error = NullFunction(domain)

        self._ex = x_error
        self._ey = y_error

    @property
    def dx(self):
        return self._ex

    @property
    def dy(self):
        return self._ey

    def to_function(cls, domain, feval, x_err=None, y_err=None):
        f = Function.to_function(domain, feval)

        if x_err is not None:
            x_err = Function.to_function(domain, x_err)
        if y_err is not None:
            y_err = Function.to_function(domain, y_err)

        return cls(f, x_err, y_err)

    def from_function(cls, fun: Function):
        return cls(Function.from_function(fun), None, None)

    def __call__(self, domain):
        return self._f(domain)

    @property
    def real(self):
        return ErrorFunction(self._f.real, self._ex.real, self._ey.real)

    @property
    def imag(self):
        return ErrorFunction(self._f.imag, self._ex.imag, self._ey.imag)

    def __add__(self, other):
        return ErrorFunction(self._f + other, self.dx(), self.dy())

    def __sub__(self, other):
        return ErrorFunction(self._f - other, self.dx(), self.dy())

    def __mul__(self, other):
        return ErrorFunction(self._f * other, self.dx(), self.dy() * other)

    def __truediv__(self, other):
        return ErrorFunction(self._f / other, self.dx(), self.dy() / other)

    def __pow__(self, power):
        return ErrorFunction(self._f ** power, self.dx(), self.dy() ** power)

    def plot(self, plot_space=None, show=False, x_error=True, y_error=True, **kwargs):
        import pylab

        if plot_space is None:
            plot_space = self._f.get_domain()

        feval = self._f(plot_space)
        xerr, yerr = None, None
        if not isinstance(self._ex, NullFunction) and x_error:
            xerr = self.dx(plot_space)
        if not isinstance(self._ey, NullFunction) and y_error:
            yerr = self.dy(plot_space)

        pylab.errorbar(plot_space, feval, yerr=yerr, xerr=xerr, **kwargs)

        if show is True:
            pylab.show()

    def show(self):
        self.plot(show=True)
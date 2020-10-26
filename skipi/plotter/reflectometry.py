from skipi.plot import FunctionPlotter
from skipi.function import Function, FunctionFileLoader

class ReflectivityPlotter(FunctionPlotter):
    def plot(self, name, f: Function, **kwargs):

        if 'label' not in kwargs:
            kwargs['label'] = name

        self.axs.set_yscale("log")
        self.axs.set_xlabel("q [$\AA^{-1}$]")
        self.axs.set_ylabel("log Intensity [1]")
        self.axs.legend()

        super(ReflectivityPlotter, self).plot(name, f, label=name)

        return self


class SLDPlotter(FunctionPlotter):
    def _default_args(self):
        self._ylabel = "SLD [$\AA^{-2}$]"

    def plot(self, name, f: Function, **kwargs):
        self.axs.plot(f.get_domain(), f.eval(), label=name, **self._plot_args, **kwargs)
        self.axs.set_xlabel("depth [$\AA$]")
        self.axs.set_ylabel(self._ylabel)
        self.axs.legend()

        return self

class XRaySLDPlotter(SLDPlotter):
    def _default_args(self):
        self._ylabel = "SLD [$r_e /\AA ^ 3$]"


class PhasePlotter(FunctionPlotter):
    def _default_args(self):
        self.scale = True
        self.real = True
        self.imag = True

    @classmethod
    def transform(cls):
        return lambda x, fx: (100 * x) ** 2 * fx

    @classmethod
    def inverse_transform(cls):
        return lambda x, fx: fx / (100 * x)**2

    def plot(self, name, f: Function, **kwargs):
        if f is None:
            return

        transform = self.transform()
        dy = f.dy
        if self.scale:
            f = f.transform(transform)
            if dy is not None:
                dy = dy.transform(transform)
            ylabel = "(100 q$)^2$ $R(q)$ [$10^{-4} \AA^{-2}$]"
        else:
            ylabel = "R(q) [1]"

        feval = f.eval()

        color = [None, None]

        if "color" in kwargs:
            if not isinstance(kwargs["color"], list) or len(kwargs["color"]) <= 1:

                color = [kwargs["color"], kwargs["color"]]
                kwargs.pop("color")
            else:
                color = kwargs.pop("color")

        if self.real:
            if name is not None:
                kwargs["label"] = "Re R(q) {}".format(name)
            self._do_plot(f.get_domain(), feval.real, color[0], dy, **kwargs)

        if self.imag:
            if name is not None:
                kwargs["label"] = "Im R(q) {}".format(name)

            self._do_plot(f.get_domain(), feval.imag, color[1], dy, imag=True, **kwargs)

        self.axs.set_xlabel("q [$\AA$]")
        self.axs.set_ylabel(ylabel)
        self.axs.legend()

        return self

    def _do_plot(self, x, y, color, dy=None, imag=False, **kwargs):
        if dy is None:
            self.axs.plot(x, y, **self._plot_args, color=color, **kwargs)
        else:
            if imag:
                dy = dy.imag
            else:
                dy = dy.real

            self.axs.errorbar(x, y, yerr=dy.eval(), **self._plot_args, color=color, **kwargs)


class ReflectionFileLoader(FunctionFileLoader):
    def to_file(self, function: Function):
        transform = PhasePlotter.transform()

        f = function.transform(transform)
        if function.dy is not None:
            f.set_dy(function.dy.transform(transform))
        header = "q, (100 q)**2 Re R(q), (100 q)**2 Im R(q)"
        return super(ReflectionFileLoader, self).to_file(f, header=header)

    def from_file(self, has_errors=False):
        f = super(ReflectionFileLoader, self).from_file(has_errors=has_errors)

        transform = PhasePlotter.inverse_transform()
        fun = f.transform(transform)
        if f.dy is not None:
            fun.set_dy(f.dy.transform(transform))
        return fun

class AmorFileLoader(FunctionFileLoader):
    def from_file(self):
        from numpy import loadtxt
        q, R, dq, dR = loadtxt(self._file).T

        f = Function.to_function(q, R)
        df = Function.to_function(q, dR)
        dx = Function.to_function(q, dq)
        f.set_dy(df)
        f.set_dx(dx)
        return f

    def to_file(self):
        raise RuntimeError("Should not write AMOR data files")

class NarzissFileLoader(FunctionFileLoader):
    def from_file(self):
        from numpy import loadtxt

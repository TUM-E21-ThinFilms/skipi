from skipi.plot import FunctionPlotter
from skipi.function import Function

class ReflectivityPlotter(FunctionPlotter):
    def plot(self, name, f: Function, **kwargs):
        self.axs.plot(f.get_domain(), f.eval(), label=name, **self._plot_args, **kwargs)
        self.axs.set_yscale("log")
        self.axs.set_xlabel("q [$\AA^{-1}$]")
        self.axs.set_ylabel("log Intensity [1]")
        self.axs.legend()

class SLDPlotter(FunctionPlotter):
    def _default_args(self):
        self._ylabel = "SLD [$\AA^-2$]"

    def plot(self, name, f: Function, **kwargs):
        self.axs.plot(f.get_domain(), f.eval, label=name, **self._plot_args, **kwargs)
        self.axs.set_xlabel("depth [$\AA$]")
        self.axs.set_ylabel(self._ylabel)
        self.axs.legend()

class XRaySLDPlotter(SLDPlotter):
    def _default_args(self):
        self._ylabel = "SLD [$r_e /\AA ^ 3$]"

class PhasePlotter(FunctionPlotter):
    def _default_args(self):
        self.scale = True
        self.real = True
        self.imag = True

    def plot(self, name, f: Function, **kwargs):
        transform = lambda x, fx: (100 * x)**2 * fx

        if self.scale:
            f = f.transform(transform)
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

            self.axs.plot(f.get_domain(), feval.real, **self._plot_args, color=color[0], **kwargs)

        if self.imag:
            if name is not None:
                kwargs["label"] = "Im R(q) {}".format(name)

            self.axs.plot(f.get_domain(), feval.imag, **self._plot_args, color=color[1], **kwargs)

        self.axs.set_xlabel("q [$\AA$]")
        self.axs.set_ylabel(ylabel)
        self.axs.legend()
import matplotlib.pyplot as plt

from skipi.function import Function

class FunctionPlotter(object):
    def __init__(self, axis=None, figure=None):
        if figure is None:
            figure = plt.figure()

        if axis is None:
            axis = plt.axes()

        self._axs, self._fig = axis, figure
        self._plot_args = {}

        self.plot_dx = True
        self.plot_dy = True

        self._default_args()

    @property
    def colors(self):
        return plt.rcParams['axes.prop_cycle'].by_key()['color']

    @property
    def colors2(self):
        # Add here contrary colors
        return [['b', 'y'], ['r', 'c']]

    def _default_args(self):
        pass

    @property
    def fig(self):
        return self.get_figure()

    @property
    def axs(self):
        return self.get_axis()

    @axs.setter
    def axs(self, axs):
        self._axs = axs

    def get_figure(self):
        return self._fig

    def get_axis(self):
        return self._axs

    def plot_args(self, **kwargs):
        self._plot_args = kwargs

    """
    def plot(self, name, f: Function, **kwargs):
        f = self._manipulate(f)

        plot_space = self._space(f)

        self._axs.plot(plot_space, f(plot_space), **self._plot_args, label=name, **kwargs)
        self._axs.legend()
    """

    def plot(self, name, f: Function, **kwargs):

        x = f.get_domain()
        y = f.eval()

        plot_function = self._axs.plot

        kwargs['label'] = name

        if f.dx is not None or f.dy is not None:
            if f.dx is not None and self.plot_dx is True:
                kwargs['xerr'] = f.dx.eval()

            if f.dy is not None and self.plot_dy is True:
                kwargs['yerr'] = f.dy.eval()

            plot_function = self._axs.errorbar

        plot_function(x, y, **self._plot_args, **kwargs)

        return self

    def show(self):
        plt.show()


class FillBetweenPlotter(FunctionPlotter):
    def _default_args(self):
        self._plot_args['alpha'] = 0.3
        self._plot_args['color'] = self.colors[0]

    def plot(self, name, f: Function, f0: Function, f1: Function = None, **kwargs):
        x = f.get_domain()
        y = f.eval()

        kwargs['label'] = name
        if 'color' in kwargs:
            self._plot_args['color'] = kwargs['color']

        self._axs.fill_between(x, y, f0(x), **self._plot_args)
        if f1 is not None:
            self._axs.fill_between(x, y, f1(x), **self._plot_args)

        super(FillBetweenPlotter, self).plot(name, f, **kwargs)

        return self
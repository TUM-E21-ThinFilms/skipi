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
        return self._figure

    def get_axis(self):
        return self._axs

    def _manipulate(self, f: Function):
        return f

    def _space(self, f: Function):
        return f.get_domain()

    def plot_args(self, **kwargs):
        self._plot_args = kwargs

    def plot(self, name, f: Function, **kwargs):
        f = self._manipulate(f)

        plot_space = self._space(f)

        self._axs.plot(plot_space, f(plot_space), **self._plot_args, label=name, **kwargs)
        self._axs.legend()

    def show(self):
        plt.show()


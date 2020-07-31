import numpy

from typing import List


class Domain:
    def __init__(self, x_min, x_max, npts):
        self._xmin = x_min
        self._xmax = x_max
        self._npts = npts

    @classmethod
    def linear(cls, x_min, x_max, npts):
        return numpy.linspace(x_min, x_max, npts)

    @classmethod
    def get_dx(self, grid):
        return grid[1] - grid[0]

    @classmethod
    def grid(self, grids, dx):
        x_min = min(map(min, grids))
        x_max = max(map(max, grids))
        return self.linear(x_min, x_max, int((x_max - x_min) / dx) + 1)

    @classmethod
    def fine_grid(self, grids: List):
        dx = min(map(self.get_dx, grids))
        return self.grid(grids, dx)

    @classmethod
    def coarse_grid(self, grids: List):
        dx = max(map(self.get_dx, grids))
        return self.grid(grids, dx)

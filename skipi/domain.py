import numpy

from typing import List

from skipi.util import is_number, vslice

PRINT_PRECISION = 3


class Domain:
    def __init__(self, x_min: float, x_max: float, npts: int = 3):
        self._xmin = x_min
        self._xmax = x_max
        self._npts = npts

        self._dom = None

    def get(self):
        if self._dom is None:
            self._dom = self.create()

        return self._dom

    def min(self):
        return self._xmin

    def max(self):
        return self._xmax

    def length(self):
        return self._xmax - self._xmin

    def points(self):
        return self._npts

    def dx(self):
        return self.get_dx(self)

    def create(self):
        return self.linear(self._xmin, self._xmax, self._npts)

    def respace(self, dx):
        return Domain(self._xmin, self._xmax, int((self._xmax - self._xmin) / dx) + 1)

    def resample(self, npts):
        return Domain(self._xmin, self._xmax, npts)

    def oversample(self, n):
        if n <= 0:
            raise RuntimeError("The oversampling-factor n has to be a positive integer")

        return Domain(self._xmin, self._xmax, self._npts * n + 1)

    def shift(self, offset):
        return Domain(self._xmin + offset, self._xmax + offset, self._npts)

    def scale(self, factor):
        return Domain(self._xmin * factor, self._xmax * factor, self._npts)

    def vremesh(self, *selector, dstart=0, dstop=0):
        return Domain.from_domain(vslice(self.get(), *selector, dstart=dstart, dstop=dstop))

    def __add__(self, other):
        if is_number(other):
            return self.shift(other)

    def __sub__(self, other):
        if is_number(other):
            return self.shift(-other)

    def __str__(self):
        p = str(PRINT_PRECISION)
        format_str = "[{:." + p + "e}, {:." + p + "e}] @ dx = {:." + p + "e} (#pts = {})"
        return format_str.format(self._xmin, self._xmax, self.dx(), self._npts)

    def __repr__(self):
        return self.__str__

    @classmethod
    def from_domains(cls, others: List['Domain'], method_or_mesh=None):
        if method_or_mesh is None:
            method_or_mesh = cls.coarse_grid

        if isinstance(method_or_mesh, Domain):
            return method_or_mesh.get()

        if callable(method_or_mesh):
            return method_or_mesh(others)

        return method_or_mesh

    @classmethod
    def from_domain(cls, domain):
        # Assuming that the domain is equidistantly spaced
        if isinstance(domain, numpy.ndarray):
            return cls(domain.min(), domain.max(), len(domain))

        if isinstance(domain, Domain):
            return domain

        if callable(domain):
            return cls.from_domain(domain())

        raise RuntimeError("Unknown type of domain to create a Domain class from")

    @classmethod
    def as_array(cls, domain):
        if isinstance(domain, numpy.ndarray):
            return domain

        if isinstance(domain, Domain):
            return domain.get()

        if callable(domain):
            return cls.get_from_domain(domain())

        raise RuntimeError("Unknown type of domain to create a Domain class from")

    @classmethod
    def linear(cls, x_min, x_max, npts):
        return numpy.linspace(x_min, x_max, npts)

    # TODO: implement iterator

    def __iter__(self):
        dom = self.get()
        return iter(dom)

    def __len__(self):
        return self.len()

    def len(self):
        return self._npts

    @classmethod
    def get_dx(self, grid):
        if isinstance(grid, Domain):
            return (grid._xmax - grid._xmin) / grid._npts

        return grid[1] - grid[0]

    @classmethod
    def grid(self, grids, dx):
        x_min = min(map(lambda x: x.min(), grids))
        x_max = max(map(lambda x: x.max(), grids))
        return Domain(x_min, x_max, int((x_max - x_min) / dx) + 1)

    @classmethod
    def fine_grid(self, grids: List):
        dx = min(map(self.get_dx, grids))
        return self.grid(grids, dx)

    @classmethod
    def coarse_grid(self, grids: List):
        dx = max(map(self.get_dx, grids))
        return self.grid(grids, dx)

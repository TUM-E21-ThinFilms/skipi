import numpy

from typing import List

from skipi.util import is_number, vslice

PRINT_PRECISION = 3

class Domain:
    def __init__(self, x_min: float, x_max: float, npts: int = 3):
        self._xmin = x_min
        self._xmax = x_max
        self._npts = int(npts)

        self._dom = None

    def get(self):
        if self._dom is None:
            self._dom = self.create()

        return self._dom

    def min(self):
        return self._xmin

    def max(self):
        return self._xmax

    def limits(self):
        return self.min(), self.max()

    def length(self):
        return self._xmax - self._xmin

    def points(self):
        return self._npts

    def len(self):
        return self._npts

    def dx(self):
        return self.get_dx(self)

    def create(self):
        return self.linear(self._xmin, self._xmax, self._npts)

    def respace(self, dx):
        return Domain(self._xmin, self._xmax, int((self._xmax - self._xmin) / dx) + 1)

    def extend(self, xmin=None, xmax=None):
        dx = self.dx()
        new_xmin, new_xmax = self._xmin, self._xmax
        if xmin < self._xmin:
            new_xmin = self._xmin - ((self._xmin - xmin) // dx) * dx

        if xmax > self._xmax:
            new_xmax = self._xmax + ((xmax - self._xmax) // dx) * dx

        return Domain.from_spacing(new_xmin, new_xmax, dx)

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

    def idx(self, items):
        items = numpy.array(items)
        idxs = numpy.array((items - self._xmin) / self.dx(), dtype=int)
        sel = numpy.bitwise_and(self._xmin <= items, items < self._xmax)

        return idxs, sel

    def array(self, empty_value=0):
        return numpy.full((self.len()), empty_value, dtype=float)

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
        return self.__str__()

    def __contains__(self, item):
        return self._xmin <= item < self._xmax

    def __iter__(self):
        return iter(self.get())

    def __len__(self):
        return self.len()

    def __lt__(self, other):
        if isinstance(other, Domain):
            return  self.min() < other.min()
        elif is_number(other):
            return self.min() < other
        else:
            raise RuntimeError("Unknown type to compare domain with")

    def __gt__(self, other):
        if isinstance(other, Domain):
            return  self.max() > other.max()
        elif is_number(other):
            return self.max() > other
        else:
            raise RuntimeError("Unknown type to compare domain with")

    def has_intersection(self, domain):
        _min = max(self.min(), domain.min())
        _max = min(self.max(), domain.max())

        if _min > _max:
            return False
        return True

    def intersect(self, domain):
        _min = max(self.min(), domain.min())
        _max = min(self.max(), domain.max())

        if _min > _max:
            raise RuntimeError("Intersection is the empty set")

        return self.vremesh((_min, _max))

    def unite(self, domain):
        _min = min(self.min(), domain.min())
        _max = max(self.max(), domain.max())

        if not self.has_intersection(domain):
            raise RuntimeError("Intersection is the empty set")

        return Domain.from_spacing(_min, _max, self.dx())

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

    @classmethod
    def from_spacing(cls, x_min, x_max, dx):
        return Domain(x_min, x_max, (x_max - x_min) / dx + 1)

    @classmethod
    def from_array(cls, values):
        values = numpy.array(values)
        return Domain(values.min(), values.max(), len(values))

    @classmethod
    def get_dx(self, grid):
        if isinstance(grid, Domain):
            return (grid._xmax - grid._xmin) / grid._npts

        return grid[1] - grid[0]

    @classmethod
    def grid(cls, grids, dx):
        x_min = min(map(lambda x: x.min(), grids))
        x_max = max(map(lambda x: x.max(), grids))
        return cls.from_spacing(x_min, x_max, dx)

    @classmethod
    def fine_grid(cls, grids: List):
        dx = min(map(cls.get_dx, grids))
        return cls.grid(grids, dx)

    @classmethod
    def coarse_grid(cls, grids: List):
        dx = max(map(cls.get_dx, grids))
        return cls.grid(grids, dx)


class IntersectionDomain(Domain):
    @classmethod
    def from_domains(cls):
        pass


class UnionDomain(Domain):
    @classmethod
    def from_domains(cls):
        pass


class Domain2D(object):
    def __init__(self, dom_x: Domain, dom_y: Domain):
        self._domx = dom_x
        self._domy = dom_y

    def dx(self):
        return self._domx.dx()

    def dy(self):
        return self._domy.dx()

    def points(self):
        return self._domx.points() * self._domy.points()

    def get(self):
        return numpy.meshgrid(self._domx.get(), self._domy.get())

    def array(self, empty_value=0):
        return numpy.full((len(self._domx), len(self._domy)), empty_value, dtype=float)

    def transpose(self):
        return Domain2D(self._domy, self._domx)

    def idx(self, items_x, items_y):
        x_idx, x_sel = self._domx.idx(items_x)
        y_idx, y_sel = self._domy.idx(items_y)

        return x_idx, y_idx, numpy.bitwise_and(x_sel, y_sel)
import numpy

from typing import List

from .function import Function

class FunctionCollection(object):
    def __init__(self, fs: List[Function]):
        self._fs = fs
        self._ds = [f.get_domain() for f in fs]

    def add(self, function: Function):
        self._fs.append(function)
        self._ds.append(function.get_domain())
        pass

    def get_domain(self):
        return numpy.concatenate(self._ds)

    def __call__(self, x):
        return numpy.sum([f(x) for f in self._fs])

    def __add__(self, other):
        pass
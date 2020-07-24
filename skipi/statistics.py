import numpy

from typing import List
from skipi.function import Function


class AverageFunction(Function):

    @classmethod
    def from_functions(cls, functions: List[Function], domain=None):
        r"""
        Returns the average function based on the functions given as a list F = [f_1, ..., f_n]
        ::math..
            f_avg(x) = 1/n * (f_1(x) + \ldots + f_n(x))
        where f_i is an element of F

        :param functions: List of functions to average
        :return:
        """
        n = len(functions)

        if n == 0:
            raise RuntimeError("Cannot average functions if no function was given")
        if n == 1:
            return functions[0]

        if domain is None:
            domain = functions[0].get_domain()

        # sum of axis=0, since x might be a vector containing multiple evaluation points
        return cls(domain, lambda x: numpy.sum([f(x) for f in functions], axis=0) / n)

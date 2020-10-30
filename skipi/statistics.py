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


class ComputeAverage(Function):
    @classmethod
    def from_functions(cls, functions: [Function], domain=None, avg_fun=None):
        if domain is None:
            domain = functions[0].get_domain()
        if avg_fun is None:
            avg_fun = cls.avg

        return Function.to_function(domain, lambda x: avg_fun([f(x) for f in functions]))

    @staticmethod
    def avg(numbers):
        numbers = numpy.array(numbers)
        return numpy.average(numbers.real) + 1j * numpy.average(numbers.imag)


class DrawFromFunction(Function):
    @classmethod
    def from_function(cls, function: Function):
        dy = function.dy

        if dy is None:
            return function

        value = numpy.random.normal(function.eval().real, dy.eval().real)

        if function.is_complex():
            value = value + 1j * numpy.random.normal(function.eval().imag, dy.eval().imag)

        return Function.to_function(function.get_dom(), value)


class ComputeStandardDeviation(Function):
    @classmethod
    def from_functions(cls, functions: [Function], domain=None, std_fun=None):
        """
        Computes the standard deviation (pointwise) using all functions

        If domain is None, the domain from the first function will be used

        If std_fun is None, the "complex" standard deviation will be used, see the method cstd.


        :param functions: A list of functions from which the std should be calculated
        :param domain: A domain
        :param std_fun: A function calculating the std
        :return: new Function
        """
        if domain is None:
            domain = functions[0].get_domain()

        if std_fun is None:
            std_fun = cls.cstd

        return Function.to_function(domain, lambda x: std_fun([f(x) for f in functions]))

    @staticmethod
    def cstd(complexs):
        """
        Calculates the standard deviation of a complex number by splitting it into the real and imaginary
        part, resulting in a complex standard deviation:

            cstd(complex) = std(complex.real) + 1j*std(complex.imag).

        :param complexs:
        :return:
        """
        complexs = numpy.array(complexs)
        return numpy.std(complexs.real) + 1j * numpy.std(complexs.imag)


class MaxOfFunctions(Function):
    @classmethod
    def from_functions(cls, functions: [Function]):
        return Function.to_function(functions[0].get_dom(), lambda x: numpy.max([f(x) for f in functions]))


class MinOfFunctions(Function):
    @classmethod
    def from_functions(cls, functions: [Function]):
        return Function.to_function(functions[0].get_dom(), lambda x: numpy.min([f(x) for f in functions]))

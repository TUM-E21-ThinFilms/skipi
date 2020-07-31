import numpy

from typing import List, Callable, Any

from skipi.function import Function
from skipi.domain import Domain


class ParametrizedFunction(Function):

    def __init__(self, domain, parametrized_function_callable: Callable[[Any], Callable[[Any], Any]],
                 parameter_names: List[str], parameters: List[Any]):
        """
        Creates a parametrized function.

        A parametrized function is a function which depends on given parameters. So, given a set of
        parameters, one can evaluate the function at any x value which is in the domain (the domain itself
        might depend on the parameters again).

        This class acts like a normal function, but it contains additionally a function/domain factory which
        creates the domain/function given the parameters. BUT: After using it together with other functions,
        the class will degenerate to a normal function.

        The parameters are assumed to be an array of values.

        If the domain does not depend on the parameters (or is constant so to speak), you can simply pass a
        numpy linspace grid to it, and the domain factory will be created automatically.

        The parametrized_function_callable is a function which accepts _only_ parameters and returns _only_
        a function which can be called to evaluate the function, i.e.
        :Example:
        >>> lambda p: lambda x: p * x
        >>> lambda p: lambda x: lambda p: lambda x: numpy.exp(-(x - p[0]) ** 2 / 2*(p[1] ** 2))
        The first example will a parametrized linear function, f_p(x) = p*x
        and the second example creates the gaussian function (unnormalized) with p[0] being mu and p[1]
        being sigma, i.e.
            f_m,s(x) = e^(- (x-m)^2 / 2s^2) )

        :param domain: grid, or callable which creates a grid given parameters
        :param parametrized_function_callable: callable which creates a function given parameters
        :param parameter_names: a list containing the parameters
        :param parameters: a list of the parameters for the first initialization
        """
        if not callable(domain):
            if not isinstance(domain, numpy.ndarray):
                domain = numpy.array(domain)
            domain = lambda x: domain

        self._dom_factory = domain

        if not callable(parametrized_function_callable):
            raise RuntimeError("function must be callable")

        self._dom_factory = domain
        self._f_factory = parametrized_function_callable

        if not len(parameter_names) == len(parameters):
            raise RuntimeError("Number of given parameters does not match the expected number")

        self._param_names = parameter_names
        self._params = parameters

        dom = self._dom_factory(parameters)
        f = self._f_factory(parameters)

        super(ParametrizedFunction, self).__init__(dom, f)

    def copy(self):
        return ParametrizedFunction(self._dom_factory, self._f_factory, self._param_names, self._params)

    def reparametrize(self, params):
        """
        Returns a new parametrized function, just with the parameters updated

        :param params:  a list of the parameters
        :return:
        """
        return ParametrizedFunction(self._dom_factory, self._f_factory, self._param_names, params)

    def get_params(self):
        """
        Returns the parameters used for calculation

        :return:
        """
        return self._params

    def get_param_names(self):
        """
        Returns the parameter names

        :return:
        """
        return self._param_names

    def get_params_dict(self):
        """
        Returns a dictionary containing the parameter names as keys, and the parameter values as its values

        :return:
        """
        return dict(zip(self._param_names, self._params))

    def get_domain_factory(self):
        """
        Returns the factory for creating the domain, given parameters

        :return:
        """
        return self._dom_factory

    def get_function_factory(self):
        """
        Returns the factory for creating the internal function, given parameters

        :return:
        """
        return self._f_factory


class Combine(Function):
    def __init__(self, domain, functions: List[Callable], operator: Callable):

        self._fs = functions
        self._operator = operator

        super(Combine, self).__init__(domain, self._operation(functions, operator))

    @classmethod
    def _operation(cls, functions: List[Callable], operator: Callable):
        return lambda x: operator(functions)(x)

    @classmethod
    def from_functions(cls, functions: List[Callable], operator=numpy.sum, domain=Domain.fine_grid):
        if callable(domain):
            grids = [f.get_domain() for f in functions]
            domain = domain(grids)

        return cls(domain, functions, operator)

    def get_functions(self):
        return self._fs

    def __getitem__(self, item):
        return self._fs[item]

    def __setitem__(self, item, value):
        if value is None:
            del self[item]
        else:
            self._fs[item] = value

    def __delitem__(self, key):
        del self._fs[key]

    def __or__(self, other):
        if isinstance(other, Function):
            self._fs.append(other)
        else:
            raise RuntimeError("Unknown type of other")

        return self

    def append(self, others):
        if isinstance(others, list):
            self._fs.extend(others)
        else:
            self._fs.append(others)


Gaussian = ParametrizedFunction(lambda p: numpy.linspace(p[0] - 5 * p[1], p[0] + 5 * p[1], 2000),
                                lambda p: lambda x: numpy.exp(-(x - p[0]) ** 2 / p[1] ** 2) * 1 / numpy.sqrt(
                                    2 * numpy.pi * p[1] ** 2),
                                ["mu", "sigma"],
                                [2.3, 1.0])

f2 = Gaussian.reparametrize([1.0, 1.0])
f3 = Gaussian.reparametrize([0, 20])

F = Combine.from_functions([f2, Gaussian, f3], operator=numpy.sum, domain=Domain.fine_grid)

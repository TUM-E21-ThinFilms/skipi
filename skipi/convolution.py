import numpy
from skipi.function import Function

class AbstractConvolution(Function):
    @classmethod
    def get_kernel(cls, fun: Function, **kwargs):
        raise RuntimeError("Not implemented")

    @classmethod
    def from_function(cls, fun: Function, **kwargs):
        domain = fun.get_domain()
        conv = numpy.convolve(fun(domain), cls.get_kernel(fun, **kwargs), mode='same')
        return Function.to_function(domain, conv)


class GaussianSmoothing(AbstractConvolution):
    @classmethod
    def get_kernel(cls, fun: Function, **kwargs):
        r"""
        Returns the gaussian kernel, centered at \mu = 0 and variance = sigma

        :param fun: Function to apply the gaussian kernel (needed for spacing purposes)
        :param kwargs: possible variable: sigma
        :return:
        """
        sigma = kwargs.get('sigma', 1.0)
        spacing = fun.get_dx(fun.get_domain())
        width = numpy.arange(-5 * sigma, 5 * sigma, spacing)
        kernel = 1.0 / numpy.sqrt(2 * numpy.pi * sigma ** 2) * numpy.exp(
            -numpy.square(width / sigma) / 2.0) * spacing
        return kernel
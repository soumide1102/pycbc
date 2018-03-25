# Copyright (C) 2016  Christopher M. Biwer, Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
This modules provides classes for evaluating Gaussian distributions.
"""

import numpy
import scipy.stats as stats
from pycbc.distributions import bounded
from sklearn import neighbors
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
import pylab
from pycbc.io.record import FieldArray

class GaussianBimodal(bounded.BoundedDist):
    r"""A Gaussian distribution on the given parameters; the parameters are
    independent of each other.
    
    Bounds can be provided on each parameter, in which case the distribution
    will be a truncated Gaussian distribution.  The PDF of a truncated
    Gaussian distribution is given by:

    .. math::
        p(x|a, b, \mu,\sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}\frac{e^{- \frac{\left( x - \mu \right)^2}{2 \sigma^2}}}{\Phi(b|\mu, \sigma) - \Phi(a|\mu, \sigma)},

    where :math:`\mu` is the mean, :math:`\sigma^2` is the variance,
    :math:`a,b` are the bounds, and :math:`\Phi` is the cumulative distribution
    of an unbounded normal distribution, given by:

    .. math::
        \Phi(x|\mu, \sigma) = \frac{1}{2}\left[1 + \mathrm{erf}\left(\frac{x-\mu}{\sigma \sqrt{2}}\right)\right].

    Note that if :math:`[a,b) = [-\infty, \infty)`, this reduces to a standard
    Gaussian distribution.

    
    Instances of this class can be called like a function. By default, logpdf
    will be called, but this can be changed by setting the class's __call__
    method to its pdf method.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and
        (optionally) some bounds, as either a tuple or a
        `boundaries.Bounds` instance. The mean and variance of each
        parameter can be provided by additional keyword arguments that have
        `_mean` and `_var` adding to the parameter name. For example,
        `foo=(-2,10), foo_mean=3, foo_var=2` would create a truncated Gaussian
        with mean 3 and variance 2, bounded between :math:`[-2, 10)`. If no
        mean or variance is provided, the distribution will have 0 mean and
        unit variance. If None is provided for the bounds, the distribution
        will be a normal, unbounded Gaussian (equivalent to setting the bounds
        to `[-inf, inf)`).

    Attributes
    ----------------
    name : 'guassian'
        The name of this distribution.

    Examples
    --------
    Create an unbounded Gaussian distribution with zero mean and unit variance:
    >>> dist = distributions.Gaussian(mass1=None)

    Create a bounded Gaussian distribution on :math:`[1,10)` with a mean of 3
    and a variance of 2:
    >>> dist = distributions.Gaussian(mass1=(1,10), mass1_mean=3, mass1_var=2)
   
    Create a bounded Gaussian distribution with the same parameters, but with
    cyclic boundary conditions:
    >>> dist = distributions.Gaussian(mass1=Bounds(1,10, cyclic=True), mass1_mean=3, mass1_var=2)
    """
    name = "gaussian"

    def __init__(self, **params):

        # save distribution parameters as dict
        # calculate the norm and exponential norm ahead of time
        # and save to self._norm, self._lognorm, and self._expnorm
        self._bounds = {}
        self._mean1 = {}
        self._var1 = {}
        self._r1 = {}
        self._mean2 = {}
        self._var2 = {}
        self._r2 = {}
        self._norm1 = {}
        self._lognorm1 = {}
        self._expnorm1 = {}
        self._norm2 = {}
        self._lognorm2 = {}
        self._expnorm2 = {}
        # pull out specified means, variance
        mean1_args = [p for p in params if p.endswith('_mean1')]
        var1_args = [p for p in params if p.endswith('_var1')]
        mean2_args = [p for p in params if p.endswith('_mean2')]
        var2_args = [p for p in params if p.endswith('_var2')]
        r1_args = [p for p in params if p.endswith('_r1')]
        r2_args = [p for p in params if p.endswith('_r2')]
        self._mean1 = dict([[p[:-6], params.pop(p)] for p in mean1_args])
        self._var1 = dict([[p[:-5], params.pop(p)] for p in var1_args])
        self._mean2 = dict([[p[:-6], params.pop(p)] for p in mean2_args])
        self._var2 = dict([[p[:-5], params.pop(p)] for p in var2_args])
        self._r1 = dict([[p[:-3], params.pop(p)] for p in r1_args])
        self._r2 = dict([[p[:-3], params.pop(p)] for p in r2_args])
        # initialize the bounds
        super(GaussianBimodal, self).__init__(**params)

        # check that there are no params in mean/var that are not in params
        missing = set(self._mean1.keys()) - set(params.keys())
        print(self._mean1.keys())
        print(params.keys())
        if any(missing):
            raise ValueError("means provided for unknow params {}".format(
                ', '.join(missing)))
        missing = set(self._mean2.keys()) - set(params.keys())
        if any(missing):
            raise ValueError("vars provided for unknow params {}".format(
                ', '.join(missing)))
        missing = set(self._var1.keys()) - set(params.keys())
        if any(missing):
            raise ValueError("means provided for unknow params {}".format(
                ', '.join(missing)))
        missing = set(self._var2.keys()) - set(params.keys())
        if any(missing):
            raise ValueError("vars provided for unknow params {}".format(
                ', '.join(missing)))
        missing = set(self._r1.keys()) - set(params.keys())
        if any(missing):
            raise ValueError("means provided for unknow params {}".format(
                ', '.join(missing)))
        missing = set(self._r2.keys()) - set(params.keys())
        if any(missing):
            raise ValueError("vars provided for unknow params {}".format(
                ', '.join(missing)))

        # set default mean/var for params not specified
        self._mean1.update(dict([[p, 0.]
            for p in params if p not in self._mean1]))
        self._var1.update(dict([[p, 1.]
            for p in params if p not in self._var1]))
        self._mean2.update(dict([[p, 0.]
            for p in params if p not in self._mean2]))
        self._var2.update(dict([[p, 1.]
            for p in params if p not in self._var2]))
        self._r1.update(dict([[p, 1.]
            for p in params if p not in self._r1]))
        self._r2.update(dict([[p, 1.]
            for p in params if p not in self._r2]))

        # compute norms
        #for p,bnds in self._bounds.items():
        #    sigma1 = numpy.sqrt(self._var1[p])
        #    mu1 = self._mean1[p]
        #    sigma2 = numpy.sqrt(self._var2[p])
        #    mu2 = self._mean2[p]
        #    a,b = bnds
            #invnorm1 = scipy.stats.norm.cdf(b, loc=mu1, scale=sigmasq1**0.5) \
            #        - scipy.stats.norm.cdf(a, loc=mu1, scale=sigmasq1**0.5)
            #invnorm1 *= numpy.sqrt(2*numpy.pi*sigmasq1)
            #invnorm2 = scipy.stats.norm.cdf(b, loc=mu2, scale=sigmasq2**0.5) \
            #        - scipy.stats.norm.cdf(a, loc=mu2, scale=sigmasq2**0.5)
            #invnorm2 *= numpy.sqrt(2*numpy.pi*sigmasq2)
            #self._norm1[p] = 1./invnorm1
            #self._lognorm1[p] = numpy.log(self._norm1[p])
            #self._expnorm1[p] = -1./(2*sigmasq1)
            #self._norm2[p] = 1./invnorm2
            #self._lognorm2[p] = numpy.log(self._norm2[p])
            #self._expnorm2[p] = -1./(2*sigmasq2)


    @property
    def mean1(self):
        return self._mean1


    @property
    def var1(self):
        return self._var1

    @property
    def mean2(self):
        return self._mean2


    @property
    def var2(self):
        return self._var2

    @property
    def r1(self):
        return self._r1
 
    @property
    def r2(self):
        return self._r2


    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        pdf=1.0
        print("self._params", self._params)
        for p in self._params:
            print("p",p)
            print("kwargs[p]")
            print(kwargs[p])
            #print("self._mean1[p]", self._mean1[p])
            #print("self._var1[p]", self._var1[p])
            #g1 = stats.norm.pdf(kwargs[p], loc=self._mean1[p], scale=numpy.sqrt(self._var1[p]))
            #g2 = stats.norm.pdf(kwargs[p], loc=self._mean2[p], scale=numpy.sqrt(self._var2[p]))
            g1, g2 = self.get_gaussian_pdf(kwargs[p], self._mean1[p], self._mean2[p], self._var1[p], self._var1[p])
            pdf_param=self.r1[p]*g1 + self.r2[p]*g2
            #print(pdf_param)
            #pdf_param_norm=pdf_param/pdf_param.sum()
            #print(pdf_param_norm)
            #pdf_norm*=pdf_param
            pdf*=pdf_param
            #print(pdf_norm)
        return pdf
        #return numpy.exp(self._logpdf(**kwargs))

    @staticmethod
    def get_gaussian_pdf(x, mean1, mean2, var1, var2):
        g1 = stats.norm.pdf(x, loc=mean1, scale=numpy.sqrt(var1))
        g2 = stats.norm.pdf(x, loc=mean2, scale=numpy.sqrt(var2))
        return g1, g2

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        if kwargs in self: 
            return numpy.log(self._pdf(**kwargs))
            #return sum([self._lognorm[p] +
            #            self._expnorm[p]*(kwargs[p]-self._mean[p])**2.
            #            for p in self._params])
        else:
            return -numpy.inf

    #def set_pdf(self, x, pdf, Nrl=1000):
    #    self.x = x
    #    self.pdf = pdf
        

    def rvs(self, size=1, param=None):
        """Gives a set of random values drawn from this distribution.

        Parameters
        ----------
        size : {1, int}
            The number of values to generate; default is 1.
        param : {None, string}
            If provided, will just return values for the given parameter.
            Otherwise, returns random values for each parameter.

        Returns
        -------
        structured array
            The random values in a numpy structured array. If a param was
            specified, the array will only have an element corresponding to the
            given parameter. Otherwise, the array will have an element for each
            parameter in self's params.
        """
        if param is not None:
            dtype = [(param, float)]
        else:
            dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        for (p,_) in dtype:
            sigma1 = numpy.sqrt(self._var1[p])
            mu1 = self._mean1[p]
            sigma2 = numpy.sqrt(self._var2[p])
            mu2 = self._mean2[p]
            r1 = self._r1[p]
            r2 = self._r2[p]
            a,b = self._bounds[p]
            x = numpy.linspace(a,b,1000)
            #d = {p : list(x)}
            g1, g2 = self.get_gaussian_pdf(x, self._mean1[p], self._mean2[p], self._var1[p], self._var1[p])
            pdf_param=self.r1[p]*g1 + self.r2[p]*g2
            pdf = pdf_param/pdf_param.sum()
            #pdf=self._pdf(**d)
            #pdf=pdf[0]/pdf[0].sum()
            #print("pdf")
            #print(pdf)
            cdf = pdf.cumsum()
            #print("cdf")
            #print(cdf)
            nrl = 100000
            inversecdfbins = nrl
            #self.Nrl = Nrl
            y = pylab.arange(nrl)/float(nrl)
            delta = 1.0/nrl
            inversecdf = pylab.zeros(nrl)
            inversecdf[0] = x[0]
            cdf_idx = 0
            for n in xrange(1,inversecdfbins):
                while cdf[cdf_idx] < y[n] and cdf_idx < nrl:
                    cdf_idx += 1
                #print(cdf[cdf_idx], cdf[cdf_idx-1])
                inversecdf[n] = x[cdf_idx-1] + (x[cdf_idx] - x[cdf_idx-1]) * (y[n] - cdf[cdf_idx-1])/(cdf[cdf_idx] - cdf[cdf_idx-1])
                if cdf_idx >= nrl:
                    break
            delta_inversecdf = pylab.concatenate((pylab.diff(inversecdf), [0]))

            #numpy.random.seed(seed=0)
            idx_f = numpy.random.uniform(size = size, high = nrl-1)
            idx = pylab.array([idx_f],'i')
            arr[p][:] = inversecdf[idx] + (idx_f - idx)*delta_inversecdf[idx]
            
        return arr

            #arr[p][:] = scipy.stats.truncnorm.rvs((a-mu)/sigma, (b-mu)/sigma,
            #    loc=self._mean[p], scale=sigma, size=size)
            #kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)
            
            #params = {'bandwidth': numpy.logspace(-1, 1, 20)}
            #grid = GridSearchCV(neighbors.KernelDensity(), params)
            #grid.fit(x)
            #kde = grid.best_estimator_
            #print(kde.sample())
            #arr[p][:] = kde.sample()
        #return arr


    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a Gaussian distribution based on a configuration file. The
        parameters for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Boundary arguments should be provided in the same way as described in
        `get_param_bounds_from_config`. In addition, the mean and variance of
        each parameter can be specified by setting `{param}_mean` and
        `{param}_var`, respectively. For example, the following would create a
        truncated Gaussian distribution between 0 and 6.28 for a parameter
        called `phi` with mean 3.14 and variance 0.5 that is cyclic:

        .. code-block:: ini

            [{section}-{tag}]
            min-phi = 0
            max-phi = 6.28
            phi_mean = 3.14
            phi_var = 0.5
            cyclic =

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        Gaussain
            A distribution instance from the pycbc.inference.prior module.
        """
        return bounded.bounded_from_config(cls, cp, section, variable_args,
                                                  bounds_required=False)


__all__ = ['GaussianBimodal']

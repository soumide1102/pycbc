# Copyright (C) 2016  Collin Capano
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
This modules provides classes for evaluating uniform distributions.
"""

import numpy
from pycbc.distributions import bounded

class SigmoidSum(bounded.BoundedDist):
    """
    A uniform distribution on the given parameters. The parameters are
    independent of each other. Instances of this class can be called like
    a function. By default, logpdf will be called, but this can be changed
    by setting the class's __call__ method to its pdf method.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.

    Attributes
    ----------
    name : 'sigmoid_sum'
        The name of this distribution.

    Attributes
    ----------
    params : list of strings
        The list of parameter names.
    bounds : dict
        A dictionary of the parameter names and their bounds.
    norm : float
        The normalization of the multi-dimensional pdf.
    lognorm : float
        The log of the normalization.

    Examples
    --------
    Create a 2 dimensional uniform distribution:

    >>> dist = prior.Uniform(mass1=(10.,50.), mass2=(10.,50.))

    Get the log of the pdf at a particular value:

    >>> dist.logpdf(mass1=25., mass2=10.)
        -7.3777589082278725

    Do the same by calling the distribution:

    >>> dist(mass1=25., mass2=10.)
        -7.3777589082278725

    Generate some random values:

    >>> dist.rvs(size=3)
        array([(36.90885758394699, 51.294212757995254),
               (39.109058546060346, 13.36220145743631),
               (34.49594465315212, 47.531953033719454)],
              dtype=[('mass1', '<f8'), ('mass2', '<f8')])

    Initialize a uniform distribution using a boundaries.Bounds instance,
    with cyclic bounds:

    >>> dist = distributions.Uniform(phi=Bounds(10, 50, cyclic=True))

    Apply boundary conditions to a value:

    >>> dist.apply_boundary_conditions(phi=60.)
        {'mass1': array(20.0)}

    The boundary conditions are applied to the value before evaluating the pdf;
    note that the following returns a non-zero pdf. If the bounds were not
    cyclic, the following would return 0:

    >>> dist.pdf(phi=60.)
        0.025
    """
    name = 'sigmoid_sum'
    def __init__(self, steppos=None, **params):
        super(SigmoidSum, self).__init__(**params)
        # compute the norm and save
        # temporarily suppress numpy divide by 0 warning
        numpy.seterr(divide='ignore')
        self._lognorm = 0.0
        self._norm = 1.0
        numpy.seterr(divide='warn')

        for bnds in self._bounds.values():
        a,b = self._bounds.bounds.values()
        invnorm = numpy.log(2.71828**b + 1.0) - 0.5*numpy.log(2.71828**b + 2.68812*10**43) \
                    - numpy.log(2.71828**a + 1.0) - 0.5*numpy.log(2.71828**a + 2.68812*10**43)
        self._norm = 1./invnorm
        self._lognorm = numpy.log(self._norm)

    @property
    def norm(self):
        return self._norm

    @property
    def lognorm(self):
        return self._lognorm

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        pdf = self._norm
        for p in self._params:
            pdf_param = self.get_sigmoidsum_pdf(kwargs[p], self.steppos)
            pdf*=pdf_param
            return float(pdf)
        else:
            return 0.

    @staticmethod
    def get_sigmoidsum_pdf(x, steppos):
        f1 = 1.0/(1.0 + numpy.exp(-1.0*x)) - 0.5*(1.0/(1.0 + numpy.exp(-1.0*(x - steppos))))
        return f1

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        if kwargs in self:
            return numpy.log(self._pdf(**kwargs))
        else:
            return -numpy.inf


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
            a,b = self._bounds
            #a,b = self._bounds[p]
            x = numpy.linspace(a,b,1000)
            pdf_param = self._norm*self.get_sigmoidsum_pdf(x, self.steppos)
            pdf = pdf_param/pdf_param.sum()
            cdf = pdf.cumsum()
            nrl = 100000
            inversecdfbins = nrl
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

    

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            ``VARARGS_DELIM``. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        SigmoidSum
            A distribution instance from the pycbc.inference.prior module.
        """
        return super(SigmoidSum, cls).from_config(cp, section, variable_args,
                     bounds_required=True)


__all__ = ['SigmoidSum']

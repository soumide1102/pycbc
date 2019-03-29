# Copyright (C) 2017 Christopher M. Biwer
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
This modules provides classes for evaluating multi-dimensional constraints.
"""

from pycbc import transforms
from pycbc.io import record
import numpy
from pycbc.conversions import mchirp_from_mass1_mass2

class Constraint(object):
    """ Creates a constraint that evaluates to True if parameters obey
    the constraint and False if they do not.
    """
    name = "custom"
    required_parameters = []
    def __init__(self, constraint_arg, transforms=None, **kwargs):
        self.constraint_arg = constraint_arg
        self.transforms = transforms
        for kwarg in kwargs.keys():
            setattr(self, kwarg, kwargs[kwarg])

    def __call__(self, params):
        """ Evaluates constraint.
        """
        # cast to FieldArray
        if isinstance(params, dict):
            params = record.FieldArray.from_kwargs(**params)
        elif not isinstance(params, record.FieldArray):
           raise ValueError("params must be dict or FieldArray instance")

        # try to evaluate; this will assume that all of the needed parameters
        # for the constraint exists in params
        try:
            out = self._constraint(params)
        except NameError:
            # one or more needed parameters don't exist; try applying the
            # transforms
            params = transforms.apply_transforms(params, self.transforms) \
                     if self.transforms else params
            out = self._constraint(params)
        if isinstance(out, record.FieldArray):
            out = out.item() if params.size == 1 else out
        return out

    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        return params[self.constraint_arg]

class LambdaNofQ(Constraint):
    """ Pre-defined constraint that checks if lambda1/lambda2 values are within a
    a q^n_{-} and q^{n_{0+} + qn_{1+}} boundary.
    """
    name = "lambda_n_of_q"
    required_parameters = ["mass1", "mass2", "n"]
    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        mask = numpy.ones(len(params["mass1"]),dtype=bool)
        for ii in range(len(params["mass1"])):
            #print("ii in lambda_n_of_q", ii)
            mchirp_src = mchirp_from_mass1_mass2(params['mass1'][ii], params['mass2'][ii])/1.009047379555777
            #print("mchirp", mchirp_from_mass1_mass2(params['mass1'][ii], params['mass2'][ii]))
            lamb_ratio = (params["mass2"][ii]/params["mass1"][ii])**params["n"][ii]
            q = params["mass2"][ii]/params["mass1"][ii]
            #print("mchirp_src", mchirp_src)
            if 1.15 < mchirp_src < 1.20:
                if 1.15 < mchirp_src < 1.188:
                    n_minus = 2.40789473684*mchirp_src + 2.72332105263
                    n_0plus = 12.5684210526*mchirp_src - 6.89908421053
                    n_1plus = -6.435*mchirp_src + 6.77594
                elif 1.188 < mchirp_src < 1.20:
                    n_minus = 2.49166666667*mchirp_src + 2.6238
                    n_0plus = 11.5*mchirp_src - 5.6298
                    n_1plus = -3.675*mchirp_src + 3.49706
                mask[ii] = q**(n_0plus + q*n_1plus) <= lamb_ratio <= q**n_minus
            else:
                mask[ii] = True
            #if mask[ii] == False:
            #    print("lamb_ratio=", lamb_ratio)
            #    print("q**(n_0plus + q*n_1plus)=", q**(n_0plus + q*n_1plus))
            #    print("q**n_minus=", q**n_minus)
        #print(mask)    
        return mask
#            if (lamb_ratio <= q**n_minus) and (lamb_ratio >= q**(n_0plus + q*n_1plus)):
#                mask_val = True
#            else:
#                mask_val = False
#                print("lamb_ratio={}".format(lamb_ratio))
#                print("n_minus={}".format(n_minus))
#                print("n_0plus={}".format(n_0plus))
#                print("n_1plus={}".format(n_1plus))
#            mask.append(mask_val)
#    return mask


class LambdaMin(Constraint):
    """ Pre-defined constraint that checks if lambda1/lambda2 values above a causal minimum.
    """
    name = "lambda_min"
    required_parameters = ["mass1", "mass2", "lambdasym", "n"]
    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        mask = numpy.ones(len(params["mass1"]), dtype=bool)
        for ii in range(len(params["mass1"])):
            #print("ii in lambda_min", ii)
            if all( [params['mass1'][ii] > 0.6, params['mass2'][ii] > 0.6, params['mass1'][ii] < 1.9, params['mass2'][ii] < 1.9] ):
                #print("n/2.={}".format(params['n'][ii]/2.))

                lambda1 = (params['lambdasym'][ii])*((params['mass2'][ii]/params['mass1'][ii])**(params['n'][ii]/2.))
                lambda2 = (params['lambdasym'][ii])*((params['mass1'][ii]/params['mass2'][ii])**(params['n'][ii]/2.))
                lambda1_lowlim = numpy.exp(13.42 - 23.01*(params['mass1'][ii]/2.0) + 20.53*(params['mass1'][ii]/2.0)**(2.) - 9.599*(params['mass1'][ii]/2.0)**(3.))
                lambda2_lowlim = numpy.exp(13.42 - 23.01*(params['mass2'][ii]/2.0) + 20.53*(params['mass2'][ii]/2.0)**(2.) - 9.599*(params['mass2'][ii]/2.0)**(3.))
                mask[ii] = (lambda1 > lambda1_lowlim) and (lambda2 > lambda2_lowlim)
                #if mask[ii] == False:
                #    print("lambda1=",lambda1)
                #    print("lambda1_lowlim=",lambda1_lowlim)
                #    print("lambda2=",lambda2)
                #    print("lambda2_lowlim=",lambda2_lowlim)
                

                #if (lambda1 < lambda1_lowlim) or (lambda2 < lambda2_lowlim):
                #    mask_val = False
                #else:
                #    mask_val = True
            #else:
            #    mask_val = True
            #mask.append(mask_val)
        #print(mask)
        return mask

class MtotalLT(Constraint):
    """ Pre-defined constraint that check if total mass is less than a value.
    """
    name = "mtotal_lt"
    required_parameters = ["mass1", "mass2"]

    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        return params["mass1"] + params["mass2"] < self.mtotal

class CartesianSpinSpace(Constraint):
    """ Pre-defined constraint that check if Cartesian parameters
    are within acceptable values.
    """
    name = "cartesian_spin_space"
    required_parameters = ["mass1", "mass2", "spin1x", "spin1y", "spin1z",
                           "spin2x", "spin2y", "spin2z"]

    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        if (params["spin1x"]**2 + params["spin1y"]**2 +
                params["spin1z"]**2)**2 > 1:
            return False
        elif (params["spin2x"]**2 + params["spin2y"]**2 +
                  params["spin2z"]**2)**2 > 1:
            return False
        else:
            return True

class EffectiveSpinSpace(Constraint):
    """ Pre-defined constraint that check if effective spin parameters
    are within acceptable values.
    """
    name = "effective_spin_space"
    required_parameters = ["mass1", "mass2", "q", "xi1", "xi2",
                           "chi_eff", "chi_a"]

    def _constraint(self, params):
        """ Evaluates constraint function.
        """

        # ensure that mass1 > mass2
        if params["mass1"] < params["mass2"]:
            return False

        # constraint for secondary mass
        a = ((4.0 * params["q"]**2 + 3.0 * params["q"])
                 / (4.0 + 3.0 * params["q"]) * params["xi2"])**2
        b = ((1.0 + params["q"]**2) / 4.0
                 * (params["chi_eff"] + params["chi_a"])**2)
        if a + b > 1:
            return False

        # constraint for primary mass
        a = params["xi1"]**2
        b = ((1.0 + params["q"])**2 / (4.0 * params["q"]**2)
                 * (params["chi_eff"] - params["chi_a"])**2)
        if a + b > 1:
            return False

        return True

# list of all constraints
constraints = {
    Constraint.name : Constraint,
    LambdaNofQ.name : LambdaNofQ,
    LambdaMin.name : LambdaMin,
    MtotalLT.name : MtotalLT,
    CartesianSpinSpace.name : CartesianSpinSpace,
    EffectiveSpinSpace.name : EffectiveSpinSpace,
}

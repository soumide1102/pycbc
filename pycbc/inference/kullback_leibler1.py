# Copyright (C) 2017 Soumi De
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
""" This module evaluates the Kullback-Leibler divergence diagnostic statistic.
"""

import numpy
import scipy
from decimal import Decimal

def kullback_leibler(logposterior_array, logprior_array,
                     start, end, step):
    """
    Parameters
    ----------
    posterior_array : numpy.array
        A 1D array of posterior values.
    prior_array : numpy.array
        A 1D array of prior values.
    start : int
        Index in the posterior/prior array where to start calculating the 
        Kullback-Leibler divergence statistic. 
    end : int
        Index in the posterior/prior array where to stop calculating the 
        Kullback-Leibler divergence statistic.
    step : int
        Number of steps to take along the chain before calculating the
        divergence statistic again.

    Returns
    ---------- 
    ends : numpy.array
        A 1D array of indices along the chain where the divergence statistic was
        calculated.
    stats : numpy.array
        A 1D array of values of the Kullback-Leibler divergence statistic
        calculated at the indices along the chain stored in ends.
    """
    
    ends = []
    stats = []

    logposterior_array=logposterior_array/1000
    logprior_array=logprior_array/1000
    posterior_array=numpy.exp(logposterior_array)

    # Divide the chain into blocks between the 'start' and 'end' indices in
    # steps of 'step'
    starts = numpy.arange(start, end, step)

    # Loop over the desired indices along the chain and calculate the
    # KL-divergence stat.
    for ii in range(len(starts)):
        
        # Index in the prior and posterior arrays where to calculate the stat
        end = int(starts[ii] + step)

        # Calculate the KL-divergence stat of the portions of the prior and 
        # posterior distributions between the first index and the ii'th index of
        # the arrays.
        #stat = scipy.stats.entropy(posterior_array[:end],
        #                           prior_array[:end])
     
        logposterior_scaled=logposterior_array[:end]
        logprior_scaled=logprior_array[:end]
        posterior_scaled=posterior_array[:end]
        l = len(posterior_scaled)
        stat = 0.0
        for jj in range(l):
            print("logposterior_scaled[jj]", logposterior_scaled[jj])
            print("logprior_scaled[jj]", logprior_scaled[jj])
            #log_ratio = 1000.0*(logposterior_scaled[jj] - logprior_scaled[jj])
            log_ratio = (logposterior_scaled[jj] - logprior_scaled[jj])
            print("log_ratio", log_ratio)
            print("posterior_scaled[jj]", posterior_scaled[jj])
            posterior = posterior_scaled[jj] + float(numpy.exp(Decimal(1000)))
            print("posterior", posterior)
            stat = stat + posterior*log_ratio
            
                
        # store the values of the stat
        stats.append(stat)

        # store the index where the stat was calculated
        ends.append(end)

    return numpy.array(ends), numpy.array(stats)

        
    


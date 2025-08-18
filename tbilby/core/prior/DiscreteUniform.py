# define model_prior with probability mass function
from bilby.core.prior.base import Prior
from scipy.special import erfinv
import numpy as np
from scipy.special import xlogy

#from scipy.special._ufuncs import xlogy, erf, log1p, stdtrit, gammaln, stdtr, \
#    btdtri, betaln, btdtr, gammaincinv, gammainc

class DiscreteUniform(Prior):
    def __init__(self, minimum, maximum, name=None, latex_label=None,
               unit=None,boundary=None):
        super(DiscreteUniform, self).__init__(name=name, latex_label=latex_label,
                                             minimum=minimum, maximum=maximum, unit=unit,
                                             boundary=boundary)
        
    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        if isinstance(val,int) or isinstance(val,float):
            n=1
            interval= 1 / (self.maximum - self.minimum + 1)
            while(val>=n*interval):
                n += 1

            return self.minimum + n - 1
        
        else:
            sample=np.ones(len(val))
            for i in range(len(val)):
                n=1
                interval= 1 / (self.maximum - self.minimum + 1)
                while(val[i]>=n*interval):
                    n += 1
                sample[i]=self.minimum + n - 1
            return sample
        
        
        
    def prob(self,val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        if isinstance(val,int) or isinstance(val,float):
            if not np.isclose((val % 1), 0):
                return 0
            p=((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum + 1)
            return p
        else:
            p=((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum + 1)
            mask = np.isclose((val % 1), 0)
            p[~mask]=0
            return p

    
    def ln_prob(self, val):
        """Return the log prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: log probability of val
        """
        if isinstance(val,int) or isinstance(val,float):
            if not np.isclose((val % 1), 0):
                return -np.inf
            ln_p=xlogy(1, (val >= self.minimum) & (val <= self.maximum)) - xlogy(1, self.maximum - self.minimum + 1)
            return ln_p
        else:
            ln_p=xlogy(1, (val >= self.minimum) & (val <= self.maximum)) - xlogy(1, self.maximum - self.minimum + 1)
            mask = np.isclose((val % 1), 0)
            ln_p[~mask]=-np.inf

            return ln_p
        
    
    
    def cdf(self, val):
        if isinstance(val,int) or isinstance(val,float):
            n=1
            interval= 1 / (self.maximum - self.minimum + 1)
            if (val < self.minimum):
                return 0

            if (val > self.maximum):
                return 1

            while(val>= n + self.minimum):
                n += 1           
            return interval*n
        else:
            p=np.ones(len(val))
            for i in range(len(val)):
                n=1
                interval= 1 / (self.maximum - self.minimum + 1)
                if (val[i] < self.minimum):
                    p[i] = 0

                elif (val[i] > self.maximum):
                    p[i] = 1

                else:
                    while(val[i]>= n + self.minimum):
                        n += 1           
                        p[i] = interval*n
            return p

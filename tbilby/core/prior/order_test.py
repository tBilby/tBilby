import bilby 
import math
import numpy as np
import scipy
import tbilby
from scipy.stats import norm,beta
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
# np.math.factorial = np.vectorize(np.math.factorial)

class AscendingOrderStatPrior(bilby.prior.Prior):

    def __init__(self, prev_val,this_order_num,minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Uniform prior with bounds

        Parameters
        ==========
        prev_val:
            last SNR
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        
        self._prev_val=prev_val
        
        self._this_order_num=this_order_num
        # self._tot_order_num=tot_order_num

        super(AscendingOrderStatPrior, self).__init__(name=name, latex_label=latex_label,
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
        
        res=[]
        size=False
        if not isinstance(val, np.ndarray):
            size=True
            val_est=np.array([val])
            self._tot_order_num=np.array([self._tot_order_num])
        elif not isinstance(self._tot_order_num, np.ndarray):
            self._tot_order_num=np.array([self._tot_order_num])
            val_est=val.copy() 
        else:
            val_est=val.copy()
        for i, v in enumerate(val_est):
            if self._this_order_num <= self._tot_order_num[i]:
                # draw samples from ordering stat prior
                if isinstance(self._prev_val, np.ndarray):
                    prev_val= self._prev_val[i]
                else:
                    prev_val= self._prev_val    
                res.append(self.normalized_conditional_icdf(v, (prev_val- self.minimum)/(self.maximum - self.minimum), self._tot_order_num[i], self._this_order_num)*(self.maximum - self.minimum)+self.minimum)
            else:
                # ghost parameters, draw from unifrom[minimum, maximum]
                res.append(v*(self.maximum-self.minimum)+self.minimum)
        if size:
            return np.array(res)[0]
        return np.array(res)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        
        
        """
        # val /= (self.maximum-self.minimum)
        if (not isinstance(val, np.ndarray)) and (not isinstance(val, pd.Series)):
            if self._this_order_num <= self._tot_order_num:
                # ordering stats prior
                return self.normalized_pdf_order_statistics((self._prev_val-self.minimum)/(self.maximum-self.minimum), 
                                                    (val-self.minimum)/(self.maximum-self.minimum),
                                                    self._this_order_num-1, 
                                                    self._this_order_num, 
                                                    self._tot_order_num)/(self.maximum-self.minimum)
            else:
                return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)
        elif (not isinstance(self._tot_order_num, np.ndarray)) and (not isinstance(self._tot_order_num, pd.Series)):
            self._tot_order_num=np.array([self._tot_order_num])
        if (not isinstance(self._prev_val, np.ndarray)) and (not isinstance(self._prev_val, pd.Series)):
            self._prev_val=np.array([self._prev_val])
        mask = self._this_order_num <= self._tot_order_num
        prob=np.ones(self._tot_order_num.shape)
        if sum(mask) > 0:
            prob[mask] *= self.normalized_pdf_order_statistics((self._prev_val[mask]-self.minimum)/(self.maximum-self.minimum), 
                                        (val[mask]-self.minimum)/(self.maximum-self.minimum),
                                        self._this_order_num-1, 
                                        self._this_order_num, 
                                        self._tot_order_num[mask])/(self.maximum-self.minimum)
        if sum(~mask)>0:
            prob[~mask] *= ((val[~mask] >= self.minimum) & (val[~mask] <= self.maximum)) / (self.maximum - self.minimum)
        return prob
    
    def ln_prob(self, val):
        return np.log(self.prob(val))
    
    def normalized_pdf_order_statistics(self, u, v, this_order_num, n): # should work on normalize u and v 
        i = n-this_order_num+1
        j = i+1
        if this_order_num==1:
            return self.beta_dist(u, i, n+1-i)
        try:
            n_factorial = np.math.factorial(n)
            i_minus_1_factorial = np.math.factorial(i - 1)
            j_minus_i_minus_1_factorial = np.math.factorial(j - i - 1)
            n_minus_j_factorial = np.math.factorial(n - j)
        except TypeError:
            n_factorial = np.vectorize(np.math.factorial)(n)
            i_minus_1_factorial = np.vectorize(np.math.factorial)(i - 1)
            j_minus_i_minus_1_factorial = np.vectorize(np.math.factorial)(j - i - 1)
            n_minus_j_factorial = np.vectorize(np.math.factorial)(n - j)
        up = (n_factorial * (u ** (i - 1)) * ((v - u) ** (j - i - 1)) * ((1 - v) ** (n - j)))
        down = (i_minus_1_factorial * j_minus_i_minus_1_factorial * n_minus_j_factorial)
        pdf_value = up  / down 
        if isinstance(v, np.ndarray): # get ride of not order things
            pdf_value[u>=v]=0
        
        return pdf_value/self.beta_dist(v, j, n+1-j)

    def beta_dist(self,x, alpha, beta):
        # when alpha and beta are integers
        try:
            value = x**(alpha-1)*(1-x)**(beta-1)*np.math.factorial(alpha+beta-1)/np.math.factorial(alpha-1)/np.math.factorial(beta-1)
        except TypeError:
            value = x**(alpha-1)*(1-x)**(beta-1)*np.vectorize(np.math.factorial)(alpha+beta-1)/np.math.factorial(alpha-1)/np.vectorize(np.math.factorial)(beta-1)
        return value
    def beta_inc(self,x, a, b):
        #incomplete beta function when a and b are int
        value = scipy.special.betainc(a, b, x)
        return value * np.math.factorial(a-1) * np.math.factorial(b-1) / np.math.factorial(a+b-1)
   
    def nomalized_conditional_cdf(self,x, u, n, i, j):
        # for normalized x and u
        if self._this_order_num==1:
            # this is just the beta distribution 
            return scipy.stats.beta.cdf(x, j,n+1-j)
        
        # cdf of \pi(v|u) with v>u
        value = ((1-u)**(-i+n)* u**(-1+i)*self.beta_inc((u-x)/(-1+u), -i+j, 1-j+n)*np.math.factorial(n))/\
        (np.math.factorial(i-1)*np.math.factorial(-i+j-1)*np.math.factorial(1-j+n-1))
        return value/self.beta_dist(u, i, n+1-i)

    def normalized_conditional_icdf(self, y, u, n, this_order_num):

        i = this_order_num - 1
        j = i+1
        if this_order_num==1:
            # the first one is the largest
            # this is just the beta distribution
            return scipy.special.betaincinv(j,n+1-j,y)
        
        # cdf of \pi(u|v) with v>u
        value = y * self.beta_dist(u, i, n+1-i)*np.math.factorial(i-1)*np.math.factorial(n-i)/((1-u)**(-i+n)*u**(-1+i)*np.math.factorial(n))

        return u - scipy.special.betaincinv(-i+j, 1-j+n, value) * (-1+u)
    
class ConditionalAscendingOrderStatPrior(bilby.prior.conditional.conditional_prior_factory(AscendingOrderStatPrior)):
    pass    


class TransdimensionalConditionalAscendingOrderStatPrior(tbilby.core.prior.transdimensional_conditional_prior_factory(ConditionalAscendingOrderStatPrior)):
    pass

class DescendingOrderStatPrior(bilby.prior.Prior):

    def __init__(self, prev_val,this_order_num,minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Uniform prior with bounds

        Parameters
        ==========
        prev_val:
            last SNR
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        
        self._prev_val=prev_val
        
        self._this_order_num=this_order_num
        # self._tot_order_num=tot_order_num

        super(DescendingOrderStatPrior, self).__init__(name=name, latex_label=latex_label,
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
        
        res=[]
        size=False
        if not isinstance(val, np.ndarray):
            val_est=np.array([val])
            size=True #sample()
            self._tot_order_num=np.array([self._tot_order_num])
        elif not isinstance(self._tot_order_num, np.ndarray):
            self._tot_order_num=np.array([self._tot_order_num])
            val_est=val.copy()
        else:
            val_est=val.copy()    
        for i, u in enumerate(val_est):
            if self._this_order_num <= self._tot_order_num[i]:
                # draw samples from ordering stat prior
                if isinstance(self._prev_val, np.ndarray):
                    prev_val= self._prev_val[i]
                else:
                    prev_val= self._prev_val
                res.append(self.normalized_conditional_icdf(u,(prev_val- self.minimum)/(self.maximum - self.minimum), self._tot_order_num[i], self._this_order_num)*(self.maximum - self.minimum)+self.minimum)
            else:
                # ghost parameters, draw from unifrom[minimum, maximum]
                res.append(u*(self.maximum-self.minimum)+self.minimum)
        if size:
            return np.array(res)[0]
        return np.array(res)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        
        
        """
        if (not isinstance(val, np.ndarray)) and (not isinstance(val, pd.Series)):
            if self._this_order_num <= self._tot_order_num:
                # ordering stats prior
                return self.normalized_pdf_order_statistics((val-self.minimum)/(self.maximum-self.minimum),
                                                    (self._prev_val-self.minimum)/(self.maximum-self.minimum), 
                                                    self._this_order_num, 
                                                    self._tot_order_num)/(self.maximum-self.minimum)
            else:
                return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)
        elif (not isinstance(self._tot_order_num, np.ndarray)) and (not isinstance(self._tot_order_num, pd.Series)):
            self._tot_order_num=np.array([self._tot_order_num])
        if (not isinstance(self._prev_val, np.ndarray)) and (not isinstance(self._prev_val, pd.Series)):
            self._prev_val=np.array([self._prev_val])
        mask = self._this_order_num <= self._tot_order_num
        prob=np.ones(self._tot_order_num.shape)
        if sum(mask) > 0:
            prob[mask] *= self.normalized_pdf_order_statistics((val[mask]-self.minimum)/(self.maximum-self.minimum),
                                                    (self._prev_val[mask]-self.minimum)/(self.maximum-self.minimum), 
                                                    self._this_order_num, 
                                                    self._tot_order_num[mask])/(self.maximum-self.minimum)
        if sum(~mask) > 0:
            prob[~mask] *= ((val[~mask] >= self.minimum) & (val[~mask] <= self.maximum)) / (self.maximum - self.minimum)
        return prob

    def ln_prob(self, val):
        return np.log(self.prob(val))
    
    def normalized_pdf_order_statistics(self, u, v, this_order_num, n): # should work on normalize u and v 
        i = n-this_order_num+1
        j = i+1
        if this_order_num==1:
            return self.beta_dist(u, i, n+1-i)
        try:
            n_factorial = np.math.factorial(n)
            i_minus_1_factorial = np.math.factorial(i - 1)
            j_minus_i_minus_1_factorial = np.math.factorial(j - i - 1)
            n_minus_j_factorial = np.math.factorial(n - j)
        except TypeError:
            n_factorial = np.vectorize(np.math.factorial)(n)
            i_minus_1_factorial = np.vectorize(np.math.factorial)(i - 1)
            j_minus_i_minus_1_factorial = np.vectorize(np.math.factorial)(j - i - 1)
            n_minus_j_factorial = np.vectorize(np.math.factorial)(n - j)
        up = (n_factorial * (u ** (i - 1)) * ((v - u) ** (j - i - 1)) * ((1 - v) ** (n - j)))
        down = (i_minus_1_factorial * j_minus_i_minus_1_factorial * n_minus_j_factorial)
        pdf_value = up  / down 
        if isinstance(v, np.ndarray): # get ride of not order things
            pdf_value[u>=v]=0
        
        return pdf_value/self.beta_dist(v, j, n+1-j)

    def beta_dist(self,x, alpha, beta):
        # when alpha and beta are integers
        try:
            value = x**(alpha-1)*(1-x)**(beta-1)*np.math.factorial(alpha+beta-1)/np.math.factorial(alpha-1)/np.math.factorial(beta-1)
        except TypeError:
            value = x**(alpha-1)*(1-x)**(beta-1)*np.vectorize(np.math.factorial)(alpha+beta-1)/np.vectorize(np.math.factorial)(alpha-1)/np.vectorize(np.math.factorial)(beta-1)
        return value
    def beta_inc(self,x, a, b):
        #incomplete beta function when a and b are int
        value = scipy.special.betainc(a, b, x)
        return value * np.math.factorial(a-1) * np.math.factorial(b-1) / np.math.factorial(a+b-1)
   
    def nomalized_conditional_cdf(self, x, v, n, this_order_num):
        # for normalized x and u
        i = n-this_order_num+1
        j = i+1
        if this_order_num==1:
            # the first one is the largest
            # this is just the beta distribution
            return scipy.stats.beta.cdf(x, i,n+1-i)
        
        # cdf of \pi(u|v) with v>u
        value = ((1-v)**(-j+n)*v**(-1+j)*scipy.special.betainc(i, -i+j, x/v)*np.math.factorial(n))/\
        (np.math.factorial(j-1)*np.math.factorial(1-j+n-1))
        return value/self.beta_dist(v, j, n+1-j)

    def normalized_conditional_icdf(self, y, v, n, this_order_num):

        i = n-this_order_num+1
        j = i+1
        if this_order_num==1:
            # the first one is the largest
            # this is just the beta distribution
            return scipy.special.betaincinv(i,n+1-i,y)
        
        # cdf of \pi(u|v) with v>u
        value = y * self.beta_dist(v, j, n+1-j)*np.math.factorial(j-1)*np.math.factorial(1-j+n-1)/((1-v)**(-j+n)*v**(-1+j)*np.math.factorial(n))

        return scipy.special.betaincinv(i, -i+j, value) * v

class ConditionalDescendingOrderStatPrior(bilby.prior.conditional.conditional_prior_factory(DescendingOrderStatPrior)):
    pass

class TransdimensionalConditionalDescendingOrderStatPrior(tbilby.core.prior.transdimensional_conditional_prior_factory(ConditionalDescendingOrderStatPrior)):
    pass
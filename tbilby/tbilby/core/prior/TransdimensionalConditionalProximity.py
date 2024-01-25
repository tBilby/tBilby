from bilby.core.prior import Prior, Uniform, ConditionalBeta, Beta, ConditionalUniform, ConditionalBasePrior, LogUniform
from bilby.core.prior import ConditionalLogUniform, Constraint, DeltaFunction
from scipy.interpolate import interp1d
from scipy.special._ufuncs import xlogy
import numpy as np

from bilby.core.utils import infer_parameters_from_function
import bilby
import pandas as pd

from bilby.core.prior import PriorDict
from bilby.core.utils import infer_args_from_method, infer_parameters_from_function
from bilby.core.prior import conditional_prior_factory
from scipy.special import erf, erfinv



 



class ConditionalUniformReveredGaussian(ConditionalBasePrior):
    # this must be defined in order for us to know what is the name of the parameters inside of the class are allowed to change dimensionality  
    tparams = ['mu'] 
    

    def __init__(self, condition_func, name=None, latex_label=None, unit=None, boundary=None, **reference_params):
        # reference_params: mu[0..N], sigma[0....N] minimum, maximum
        
        """UniformReveredGaussian

        Parameters
        ==========
        mu X N : float
            Mean of the Gaussian prior
        constant sigma:
            Width/Standard deviation of the Gaussian 
        
        
        minimum: float
            
        maximum: float
            
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(ConditionalUniformReveredGaussian, self).__init__(condition_func,name=name, latex_label=latex_label, unit=unit,
                                                                    boundary=boundary, minimum=reference_params['minimum'], maximum=reference_params['maximum'])
        
        self.__class__.__name__ = 'ConditionalUniformReveredGaussian'
        self.__class__.__qualname__ = 'ConditionalUniformReveredGaussian'

        self._reference_params = reference_params
        self.sigma= reference_params['sigma']
        
        for key, value in reference_params.items():
            setattr(self, key, value)
       
    def set_n_fix_mu(self,val=None):
        
        # for mu the first axis is the other mus 
        # the second axis is the different samples 
        # use this function to order things according to the request samples number 
        if val!=None: # this is the probability we have to work hard to set things right     
            if isinstance(val,float):
                val=[val]
            sz = len(val)
            num_of_dependent_params = 0 
            params_list = infer_parameters_from_function(self.condition_func)
            param_name = ''.join([i for i in self.name if not i.isdigit()])
            for p in params_list: 
                result = ''.join([i for i in p if not i.isdigit()])
                if result == param_name:
                    num_of_dependent_params +=1
        # make sure we know what is going on 
            self.mu=self.mu.reshape((num_of_dependent_params,sz))
            self.n= num_of_dependent_params	
        if hasattr(self,'mu') and len(self.mu) > 0 and val==None: # this is teh case for rescale which works well
            
                        
            if len(self.mu.shape)==1:
                self.n = 1   
                self.mu = self.mu.reshape(1,-1)
            if len(self.mu.shape)==2:
                self.n= self.mu.shape[0]
            
        else:             
            self.n =0 
            
       

            
    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.update_conditions(**required_variables)
        #self.n = len(required_variables['mu'])  if 'mu' in required_variables else 0  
        self.set_n_fix_mu() 
        #try:
        #print(val)
        if isinstance(val,float):
            val=[val]
        sz = len(val)
        samples=np.zeros(sz)
        #define inverse_cdf
        x = np.linspace(self.minimum, self.maximum, 10000).reshape(-1,1)            
        x_for_cdf = np.tile(x,(1,sz))
        cdf = self.cdf(x_for_cdf)
        
        # numerical claculation of the inverse cdf numerically 
             
        for i in range(sz):
            
            samples[i]=interp1d(cdf[:,i], x.reshape(-1,))(val[i])
      
        return samples
        #except:
            # this works for teh full sample togetther - shoudnt work here ....
            #define inverse_cdf
        #    x = np.linspace(self.minimum, self.maximum, 10000)
        #    cdf = self.cdf(x)
           
        #    sample=interp1d(cdf, x)(val)

        #    return sample
              

    def prob(self, val, **required_variables):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        self.update_conditions(**required_variables)
        #self.n = len(required_variables['mu'])  if 'mu' in required_variables else 0  
        self.set_n_fix_mu(val=val) 
        
        if self.n==0: # just uniform prior 
            return (1/(self.maximum -self.minimum))* self.is_in_prior_range(val)
        
        
        
        # the PDF = C - sum_n(N(mu_n,sigma))
        # such that integral PDF = 1 = > C (max-min) = 1 + sum_n (integral_a^b (N(mu_n,sigma)))
        # lets calculate C horray
        prob=0 
        C = 1
        for i in range(self.n):
            upper_limit_arg = (self.maximum - self.mu[i]) / (self.sigma * np.sqrt(2))
            lower_limit_arg = (self.minimum - self.mu[i]) / (self.sigma * np.sqrt(2))
            C += 0.5 *( erf(upper_limit_arg) - erf(lower_limit_arg) )
            # calculat ethe actuall probability             
            prob -= (np.exp(-((val -  self.mu[i]) / self.sigma)**2/2.0) / np.sqrt(2.0*np.pi) / self.sigma)
        
        C /= (self.maximum - self.minimum)
        prob+=C
        
        
       

                
        probs = prob * self.is_in_prior_range(val)# / norm_const       
        if len(probs)==1:
            probs=probs[0]# turn into scalar otherwise keep as array 
        
        return  probs

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables)
        #self.n = len(required_variables['mu'])  if 'mu' in required_variables else 0    
        self.set_n_fix_mu() 
            
        if self.n==0: # just linear 
            cdf = (val-self.minimum)/(self.maximum -self.minimum)
        else:
            
            C = 1
            cdf =0 
            for i in range(self.n):
                upper_limit_arg = (self.maximum - self.mu[i]) / (self.sigma * np.sqrt(2))
                cdf_upper_limit_arg = (val - self.mu[i].reshape(1,-1)) / (self.sigma * np.sqrt(2))
                lower_limit_arg = (self.minimum - self.mu[i]) / (self.sigma * np.sqrt(2))
                C += 0.5 *( erf(upper_limit_arg) - erf(lower_limit_arg) )
                cdf -= 0.5 *( erf(cdf_upper_limit_arg) - erf(lower_limit_arg).reshape(1,-1) )
                # calculat ethe actuall probability             
                
            C /= (self.maximum - self.minimum)
            cdf += C.reshape(1,-1)*(val- self.minimum)
            
            
            # frac_cdf = (val -self.minimum)/(np.sqrt(2.0*np.pi) / self.sigma)
            # inv_norm_const = (self.maximum -self.minimum)/(np.sqrt(2.0*np.pi) / self.sigma)
            # for i in range(self.n):
            #      gaus = 0.5 * (erf((val - self.mu[i]) / (self.sigma * 2**0.5)) - erf((self.minimum - self.mu[i]) / (self.sigma * 2**0.5)))
            #      frac_cdf-= gaus  
            #      gausFull = 0.5 * (erf((self.maximum - self.mu[i]) / (self.sigma * 2**0.5)) - erf((self.minimum - self.mu[i]) / (self.sigma * 2**0.5)))
            #      inv_norm_const-= gausFull
            # cdf = frac_cdf/inv_norm_const
               
        try:
            cdf[val > self.maximum] = 1
            cdf[val < self.minimum] = 0
        except:
            if val > self.maximum:
                cdf = 1
            elif val < self.minimum:
                cdf = 0
        return cdf

    

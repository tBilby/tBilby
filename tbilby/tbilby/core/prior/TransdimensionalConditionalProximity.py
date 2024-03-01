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
        if val is not None: # this is the probability we have to work hard to set things right     
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
        x,h = np.linspace(self.minimum, self.maximum, 10000).reshape(-1,1)            
        x_for_cdf = np.tile(x,(1,sz))
        cdf = self.cdf(x_for_cdf)
        
        # numerical claculation of the inverse cdf numerically 
             
        for i in range(sz):
                    
            samples[i]=interp1d(cdf[:,i], x.reshape(-1,))(val[i])
      
        if len(samples)==1:
            samples=samples[0]
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

    


class MarginalizedTruncatedHollowedGaussianOld(ConditionalBasePrior):

    def __init__(self, condition_func, name=None, latex_label=None, unit=None, boundary=None, **reference_params):
        # reference_params: alpha, beta, sigma_t, sigma_f, minimum_t, maximum_t, minimum_f, maximum_f, index
        # condition_func: t0, f0 ... t_{n-1}, f_{n-1}
        """Truncated Hollow Gaussian prior with mean mu, width alpha and hollow width beta

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        alpha:
            Width/Standard deviation of the Gaussian prior
        beta:
            With of the hollowed out portion of the Gaussian
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
        super(MarginalizedTruncatedHollowedGaussian, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                                                    boundary=boundary, condition_func = condition_func,
                                                                    minimum=reference_params['minimum_t'], maximum=reference_params['maximum_t'])
        
        self.__class__.__name__ = 'MarginalizedTruncatedHollowedGaussian'
        self.__class__.__qualname__ = 'MarginalizedTruncatedHollowedGaussian'
        self._reference_params = reference_params
        x = np.linspace(self._reference_params['minimum_t'], self._reference_params['maximum_t'], 10000)
        
        
        self.x=x   
        self.xx=np.nan
          
        
        

        
        for key, value in reference_params.items():
            setattr(self, key, value)
        self._normalisation = None

                
    @property
    def normalisation(self):
        """ Calculates the proper normalisation of the truncated hollowed Gaussian

        Returns
        =======
        float: Proper normalisation of the truncated hollowed Gaussian
        """
        return self._normalisation
        
    # def inverse_cdf(self,val):
    #     return self._inverse_cdf(val)
    
    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.update_conditions(**required_variables)
        try:

            samples=np.zeros(len(val))
            #define inverse_cdf
            
            
            if np.isnan(self.xx ) or self.xx.shape[1] != len(self.f[0]): # do this tsep only if we have to 
                self.xx = np.zeros((len(self.x),len(self.f[0])))
                for i in range(len(self.f[0])):
                    self.xx[:,i] = self.x 
            

            cdf = self.cdf(self.xx)

            for i in range(len(self.f[0])):
                samples[i]=interp1d(cdf[:,i], self.x)(val[i])

            return samples
        except:

            #define inverse_cdf            
            cdf = self.cdf(self.x)
            sample=interp1d(cdf, self.x)(val)

            return sample
        
        #return erfinv(2 * val * self.normalisation + erf(
        #    (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu
    def _fix_t_array(self):
        self.n = len(self.f)
        if (len(self.t)!=len(self.f)):# should be fixed 
            if len(self.t)==0:
                self.t = np.zeros(self.f[0].shape).reshape(1,-1)
            else:    
                self.t = np.concatenate((np.zeros(self.f[0].shape).reshape(1,-1),self.t),axis=0)
        
        
        
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
        probs = np.zeros(self.f[0].shape)
        self._fix_t_array()
       
        
        if self.n==1: # meaning we are teh first t
           return 1/(self.maximum_t-self.minimum_t) # return uniform 
        
        normalisation = np.zeros(self.f.shape)
        # add in the start of t
        
        
        for i in range(self.n):
     
            normalisation[i] = (self.alpha**2 * \
                       (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                        erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                       (erf((-self.t[i] + self.maximum_t)/(self.alpha * self.sigma_t)) + \
                        erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                        self.beta**2 * \
                       (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                        erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                       (erf((-self.t[i] + self.maximum_t)/(self.beta * self.sigma_t)) + \
                        erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                       (4 * (self.alpha**2 - self.beta**2))
       
        
        for i in range(self.n):
            probs += (np.exp(-(val-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                             self.alpha*(erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                                         erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                             np.exp(-(val-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                             self.beta*(erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                                        erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                             / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) \
                             / normalisation[i]
                
        probs = probs * self.is_in_prior_range(val) / self.n        
        
        return  probs

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables)
       
        self._fix_t_array() 
        
        if self.n==1: # meaning we are teh first t , return uniform 
           return (val - self.minimum_t)/(self.maximum_t - self.minimum_t)
        
        
        normalisation = np.zeros(self.f.shape)
        _cdf = np.zeros(val.shape)
        for i in range(self.n):
    
           normalisation[i] = (self.alpha**2 * \
                      (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                       erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                      (erf((-self.t[i] + self.maximum_t)/(self.alpha * self.sigma_t)) + \
                       erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                       self.beta**2 * \
                      (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                       erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                      (erf((-self.t[i] + self.maximum_t)/(self.beta * self.sigma_t)) + \
                       erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                      (4 * (self.alpha**2 - self.beta**2))
       
       
        for i in range(self.n):
                  est = (self.alpha**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.alpha * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                            self.beta**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.beta * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                           (4 * (self.alpha**2 - self.beta**2))
                        
                        
                  _cdf +=   est/normalisation[i]   
                        
        _cdf /= self.n
        try:
            _cdf[val > self.maximum] = 1
            _cdf[val < self.minimum] = 0
        except:
            if val > self.maximum:
                _cdf = 1
            elif val < self.minimum:
                _cdf = 0
        return _cdf
    
    
class MarginalizedTruncatedHollowedGaussian(ConditionalBasePrior):

    def __init__(self, condition_func, name=None, latex_label=None, unit=None, boundary=None, **reference_params):
        # reference_params: alpha, beta, sigma_t, sigma_f, minimum_t, maximum_t, minimum_f, maximum_f, index
        # condition_func: t0, f0 ... t_{n-1}, f_{n-1}
        """Truncated Hollow Gaussian prior with mean mu, width alpha and hollow width beta

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        alpha:
            Width/Standard deviation of the Gaussian prior
        beta:
            With of the hollowed out portion of the Gaussian
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
        super(MarginalizedTruncatedHollowedGaussian, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                                                    boundary=boundary, condition_func = condition_func,
                                                                    minimum=reference_params['minimum_t'], maximum=reference_params['maximum_t'])
        
        self.__class__.__name__ = 'MarginalizedTruncatedHollowedGaussian'
        self.__class__.__qualname__ = 'MarginalizedTruncatedHollowedGaussian'
        self._reference_params = reference_params
        x = np.linspace(self._reference_params['minimum_t'], self._reference_params['maximum_t'], 10000)
        
        
        self.x=x   
        self.xx=np.nan
          
        self.add_uniform=True
        self.gamma=minimum=reference_params['gamma']
        

        
        for key, value in reference_params.items():
            setattr(self, key, value)
        self._normalisation = None

                
    @property
    def normalisation(self):
        """ Calculates the proper normalisation of the truncated hollowed Gaussian

        Returns
        =======
        float: Proper normalisation of the truncated hollowed Gaussian
        """
        return self._normalisation
        
    # def inverse_cdf(self,val):
    #     return self._inverse_cdf(val)
    
    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.update_conditions(**required_variables)
        try:

            samples=np.zeros(len(val))
            #define inverse_cdf
            
            
            if np.isnan(self.xx ) or self.xx.shape[1] != len(self.f[0]): # do this tsep only if we have to 
                self.xx = np.zeros((len(self.x),len(self.f[0])))
                for i in range(len(self.f[0])):
                    self.xx[:,i] = self.x 
            

            cdf = self.cdf(self.xx)

            for i in range(len(self.f[0])):
                samples[i]=interp1d(cdf[:,i], self.x)(val[i])

            return samples
        except:

            #define inverse_cdf            
            cdf = self.cdf(self.x)
            sample=interp1d(cdf, self.x)(val)

            return sample
        
        #return erfinv(2 * val * self.normalisation + erf(
        #    (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu
    def _fix_f_array(self):
        if isinstance(self.f,float):
            # turn this into array 
            self.f = np.array([self.f]).reshape((1,1))
    def _fix_t_array(self):
        self.n = len(self.t)
        # if (len(self.t)!=len(self.f)):# should be fixed 
        #     if len(self.t)==0:
        #         self.t = np.zeros(self.f[0].shape).reshape(1,-1)
        #     else:    
        #         self.t = np.concatenate((np.zeros(self.f[0].shape).reshape(1,-1),self.t),axis=0)
        
        
        
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
        self._fix_f_array()
        self._fix_t_array()
        #print(self.f)
        probs = np.zeros(self.f[0].shape)
        
       
        
        if self.n==0: # meaning we are teh first t
           return 1/(self.maximum_t-self.minimum_t) # return uniform 
        
        normalisation = np.zeros(self.f.shape)
        # add in the start of t
        
        
        for i in range(self.n):
     
            normalisation[i] = (self.alpha**2 * \
                       (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                        erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                       (erf((-self.t[i] + self.maximum_t)/(self.alpha * self.sigma_t)) + \
                        erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                        self.beta**2 * \
                       (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                        erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                       (erf((-self.t[i] + self.maximum_t)/(self.beta * self.sigma_t)) + \
                        erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                       (4 * (self.alpha**2 - self.beta**2))
       
        
        for i in range(self.n):
            probs += (np.exp(-(val-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                             self.alpha*(erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                                         erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                             np.exp(-(val-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                             self.beta*(erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                                        erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                             / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) \
                             / normalisation[i]
                
        probs = probs * self.is_in_prior_range(val) / self.n        
        print('dt prob ')
        print(probs)
        if self.add_uniform:
            self.gamma/(self.maximum_t-self.minimum_t) + (1-self.gamma)*probs 
            
        
        
        return  probs

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables)
       
        self._fix_t_array() 
        
        if self.n==0: # meaning we are teh first t , return uniform 
           return (val - self.minimum_t)/(self.maximum_t - self.minimum_t)
        
        
        normalisation = np.zeros(self.f.shape)
        _cdf = np.zeros(val.shape)
        for i in range(self.n):
    
           normalisation[i] = (self.alpha**2 * \
                      (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                       erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                      (erf((-self.t[i] + self.maximum_t)/(self.alpha * self.sigma_t)) + \
                       erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                       self.beta**2 * \
                      (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                       erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                      (erf((-self.t[i] + self.maximum_t)/(self.beta * self.sigma_t)) + \
                       erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                      (4 * (self.alpha**2 - self.beta**2))
       
       
        for i in range(self.n):
                  est = (self.alpha**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.alpha * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                            self.beta**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.beta * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                           (4 * (self.alpha**2 - self.beta**2))
                        
                        
                  _cdf +=   est/normalisation[i]   
                        
        _cdf /= self.n
        try:
            _cdf[val > self.maximum] = 1
            _cdf[val < self.minimum] = 0
        except:
            if val > self.maximum:
                _cdf = 1
            elif val < self.minimum:
                _cdf = 0
        
        
        
        if self.add_uniform:
            _cdf = self.gamma*(val-self.minimum_t)/(self.maximum_t-self.minimum_t) + (1-self.gamma)*_cdf 
        
        
        
        return _cdf



class ConditionalTruncatedHollowedGaussian(ConditionalBasePrior):
    def __init__(self, condition_func, name=None, latex_label=None, unit=None, boundary=None, **reference_params):
        #reference_params:, alpha, beta, sigma_t, sigma_f, minimum_t, maximum_t, minimum_f, maximum_f, index#
        # condition_fuc: t0, f0, ... t_{n-1}, f_{n-1}, t_n
        """Truncated Hollow Gaussian prior with mean mu, width alpha and hollow width beta

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        alpha:
            Width/Standard deviation of the Gaussian prior
        beta:
            With of the hollowed out portion of the Gaussian
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
        super(ConditionalTruncatedHollowedGaussian, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                                                   boundary=boundary, condition_func = condition_func,
                                                                   maximum = reference_params['maximum_f'], minimum=reference_params['minimum_f'] )
        self.__class__.__name__ = 'ConditionalTruncatedHollowedGaussian'
        self.__class__.__qualname__ = 'ConditionalTruncatedHollowedGaussian'
        self._reference_params = reference_params
        x = np.linspace(self._reference_params['minimum_f'], self._reference_params['maximum_f'], 10000)
        self.x=x
        self.xx=np.nan
        
        self.add_uniform=True
        self.gamma=minimum=reference_params['gamma']
        
        for key, value in reference_params.items():
            setattr(self, key, value)

    def prob(self, val, **required_variables):
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        
        # the first prior is uniform and on top of that we include, this yet not a 2D distribution  
        self.n = len(self.f)
        if self.n==0:
            return 1/(self.maximum_f-self.minimum_f)
        
        else:
            
            probs_2d = np.zeros(self.f[0].shape)
            
            prob_t = np.zeros(self.f.shape)
            for i in range(self.n):
                
                prob_t[i] =(np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                 self.alpha*(erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                             erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                 np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                 self.beta*(erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                 / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t)
            
            
       
            for i in range(self.n):
                probs_2d += (np.exp(-((val-self.f[i])**2 / self.sigma_f**2 + (self.t[-1]-self.t[i])**2/ self.sigma_t**2)/ self.alpha**2) - \
                           np.exp(-((val-self.f[i])**2 / self.sigma_f**2 + (self.t[-1]-self.t[i])**2/ self.sigma_t**2)/ self.beta**2)) / \
                           ( np.pi * self.sigma_t * self.sigma_f * (self.alpha**2 - self.beta**2)) / prob_t[i]
        
        
        probs_2d = probs_2d * self.is_in_prior_range(val) / self.n
        
        
        if self.add_uniform:
            self.gamma/(self.maximum_f-self.minimum_f) + (1-self.gamma)*probs_2d
        
        return probs_2d

 
    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        try:
            samples=np.zeros(len(val))

            #define inverse_cdf
            
            
            
            if np.isnan(self.xx ) or self.xx.shape[1] != len(self.t[0]): # do this tsep only if we have to 
                self.xx = np.zeros((len(self.x),len(self.t[0])))
                for i in range(len(self.t[0])):
                    self.xx[:,i] = self.x 
                        
            
            cdf = self.cdf(self.xx)

            for i in range(len(self.t[0])):
                samples[i]=interp1d(cdf[:,i], self.x)(val[i])

            return samples
        except:

            #define inverse_cdf
           
            cdf = self.cdf(self.x)
            sample=interp1d(cdf, self.x)(val)

            return sample

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        
        self.n = len(self.f)
        
        if self.n==0:
            return (val - self.minimum_f)/(self.maximum_f - self.minimum_f)
        
        else:
            _cdf = np.zeros(val.shape)            
            prob_t = np.zeros(self.f.shape)
            for i in range(self.n):
                
                prob_t[i] =(np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                 self.alpha*(erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                             erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                 np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                 self.beta*(erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                 / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t)
            
            
            for i in range(self.n):
                _cdf += (np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                     self.alpha*(erf((-self.f[i] + val)/(self.alpha * self.sigma_f)) + \
                                 erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                     np.exp(-(self.t[self.n-1]-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                     self.beta*(erf((-self.f[i] + val)/(self.beta * self.sigma_f)) + \
                                erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                     / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) / prob_t[i]
        
        _cdf /= self.n
        try:
            _cdf[val > self.maximum] = 1
            _cdf[val < self.minimum] = 0
            
        except:
            if val > self.maximum:
                _cdf = 1
            elif val < self.minimum:
                _cdf = 0 
                
        if self.add_uniform:
            _cdf = self.gamma*(val-self.minimum_f)/(self.maximum_f-self.minimum_f) + (1-self.gamma)*_cdf 
                
                
        return _cdf
    
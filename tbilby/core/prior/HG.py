from bilby.core.prior import Prior, Uniform, ConditionalBeta, Beta, ConditionalUniform, ConditionalBasePrior, LogUniform
from bilby.core.prior import ConditionalLogUniform, Constraint, DeltaFunction
from scipy.interpolate import interp1d
from scipy.special._ufuncs import xlogy, erf
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.utils import infer_parameters_from_function
import bilby
import pandas as pd
import corner
from bilby.core.prior import PriorDict
from bilby.core.utils import infer_args_from_method, infer_parameters_from_function
from bilby.core.prior import conditional_prior_factory

class ConditionalPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """

        Parameters
        ==========
        dictionary: dict
            See parent class
        filename: str
            See parent class
        """
        self._conditional_keys = []
        self._unconditional_keys = []
        self._rescale_keys = []
        self._rescale_indexes = []
        self._least_recently_rescaled_keys = []
        super(ConditionalPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )
        self._resolved = False
        self._resolve_conditions()

    def _resolve_conditions(self):
        """
        Resolves how priors depend on each other and automatically
        sorts them into the right order.
        1. All unconditional priors are put in front in arbitrary order
        2. We loop through all the unsorted conditional priors to find
        which one can go next
        3. We repeat step 2 len(self) number of times to make sure that
        all conditional priors will be sorted in order
        4. We set the `self._resolved` flag to True if all conditional
        priors were added in the right order
        """
        self._unconditional_keys = [
            key for key in self.keys() if not hasattr(self[key], "condition_func")
        ]
        conditional_keys_unsorted = [
            key for key in self.keys() if hasattr(self[key], "condition_func")
        ]
        self._conditional_keys = []
        for _ in range(len(self)):
            for key in conditional_keys_unsorted[:]:
                if self._check_conditions_resolved(key, self.sorted_keys):
                    self._conditional_keys.append(key)
                    conditional_keys_unsorted.remove(key)

        self._resolved = True
        if len(conditional_keys_unsorted) != 0:
            self._resolved = False

    def _check_conditions_resolved(self, key, sampled_keys):
        """Checks if all required variables have already been sampled so we can sample this key"""
        conditions_resolved = True
        for k in self[key].required_variables:
            if k not in sampled_keys:
                conditions_resolved = False
        return conditions_resolved

    def sample_subset(self, keys=iter([]), size=None):
        self.convert_floats_to_delta_functions()
        subset_dict = ConditionalPriorDict({key: self[key] for key in keys})
        if not subset_dict._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolvable conditions."
            )
        samples = dict()
        for key in subset_dict.sorted_keys:
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
                try:
                    samples[key] = subset_dict[key].sample(
                        size=size, **subset_dict.get_required_variables(key)
                    )
                except ValueError:
                    # Some prior classes can not handle an array of conditional parameters (e.g. alpha for PowerLaw)
                    # If that is the case, we sample each sample individually.
                    required_variables = subset_dict.get_required_variables(key)
                    samples[key] = np.zeros(size)
                    for i in range(size):
                        rvars = {
                            key: value[i] for key, value in required_variables.items()
                        }
                        samples[key][i] = subset_dict[key].sample(**rvars)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    def get_required_variables(self, key):
        """Returns the required variables to sample a given conditional key.

        Parameters
        ==========
        key : str
            Name of the key that we want to know the required variables for

        Returns
        =======
        dict: key/value pairs of the required variables
        """
        return {
            k: self[k].least_recently_sampled
            for k in getattr(self[key], "required_variables", [])
        }

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.product`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        prob = np.product(res, **kwargs)
        return prob

    def ln_prob(self, sample, axis=None):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the log probability of
        axis: Union[None, int]
            Axis along which the summation is performed

        Returns
        =======
        float: Joint log probability of all the individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].ln_prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        ln_prob = np.sum(res, axis=axis)
        return ln_prob

    def cdf(self, sample):
        self._prepare_evaluation(*zip(*sample.items()))
        res = {
            key: self[key].cdf(sample[key], **self.get_required_variables(key))
            for key in sample
        }
        return sample.__class__(res)

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        from matplotlib.cbook import flatten

        keys = list(keys)
        theta = list(theta)
        self._check_resolved()
        self._update_rescale_keys(keys)
        result = dict()
        for key, index in zip(
            self.sorted_keys_without_fixed_parameters, self._rescale_indexes
        ):
            result[key] = self[key].rescale(
                theta[index], **self.get_required_variables(key)
            )
            self[key].least_recently_sampled = result[key]
        return list(flatten([result[key] for key in keys]))

    def _update_rescale_keys(self, keys):
        if not keys == self._least_recently_rescaled_keys:
            self._rescale_indexes = [
                keys.index(element)
                for element in self.sorted_keys_without_fixed_parameters
            ]
            self._least_recently_rescaled_keys = keys

    def _prepare_evaluation(self, keys, theta):
        self._check_resolved()
        for key, value in zip(keys, theta):
            self[key].least_recently_sampled = value

    def _check_resolved(self):
        if not self._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolveable conditions."
            )

    @property
    def conditional_keys(self):
        return self._conditional_keys

    @property
    def unconditional_keys(self):
        return self._unconditional_keys

    @property
    def sorted_keys(self):
        return self.unconditional_keys + self.conditional_keys

    @property
    def sorted_keys_without_fixed_parameters(self):
        return [
            key
            for key in self.sorted_keys
            if not isinstance(self[key], (DeltaFunction, Constraint))
        ]

    def __setitem__(self, key, value):
        super(ConditionalPriorDict, self).__setitem__(key, value)
        self._resolve_conditions()

    def __delitem__(self, key):
        super(ConditionalPriorDict, self).__delitem__(key)
        self._resolve_conditions()

        

#p(t) = \int p(t,f) df
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
            x = np.linspace(self.minimum_t, self.maximum_t, 10000)
            xx = np.zeros((len(x),len(self.t[0])))
            for i in range(len(self.t[0])):
                xx[:,i] = x 
            cdf = self.cdf(xx)

            for i in range(len(self.t[0])):
                samples[i]=interp1d(cdf[:,i], x)(val[i])

            return samples
        except:

            #define inverse_cdf
            x = np.linspace(self.minimum_t, self.maximum_t, 10000)
            cdf = self.cdf(x)
            sample=interp1d(cdf, x)(val)

            return sample
        
        #return erfinv(2 * val * self.normalisation + erf(
        #    (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

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
        probs = (np.exp(-(val-self.t[0])**2/(self.alpha**2 * self.sigma_t**2))* \
                     self.alpha*(erf((-self.f[0] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                                 erf((self.f[0] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                     np.exp(-(val-self.t[0])**2/(self.beta**2 * self.sigma_t**2))* \
                     self.beta*(erf((-self.f[0] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                                erf((self.f[0] - self.minimum_f)/(self.beta * self.sigma_f))))\
                     / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) \
                     / self.normalisation[0] 
        if self.n > 1:
            for i in range(1, self.n):
                probs += (np.exp(-(val-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                             self.alpha*(erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                                         erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                             np.exp(-(val-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                             self.beta*(erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                                        erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                             / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) \
                             / self.normalisation[i]
                
        probs = probs * self.is_in_prior_range(val) / self.n        
        
        return  probs

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables)
        _cdf = (self.alpha**2 * \
                       (erf((-self.f[0] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                        erf((self.f[0] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                       (erf((-self.t[0] + val)/(self.alpha * self.sigma_t)) + \
                        erf((self.t[0] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                        self.beta**2 * \
                       (erf((-self.f[0] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                        erf((self.f[0] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                       (erf((-self.t[0] + val)/(self.beta * self.sigma_t)) + \
                        erf((self.t[0] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                       (4 * (self.alpha**2 - self.beta**2))\
                    /self.normalisation[0]
        if self.n > 1:            
            for i in range(1,self.n):
                _cdf += (self.alpha**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.alpha * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.alpha * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.alpha * self.sigma_t))) - \
                            self.beta**2 * \
                           (erf((-self.f[i] + self.maximum_f)/(self.beta * self.sigma_f)) + \
                            erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))) *  \
                           (erf((-self.t[i] + val)/(self.beta * self.sigma_t)) + \
                            erf((self.t[i] - self.minimum_t)/(self.beta * self.sigma_t)))) / \
                           (4 * (self.alpha**2 - self.beta**2))\
                        /self.normalisation[i]
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
        for key, value in reference_params.items():
            setattr(self, key, value)

    def prob(self, val, **required_variables):
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        
        probs_2d = (np.exp(-((val-self.f[0])**2 / self.sigma_f**2 + (self.t[-1]-self.t[0])**2/ self.sigma_t**2)/ self.alpha**2) - \
                   np.exp(-((val-self.f[0])**2 / self.sigma_f**2 + (self.t[-1]-self.t[0])**2/ self.sigma_t**2)/ self.beta**2)) / \
                   ( np.pi * self.sigma_t * self.sigma_f * (self.alpha**2 - self.beta**2)) / self.prob_t[0]
                   
        if self.n > 1:
            for i in range(1, self.n):
                probs_2d += (np.exp(-((val-self.f[i])**2 / self.sigma_f**2 + (self.t[-1]-self.t[i])**2/ self.sigma_t**2)/ self.alpha**2) - \
                           np.exp(-((val-self.f[i])**2 / self.sigma_f**2 + (self.t[-1]-self.t[i])**2/ self.sigma_t**2)/ self.beta**2)) / \
                           ( np.pi * self.sigma_t * self.sigma_f * (self.alpha**2 - self.beta**2)) / self.prob_t[i]
        
        
        probs_2d = probs_2d * self.is_in_prior_range(val) / self.n
        
        
        
        return probs_2d

    # def inverse_cdf(self,val, **required_variables):
    #     return self._inverse_cdf(val)

    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        try:
            samples=np.zeros(len(val))

            #define inverse_cdf
            x = np.linspace(self.minimum_f, self.maximum_f, 10000)
            xx = np.zeros((len(x),len(self.t[0])))
            for i in range(len(self.t[0])):
                xx[:,i] = x 
            cdf = self.cdf(xx)

            for i in range(len(self.t[0])):
                samples[i]=interp1d(cdf[:,i], x)(val[i])

            return samples
        except:

            #define inverse_cdf
            x = np.linspace(self.minimum_f, self.maximum_f, 10000)
            cdf = self.cdf(x)
            sample=interp1d(cdf, x)(val)

            return sample

    def cdf(self, val, **required_variables):
        self.update_conditions(**required_variables) # set self.t = t, self.prob_t = prob_t
        n = self.n
        _cdf = (np.exp(-(self.t[n]-self.t[0])**2/(self.alpha**2 * self.sigma_t**2))* \
             self.alpha*(erf((-self.f[0] + val)/(self.alpha * self.sigma_f)) + \
                         erf((self.f[0] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
             np.exp(-(self.t[n]-self.t[0])**2/(self.beta**2 * self.sigma_t**2))* \
             self.beta*(erf((-self.f[0] + val)/(self.beta * self.sigma_f)) + \
                        erf((self.f[0] - self.minimum_f)/(self.beta * self.sigma_f))))\
             / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) / self.prob_t[0]
        
        if self.n > 1:
            for i in range(1,self.n):
                _cdf += (np.exp(-(self.t[n]-self.t[i])**2/(self.alpha**2 * self.sigma_t**2))* \
                     self.alpha*(erf((-self.f[i] + val)/(self.alpha * self.sigma_f)) + \
                                 erf((self.f[i] - self.minimum_f)/(self.alpha * self.sigma_f))) - \
                     np.exp(-(self.t[n]-self.t[i])**2/(self.beta**2 * self.sigma_t**2))* \
                     self.beta*(erf((-self.f[i] + val)/(self.beta * self.sigma_f)) + \
                                erf((self.f[i] - self.minimum_f)/(self.beta * self.sigma_f))))\
                     / (2 * np.sqrt(np.pi) * (self.alpha**2-self.beta**2) * self.sigma_t) / self.prob_t[i]
        
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
    
    
def condition_func_t1(reference_params, f0):
    index=1
    t=np.array([np.zeros_like(f0)])
    f=np.array([f0])
    normalisation=np.zeros(t.shape)
    
    for i in range(index):
        normalisation[i] = (reference_params['alpha']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['alpha'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['alpha'] * reference_params['sigma_t']))) - \
                       reference_params['beta']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['beta'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['beta'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['beta'] * reference_params['sigma_t'])))) / \
                       (4 * (reference_params['alpha']**2 - reference_params['beta']**2))

    return dict(t=t, f=f, _normalisation=normalisation)

def condition_func_t2(reference_params, f0, dt1, f1):
    index=2
    t=np.array([np.zeros_like(f0), dt1])
    f=np.array([f0, f1])
    normalisation=np.zeros(t.shape)
    for i in range(index):
        normalisation[i] = (reference_params['alpha']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['alpha'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['alpha'] * reference_params['sigma_t']))) - \
                       reference_params['beta']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['beta'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['beta'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['beta'] * reference_params['sigma_t'])))) / \
                       (4 * (reference_params['alpha']**2 - reference_params['beta']**2))
    
    return dict(t=t, f=f, _normalisation=normalisation)

def condition_func_t3(reference_params, f0, dt1, f1, dt2, f2):
    index=3
    t=np.array([np.zeros_like(f0), dt1, dt2])
    f=np.array([f0, f1, f2])
    normalisation=np.zeros(t.shape)
    for i in range(index):
        normalisation[i] = (reference_params['alpha']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['alpha'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['alpha'] * reference_params['sigma_t']))) - \
                       reference_params['beta']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['beta'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['beta'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['beta'] * reference_params['sigma_t'])))) / \
                       (4 * (reference_params['alpha']**2 - reference_params['beta']**2))
    
    return dict(t=t, f=f, _normalisation=normalisation)

def condition_func_t4(reference_params, f0, dt1, f1, dt2, f2, dt3, f3):
    index=4
    t=np.array([np.zeros_like(f0), dt1, dt2, dt3])
    f=np.array([f0, f1, f2, f3])
    normalisation=np.zeros(t.shape)
    for i in range(index):
        normalisation[i] = (reference_params['alpha']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['alpha'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['alpha'] * reference_params['sigma_t']))) - \
                       reference_params['beta']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['beta'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['beta'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['beta'] * reference_params['sigma_t'])))) / \
                       (4 * (reference_params['alpha']**2 - reference_params['beta']**2))
    
    return dict(t=t, f=f, _normalisation=normalisation)

def condition_func_t5(reference_params, f0, dt1, f1, dt2, f2, dt3, f3, dt4, f4):
    index=5
    t=np.array([np.zeros_like(f0), dt1, dt2, dt3, dt4])
    f=np.array([f0, f1, f2, f3, f4])
    normalisation=np.zeros(t.shape)
    for i in range(index):
        normalisation[i] = (reference_params['alpha']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['alpha'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['alpha'] * reference_params['sigma_t']))) - \
                       reference_params['beta']**2 * \
                       (erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                        erf((f[i] - reference_params['minimum_f'])/(reference_params['beta'] * reference_params['sigma_f']))) *  \
                       (erf((-t[i] + reference_params['maximum_t'])/(reference_params['beta'] * reference_params['sigma_t'])) + \
                        erf((t[i] - reference_params['minimum_t'])/(reference_params['beta'] * reference_params['sigma_t'])))) / \
                       (4 * (reference_params['alpha']**2 - reference_params['beta']**2))
    
    return dict(t=t, f=f, _normalisation=normalisation)

def condition_func_f1(reference_params, f0, dt1):
    
    index=1
    
    t=np.array([np.zeros_like(f0),dt1])
    f=np.array([f0])
    prob_t = np.zeros(f.shape)
    for i in range(index):
        prob_t[i] = (np.exp(-(t[-1]-t[i])**2/(reference_params['alpha']**2 * reference_params['sigma_t']**2))* \
                     reference_params['alpha']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                                 erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) - \
                  np.exp(-(t[-1]-t[i])**2/(reference_params['beta']**2 * reference_params['sigma_t']**2))* \
                     reference_params['beta']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                                erf((f[i] - reference_params['minimum_f'])/(reference_params['beta']* reference_params['sigma_f']))))\
                 / (2 * np.sqrt(np.pi) * (reference_params['alpha']**2-reference_params['beta']**2) * reference_params['sigma_t'])
    
    return dict(t=t, f=f, prob_t=prob_t)

def condition_func_f2(reference_params, f0, dt1, f1, dt2):
    
    index=2
    
    t=np.array([np.zeros_like(f0),dt1,dt2])
    f=np.array([f0,f1])
    prob_t = np.zeros(f.shape)
    for i in range(index):
        prob_t[i] = (np.exp(-(t[-1]-t[i])**2/(reference_params['alpha']**2 * reference_params['sigma_t']**2))* \
                     reference_params['alpha']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                                 erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) - \
                  np.exp(-(t[-1]-t[i])**2/(reference_params['beta']**2 * reference_params['sigma_t']**2))* \
                     reference_params['beta']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                                erf((f[i] - reference_params['minimum_f'])/(reference_params['beta']* reference_params['sigma_f']))))\
                 / (2 * np.sqrt(np.pi) * (reference_params['alpha']**2-reference_params['beta']**2) * reference_params['sigma_t'])
    
    return dict(t=t, f=f, prob_t=prob_t)

def condition_func_f3(reference_params, f0, dt1, f1, dt2, f2, dt3):
    
    index=3
    
    t=np.array([np.zeros_like(f0),dt1,dt2,dt3])
    f=np.array([f0,f1,f2])
    prob_t = np.zeros(f.shape)
    for i in range(index):
        prob_t[i] = (np.exp(-(t[-1]-t[i])**2/(reference_params['alpha']**2 * reference_params['sigma_t']**2))* \
                     reference_params['alpha']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                                 erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) - \
                  np.exp(-(t[-1]-t[i])**2/(reference_params['beta']**2 * reference_params['sigma_t']**2))* \
                     reference_params['beta']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                                erf((f[i] - reference_params['minimum_f'])/(reference_params['beta']* reference_params['sigma_f']))))\
                 / (2 * np.sqrt(np.pi) * (reference_params['alpha']**2-reference_params['beta']**2) * reference_params['sigma_t'])
    
    return dict(t=t, f=f, prob_t=prob_t)

def condition_func_f4(reference_params, f0, dt1, f1, dt2, f2, dt3, f3, dt4):
    
    index=4
    
    t=np.array([np.zeros_like(f0),dt1,dt2,dt3,dt4])
    f=np.array([f0,f1,f2,f3])
    prob_t = np.zeros(f.shape)
    for i in range(index):
        prob_t[i] = (np.exp(-(t[-1]-t[i])**2/(reference_params['alpha']**2 * reference_params['sigma_t']**2))* \
                     reference_params['alpha']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                                 erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) - \
                  np.exp(-(t[-1]-t[i])**2/(reference_params['beta']**2 * reference_params['sigma_t']**2))* \
                     reference_params['beta']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                                erf((f[i] - reference_params['minimum_f'])/(reference_params['beta']* reference_params['sigma_f']))))\
                 / (2 * np.sqrt(np.pi) * (reference_params['alpha']**2-reference_params['beta']**2) * reference_params['sigma_t'])
    
    return dict(t=t, f=f, prob_t=prob_t)

def condition_func_f5(reference_params, f0, dt1, f1, dt2, f2, dt3, f3, dt4, f4, dt5):
    
    index=5
    
    t=np.array([np.zeros_like(f0),dt1,dt2,dt3,dt4,dt5])
    f=np.array([f0,f1,f2,f3,f4])
    prob_t = np.zeros(f.shape)
    for i in range(index):
        prob_t[i] = (np.exp(-(t[-1]-t[i])**2/(reference_params['alpha']**2 * reference_params['sigma_t']**2))* \
                     reference_params['alpha']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['alpha'] * reference_params['sigma_f'])) + \
                                 erf((f[i] - reference_params['minimum_f'])/(reference_params['alpha'] * reference_params['sigma_f']))) - \
                  np.exp(-(t[-1]-t[i])**2/(reference_params['beta']**2 * reference_params['sigma_t']**2))* \
                     reference_params['beta']*(erf((-f[i] + reference_params['maximum_f'])/(reference_params['beta'] * reference_params['sigma_f'])) + \
                                erf((f[i] - reference_params['minimum_f'])/(reference_params['beta']* reference_params['sigma_f']))))\
                 / (2 * np.sqrt(np.pi) * (reference_params['alpha']**2-reference_params['beta']**2) * reference_params['sigma_t'])
    
    return dict(t=t, f=f, prob_t=prob_t)
    
    
def get_A1(reference_params, amplitude1):
    maximum = amplitude1
    below = amplitude1 < reference_params['minimum']
    try:
        maximum[below] = reference_params['maximum']
    except:
        if below == True:
            maximum = reference_params['maximum']
    
    return dict(maximum=maximum)

def get_A2(reference_params, amplitude2):
    maximum = amplitude2
    below = amplitude2 < reference_params['minimum']
    try: 
        maximum[below] = reference_params['maximum']
    except:
        if below == True:
            maximum = reference_params['maximum']
    
    return dict(maximum=maximum)

def get_A3(reference_params, amplitude3):
    maximum = amplitude3
    below = amplitude3 < reference_params['minimum']
    try: 
        maximum[below] = reference_params['maximum']
    except:
        if below == True:
            maximum = reference_params['maximum']
    
    return dict(maximum=maximum)

def get_A4(reference_params, amplitude4):
    maximum = amplitude4
    below = amplitude4 < reference_params['minimum']
    try: 
        maximum[below] = reference_params['maximum']
    except:
        if below == True:
            maximum = reference_params['maximum']
    
    return dict(maximum=maximum)

def get_A5(reference_params, amplitude5):
    maximum = amplitude5
    below = amplitude5 < reference_params['minimum']
    try: 
        maximum[below] = reference_params['maximum']
    except:
        if below == True:
            maximum = reference_params['maximum']
    
    return dict(maximum=maximum)



ConditionalPrior = conditional_prior_factory(Prior)

import tbilby, bilby
from bilby.core.prior import  Uniform
import numpy as np
import matplotlib.pyplot as plt
import corner
import random
from scipy.stats import norm




# model 
n_peaks = 8
sigma = 10
Amp_noise = 0.05
def gauss(x,mu,sigma_g):
    return norm(mu,sigma_g).pdf(x)
# here we define a model which could be a sum of several functions. 
component_functions_dict = {}
component_functions_dict[gauss] = (n_peaks,'mu')
model = tbilby.core.base.create_transdimensional_model('model', component_functions_dict,returns_polarization=False ,SaveTofile=True)

# create some data to work with 
random.seed(10)
x = np.linspace(0, 100, 1001)
p = np.array([54, 15, 81])
#add some noise
noise=norm(loc=0, scale = Amp_noise).rvs(1001)
y = gauss(x, p[0], sigma) + gauss(x, p[1], sigma) + gauss(x, p[2], sigma) + noise

# priors 
priors_t = bilby.core.prior.dict.ConditionalPriorDict()
priors_t['n_gauss'] = tbilby.core.prior.DiscreteUniform(0,n_peaks,'n_gauss')
priors_t['sigma_g'] = sigma # set constant so we don't sample it 
# create a list of simple uniform priors
priors_t = tbilby.core.base.create_plain_priors(Uniform, 'mu', nmax=n_peaks,prior_dict_to_add=priors_t, minimum=0,maximum=100)

samples = priors_t.sample(size=50000)
# just to see that the conditional works 

plt.figure()
plt.plot(samples['mu0'],samples['mu1'],'o') 
plt.xlabel('mu0')
plt.ylabel('mu1')

likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma = Amp_noise*np.ones(len(x)))

# run the smapler 
result = bilby.core.sampler.run_sampler(
        likelihood,
        priors=priors_t,
        sampler='dynesty',
        label='Three_gauss_example',
        clean=True,
        nlive=10,
        outdir='outdir',
       
    )

# lets check the best number of component, this is equivalent to comparing BF   
plt.figure()
plt.hist(result.posterior['n_gauss'].astype(int),bins=np.arange(9))
plt.xlabel('n components')
plt.ylabel('freq.')
   
    
# assuming we got the what we wanted..    
tbest = result.posterior[result.posterior['n_gauss']==3]

labels = ['mu0','mu1','mu2']
samples = tbest[labels].values    
fig = corner.corner(samples, labels=labels, quantiles=[0.025, 0.5, 0.975],
                   show_titles=True, title_kwargs={"fontsize": 12})
plt.subplot(3,3,1)
plt.vlines(p[1],0,1000)
plt.subplot(3,3,5)
plt.vlines(p[0],0,1000)
plt.subplot(3,3,9)
plt.vlines(p[2],0,1000)


#tbest.log_likelihood

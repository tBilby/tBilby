from context import tbilby # this is just a hack to make the example work, usually import tbilby should work
import bilby
from bilby.core.prior import  Uniform
import numpy as np
import matplotlib.pyplot as plt
import corner
import random
from scipy.stats import norm,cauchy


# to define Transdimensional Conditional prior you need to inherit from the relevant prior
class TransdimensionalConditionalUniform(tbilby.core.prior.TransdimensionalConditionalUniform):
   # one must define the transdimensional_condition_function function, so we know what to do with conditional variables...  
    # it is an abstract function, without it you cant instantiate this class 
    def transdimensional_condition_function(self,**required_variables):
        ''' setting the mimmum according the the last peak value of the gaussian.
        Here you refer to the parameters you are 
        working with '''
        # mu is returned as an array 
        minimum = self.minimum
        if(len(self.mu)>0): # handle the first mu case
            minimum = self.mu[-1]               
            setattr(self,'minimum',minimum)  # setting the atribute of the class
        return dict(minimum=minimum)
 
# create a 
class TransdimensionalConditionalBeta(tbilby.core.prior.TransdimensionalConditionalBeta):
   # one must define the transdimensional_condition_function function, so we know what to do with conditional variables...  
    # it is an abstract function, without it you cant instantiate this class 
    def transdimensional_condition_function(self,**required_variables):
        ''' 
        note that this specific class implementation is not generally doesnt make much sense,
        It is used for demonstration 
        '''
        # here we should have access to all the relevant params 
    
        
        alpha= self.alpha
        beta= self.beta
        minimum = self.minimum       
        if(len(self.mu_l)>0): # handle the first mu case
            #minimum = self.mu_l[-1] # use the previous ones to see the minimum                
            # assuming we know that the lorentzian are located where the gusasians, lets have a dynamic prior 
            
            # here you get access to the parameters in the following way:
            # self.mu[trans-dimensional componant number ,sample_number]
            # one can expect mu[8,50000] matrix when you are trying to sample 50,000 samples from this prior
            # or it could show up as mu[8,] since a single event was requested. 
            
            
            # In bibly parameters like self.n_gauss, might show up as array-like, array or float, depending on the number of samples requested from the prior   
            # again this is cause the number of samples can change widely from 1 to 100,000 
            
            # make sure that we get what we expect
            self.mu =self.mu.reshape(8,-1)
                                    
            
            # generally, when alpha < beta the left side will peak 
            # when alpha > beta the right side will peak 
            # when alpha ~ beta, the two sides will peak similarly  
            # this doesnt make much sense to use this in real analysis, but for capability demonstration it is good enough
            
            # len(mu_l), will give us the order of the lorentzian (i.e the number of componant function we are dealing with) 
            
            if isinstance(self.n_gauss, float):
                # this means a single sample was requested 
                n_g = np.array(self.n_gauss)
                self.mu=self.mu.reshape(-1,1)
            else:
                n_g = self.n_gauss
            # since we know that the gaussian are ordered (due to their conditional mean prior), we can set alpha and beta according to their values      
            N_componant_function = self.mu_l.shape[0]
            
            beta = np.mean(self.mu[N_componant_function,:])                
            alpha = np.mean(self.mu[N_componant_function,:])                
            # skew the distrinution depending on the number of functions we are dealing with 
            if N_componant_function >= 2: 
                alpha = np.mean(self.mu[N_componant_function+2,:])/5                                
            if N_componant_function < 2:     
                beta =  np.mean(self.mu[N_componant_function+2,:])/5
            
            
            
        return dict(alpha=alpha,beta = beta, minimum=minimum)
  



# model 
n_peaks_g=8
n_peaks_l=5
sigma = 10
sigma_data_l =1 
Amp_l=0.03
Amp_noise=0.01
def gauss(x,mu,sigma_g):
    return norm(mu,sigma_g).pdf(x)
def lorentzian(x,A,mu_l,sigma_l):
    #return 0
    x_moved =(x - mu_l) / sigma_l
    return A*cauchy.pdf(x_moved) / sigma_l
    

# here we define a model which could be a sum of several functions, here it a sum of 
# gauss + lorentzian where only the mean is trans-dimensional  
componant_functions_dict={}
componant_functions_dict[gauss]=(n_peaks_g,'mu')
componant_functions_dict[lorentzian]=(n_peaks_l,'mu_l')
model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=False,SaveTofile=True)

# create some data to work with 
random.seed(10)
x=np.linspace(0,100,1001)
p=np.array([54, 15, 81])
#add some noise
noise=norm(loc=0,scale=Amp_noise).rvs(1001)
y =   gauss(x,p[0],sigma) +  gauss(x,p[1],sigma) +  gauss(x,p[2],sigma) +\
      lorentzian(x,Amp_l,p[0],sigma_data_l) + lorentzian(x,Amp_l,p[1],sigma_data_l) +lorentzian(x,Amp_l,p[2],sigma_data_l) + noise
# plot it if you want to have a look 
#plt.close('all')
#plt.plot(x,y)

# priors 
# make sure you call the discrete priors with n_function name , i.e n_lorentzian
priors_t = bilby.core.prior.dict.ConditionalPriorDict()
priors_t['n_gauss'] = tbilby.core.prior.DiscreteUniform(1,n_peaks_g,'n_gauss') # at leat one guassian
priors_t['n_lorentzian'] = tbilby.core.prior.DiscreteUniform(0,n_peaks_l,'n_loren')
priors_t['sigma_g']= sigma # set it constant so we wont sample on it 
priors_t['sigma_l']= sigma_data_l
# estimate the min and max from the data
priors_t['A'] = bilby.core.prior.Uniform(np.min(y), 2*np.max(y))# allow some extra freedom for the max 

# creates a list of conditional unifrom priors for you, now it will use all the options 
#nested_conditional_transdimensional_params is used, 
#meaning you will get dependence mu2(mu1,mu0).. 
priors_t =tbilby.core.base.create_transdimensional_priors(transdimensional_prior_class=TransdimensionalConditionalUniform,\
                                                          param_name='mu',\
                                                          nmax= n_peaks_g,\
                                                          nested_conditional_transdimensional_params=['mu'],\
                                                          conditional_transdimensional_params=[],\
                                                          conditional_params=[],\
                                                          prior_dict_to_add=priors_t,\
                                                          SaveConditionFunctionsToFile=False,\
                                                          minimum= 0,maximum=100)
    
# nested_conditional_transdimensional_params is used, conditional_transdimensional_params and conditional_params
#meaning you will get the dependence mu_l2(mu_l1,mu_l0,mu8,mu7,...,n_gauss).. so you can use these to calculate what you need 
priors_t =tbilby.core.base.create_transdimensional_priors(transdimensional_prior_class=TransdimensionalConditionalBeta,\
                                                          param_name='mu_l',\
                                                          nmax= n_peaks_l,\
                                                          nested_conditional_transdimensional_params=['mu_l'],\
                                                          conditional_transdimensional_params={'mu':n_peaks_g},\
                                                          conditional_params=['n_gauss'],\
                                                          prior_dict_to_add=priors_t,\
                                                          SaveConditionFunctionsToFile=True,\
                                                          minimum= 0,maximum=100,alpha=1,beta=1)    
    
    
samples = priors_t.sample(size=50)


# just to see that the conditional works 
plt.close('all')
plt.figure()
plt.plot(samples['mu_l0'],samples['mu_l1'],'o') 
plt.xlabel('mu_l0')
plt.ylabel('mu_l1')

plt.figure()
plt.plot(samples['mu_l1'],samples['mu_l2'],'o') 
plt.xlabel('mu_l1')
plt.ylabel('mu_l2')



likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma=Amp_noise*np.ones(len(x)))

# run the smapler 
run_sampler =True 
plot_result=True

if run_sampler: 
    result = bilby.core.sampler.run_sampler(
            likelihood,
            priors=priors_t,
            sampler='dynesty',
            label='Three_gauss_example',
            clean=True,
            nlive=100,
            outdir='outdir',
           
        )

if plot_result:
    
    # below there are some cool function to help you get the plots that you want, at the very end of the script  
    # there are even more useful one, so it worth going over the example until the very end of it 
    
    result = bilby.result.read_in_result(filename='Three_gauss_example_result.json')
        
    # lets check the best number of componant, this is equivalent to comparing BF   
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(result.posterior['n_gauss'].astype(int),bins=np.arange(9))
    plt.xlabel('n components')
    plt.ylabel('freq.')
    plt.title('n_gauss')
    plt.subplot(1,2,2)
    plt.hist(result.posterior['n_lorentzian'].astype(int),bins=np.arange(9))
    plt.xlabel('n components')
    plt.ylabel('freq.')
    plt.title('n_lorentzian')
        
    # assuming we got what we wanted..    
    tbest = result.posterior[(result.posterior['n_gauss']==3) & (result.posterior['n_lorentzian']==3) ]
    
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
    
    labels = ['mu_l0','mu_l1','mu_l2','mu_l3','mu_l4']
    samples = tbest[labels].values    
    fig = corner.corner(samples, labels=labels, quantiles=[0.025, 0.5, 0.975],
                       show_titles=True, title_kwargs={"fontsize": 12})
    for j,i in zip(np.arange(5),np.arange(1,25,5)):
        plt.subplot(5,5,i+j)
        plt.vlines(p[1],0,1000)
        #plt.subplot(3,3,5)
        plt.vlines(p[0],0,1000)
        #plt.subplot(3,3,9)
        plt.vlines(p[2],0,1000)
        
        
        
    
    # utilizing tbilby tools for plotting and error estimation 
    
    # as the name indicates 
    tbilby.core.base.plotting.corner_plot_discrete_params(result,filename='n_dist.png')
    
    tbilby.core.base.plotting.corner_plot_single_component_function(result, lorentzian, order=1, not_tparams= ['sigma_l','A'], filename='lorentzian_1.png')
    tbilby.core.base.plotting.corner_plot_single_transdimentional_component_functions(result,lorentzian,filename='mu_l_all_orders')    
    
    # this will do the processing of the results for you, remove ghost parameters and return the function which is most likely from the data based on samplign frequency, which should be the same as BF    
    result_processed,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
    tbilby.core.base.plotting.corner_plot_single_transdimenstional_param(result_processed,'mu',filename='mu_g.png')
    tbilby.core.base.plotting.corner_plot_single_transdimenstional_param(result_processed,'mu_l',filename='mu_l.png')
    
    # find the best maximum likelihood fit for you 
    best_params_post = tbilby.core.base.extract_maximal_likelihood_param_values(result_processed, model)
    # sample from the posterior 
    sampled_params = result_processed.posterior.sample(1).to_dict('records')[0] # it creates a list, we take it single value  
    del sampled_params['log_likelihood']
    del sampled_params['log_prior']
    
    plt.figure()
    plt.plot(x,y,'-oc',label='Data')
    plt.plot(x,model(x,**best_params_post),'-r',label='MLE')
    plt.plot(x,model(x,**sampled_params),'-k',label='Posterior sampled params ')
    plt.legend()
    plt.grid(True)
    
    
    
    


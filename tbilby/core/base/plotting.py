import bilby
import re
from ..prior import DiscreteUniform
from bilby.core.utils import infer_parameters_from_function
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner

def corner_plot_discrete_params(result,filename=None,**kwargs):
    '''
    

    Parameters
    ----------
    result : bilby result object           
    filename : str, the name of the file, if not None will save into a file  
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    
    result = _fix_range_issue(result)
    
    discrete_parameters = []
    for p in result.priors.keys():
        if type(result.priors[p])==DiscreteUniform:
            discrete_parameters.append(p)
    
    SaveTofile=True
    if filename is None:
        SaveTofile=False 
                
    if  len(discrete_parameters)==0:
        print('coudlnt find any discrete parameters, maybe thiis is not a real transdimensional sampleing ? quitting this function, nothing else matters  ')
        return 
        
    result.plot_corner(discrete_parameters,SaveTofile=SaveTofile,filename=filename ,**kwargs)  
    
    # tbilby addition... check the nested samples before the rejection sampling to verify teh process
    # and maybe to calculate the BF using these incase the discrete distribution peaks at one particular value? 
    
    
    
    from scipy.special import logsumexp
    Z_tot = logsumexp(np.log(result.nested_samples['weights'])+result.nested_samples['log_likelihood'])
    
    grs = result.nested_samples.groupby(discrete_parameters)
    dict_ln_Z={}
    for gr in grs:
        Z_gr = logsumexp(np.log(gr[1]['weights'])+gr[1]['log_likelihood'])
        dict_ln_Z[gr[0]]=Z_gr

    log_Z_max = dict_ln_Z[max(dict_ln_Z, key=dict_ln_Z.get) ]
    # keep only somewhat "relevant" values 
    dict_ln_Z = {key: dict_ln_Z[key] for key in dict_ln_Z.keys() if log_Z_max-dict_ln_Z[key]<=50}
       
    
    
    
    
    

    
    samples = result.nested_samples[discrete_parameters]    
    fig = corner.corner(samples, labels=discrete_parameters)        
    plt.show()
    
    if SaveTofile:
        plt.savefig('tBilby_' + filename)
        plt.close()
    
    
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],'%s' % float('%.3g' % y[i]))
    
    plt.figure()
    x,y =range(len(dict_ln_Z)), log_Z_max-np.array(list(dict_ln_Z.values()))
    plt.bar(x,y, align='center')
    plt.xticks(range(len(dict_ln_Z)), list(dict_ln_Z.keys()))
    addlabels(x, y)
    plt.xlabel('Discrete variables')
    plt.ylabel('$\log{BF}$')
    plt.show()
    
    if SaveTofile:
        plt.savefig('tBilby_log_Z_' + filename)
        plt.close()
    
    
def corner_plot_single_transdimenstional_param(result,param,overlay=False, filename=None,**kwargs):
    '''
    

    Parameters
    ----------
    result : bilby result object  
        DESCRIPTION.
    param : the transdimesional param that you are intrested in 
        DESCRIPTION.
    overlay: bool, if True, will plot the parameter of different order on top of each other, this is not the usualy bilby style yet, sorry about that...   
    
    filename : str, the name of the file, if not None will save into a file 
        DESCRIPTION. The default is None.


    This will create a corner plot of a single transdimesnional parameter of all the avilable component function order 
    e.g. param='mu' and we have 3 component functions. it will corner plot mu0,mu1,mu2  
 
    Returns
    -------
    None.

    '''
    
    
    
    SaveTofile=True
    if filename is None:
        SaveTofile=False 
    
    
    result = _fix_range_issue(result)
    
    cols=list(result.posterior.columns)   
    
    if isinstance(param, str):
        param=[param]
    
    
    for p in param:
        locs_params = sorted(_extract_words_with_numeric_suffix(partial_words=[p], full_words=cols))
    
    if len(locs_params)==0:
        print('WARNING: Couldn\'t  find the parameter you wanted to plot, maybe there is a misspeling ?')
        return 
    
    if overlay==False:        
        if filename is not None:
            result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=filename ,**kwargs)
        else:
            result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=p+'.png' ,**kwargs)
    else:
        plt.figure()
        # find teh optimal bins settings
        data = result.posterior[locs_params].values.reshape(-1,1)
        bins = np.histogram_bin_edges(data,bins= 'fd')
        
        for p in locs_params:
            plt.hist(result.posterior[p],bins=bins,alpha = 0.6, label = p )
        plt.xlabel(param[0])    
        plt.ylabel('Counts')    
        plt.legend()
        if filename is not None:
            plt.savefig(filename)
            plt.close()    
            
        
        
        
        
                    

def _extract_words_with_numeric_suffix(partial_words, full_words):
    extracted_words = []
    for partial in partial_words:
        for word in full_words:
            if partial in word:
                match = re.match(rf"{re.escape(partial)}(\d+)", word)
                if match:
                    extracted_words.append(word)
    return extracted_words

    
def corner_plot_single_transdimentional_component_functions(result,function,filename=None,**kwargs):
    '''
    

    Parameters
    ----------
    result : bilby result object 
        DESCRIPTION.
    function : the function that serves as a component function 
        DESCRIPTION.
    filename :  str, the name of the file, if not None will save into a file 
        DESCRIPTION. The default is None.


   This will create a corner plot of a single transdimesnional component function of all the avilable component function order 
   e.g. function=gauss and we have 3 component functions. it will corner plot mu0,sigma0,mu1,sigma1,mu2 ,sigma2


    Returns
    -------
    None.

    '''
    
    SaveTofile=True
    if filename is None:
        SaveTofile=False 
    
   
    params = infer_parameters_from_function(function)
    params_to_plot =[]
    all_parmas = list(result.priors.keys())
    all_parmas = [param for param in all_parmas if param in list(result.posterior.columns)]
    
    #extracted_parmas = [word for word in all_parmas if any(partial in word for partial in params)]
    extracted_parmas =_extract_words_with_numeric_suffix(params,all_parmas)
    
    locs_params = sorted(extracted_parmas)
    result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=filename,**kwargs )    


def _fix_range_issue(result):    
    
    # reset index so we wont get an exception using the first index a few lines away 
    result.posterior.reset_index(drop=True,inplace=True)
    # check for range issues, and fix it by changing a single value by 0.01%
    for p in result.priors.keys():      
        if p not in list(result.posterior.columns):
            continue 
        if result.posterior[p].max()-result.posterior[p].min() < np.abs(result.posterior[p].max()*0.0001):               
            result.posterior.at[0, p] *= 1.0001
            
    return result
    
    
def corner_plot_single_component_function(result,function,order,not_tparams = [],filename=None,**kwargs): 
    '''
    

    Parameters
    ----------
    result : bilby result object 
        DESCRIPTION.
    function : the function that serves as a component function 
        DESCRIPTION.
    order : int, the specific order of the component function 
        DESCRIPTION.
    not_tparams : list of non-transdimesional parameters. This function will fail if you provide it with wrong input 
                  e.g guass(mu,sigma), mu is transdimensional while sigma is the same for all orders, sigma should go into thsi list   
        DESCRIPTION. The default is [].
    filename :  str, the name of the file, if not None will save into a file 
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    SaveTofile=True
    if filename is None:
        SaveTofile=False 
       
    params = infer_parameters_from_function(function)
    for n_t in not_tparams:
        params.remove(n_t)
    
    params = [p+str(order) for p in params]
    params += not_tparams
    
    result = _fix_range_issue(result)
        
    result.plot_corner(params,SaveTofile=SaveTofile,filename=filename,**kwargs )    

def _format_az_error(data):
    median= np.median(data)
    err= az.hdi(data, hdi_prob=.95)
    center = str(median)[:4]
    err_p = '+' + str(err[1] - median)[:4]
    err_m =  str(err[0] - median)[:4]
    res= r'$ = \center^{\err_p}_{\err_m}$'   
    res = '= ' + center + err_p + err_m     
    
    
    return res    

def hist_maraganalized_reclustered_params(processed_dict):
    '''
    

    Parameters
    ----------
    processed_dict : this process the output of the clustering function of posterior. This is still in the experimental stage. 
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    for discrete_key in processed_dict.keys(): 
        n_clusters_dict  = processed_dict[discrete_key]
        for param,n_cluster in n_clusters_dict.keys():
            
            for t_param in n_clusters_dict[(param,n_cluster)][0].columns.values:
                plt.figure()
                for n in np.arange(n_cluster.astype(int)):                
                # plot the diagonal corner-like style
                    plt.subplot(n_cluster,n_cluster,1+ n*n_cluster + n)
                    data = n_clusters_dict[(param,n_cluster)][n][t_param[:-1]+str(n)].values
                    
                    plt.hist(data,50)
                    
                    plt.title(t_param[:-1]+str(n)  + _format_az_error(data))
            
      
 
    
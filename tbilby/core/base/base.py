import numpy as np
import pandas as pd

from bilby.core.utils import infer_parameters_from_function
import bilby
from ..prior import DiscreteUniform





def create_transdimensional_priors(transdimensional_prior_class,param_name,nmax,nested_conditional_transdimensional_params,conditional_transdimensional_params=[],conditional_params=[],prior_dict_to_add=None,SaveConditionFunctionsToFile=False,**reference_params):    
    '''
    

    Parameters
    ----------
    transdimensional_prior_class : TYPE
         the transdimensional prior type 
    param_name : TYPE
        the parameter base name, e.g. "alpha",  this will prdouce several priors for alpha0, alpha1,.. alpha_nmax
    nmax : TYPE
        The maximal number of componant functions 
    nested_conditional_transdimensional_params : TYPE
        the conditional transdimensional parameters, e.g. 'mu', will prdouce dependnce of alpha1(alpha0,mu0) ,alpha2(alpha1,alpha0,mu1,mu0),... 
   conditional_transdimensional_params : dict or list 
        additional conditional transdimensional parameters which are not related to "alpha", e.g {"beta":3 } , following the example form above  
        alpha1(alpha0,mu0,beta0,beta1,beta2), alpha2(alpha1,alpha0,mu1,mu0,beta0,beta1,beta2), if ["beta"] would be provided, the same componant function as the main variable is assumed, 
        i.e.
        alpha1(alpha0,mu0,beta0,beta1), alpha2(alpha1,alpha0,mu1,mu0,beta0,beta1,beta2)    
    conditional_params : TYPE
        just normal conditional params, following the example from above with a new 'c' parameter , this will create alpha2(alpha1,alpha0,mu1,mu0,c)   
    prior_dict_to_add : TYPE, optional
        the prior dict that you want to add these priors to 
    **reference_params : TYPE
        the parameter your prior need, like minimum and maximum, etc..  

    Returns
    -------
    conditional prior dict

    '''
    
    
    if prior_dict_to_add==None:
        prior_dict_to_add = bilby.core.prior.dict.ConditionalPriorDict()
    types =  [transdimensional_prior_class.__mro__[j].__name__ for j in np.arange(len(transdimensional_prior_class.__mro__))]
    if 'TransdimensionalConditionalPrior' not in types:
        raise Exception('create_transdimensional_priors recieved not a TransdimensionalConditionalPrior class type !!')
        
    

    for n in np.arange(nmax):        
        prior_dict_to_add[param_name+str(n)]= transdimensional_prior_class(name = param_name+str(n),componant_function_number =n ,nested_conditional_transdimensional_params=nested_conditional_transdimensional_params,conditional_transdimensional_params=conditional_transdimensional_params,conditional_params=conditional_params,debug_print_out=SaveConditionFunctionsToFile,**reference_params)
    
    
       
    
    return prior_dict_to_add 

 

def _create_priors_with_nested_condition_functions(prior_class,param_base_name='alpha',conditional_tParams_dict_conversion={'alpha':'mu','beta':'sigma'} ,nmax=3,print_out=False,SaveTofile=False,prior_dict_to_add=None,**reference_params):
    '''
    

    shouldn't be used, just old code
    '''
# nested condition functions 
    priors = bilby.core.prior.dict.ConditionalPriorDict()
    keep_functions_for_writing_to_file=''
    
    #internal params are called differenly then the external ones
    if len(prior_class.tparams)==0:
        raise Exception('You didnt define attribute tparams, it is unknown what is the internal representation of the external parameters')
    if(len(conditional_tParams_dict_conversion)!=len(prior_class.tparams)):
        raise Exception('You didnt define conditional_tParams correctly, it must be equal len to attribute tparam')
    if not set(prior_class.tparams).issubset(list(conditional_tParams_dict_conversion.values())):
        raise Exception('You didnt define conditional_tParams correctly, some parameters are missing from conditional_tParams_dict')
     
    
    condition_function_base_name="condition_func"+prior_class.__name__ +'_' +  '_'.join(conditional_tParams_dict_conversion.keys()) # helps to seperate different priors and params  
    cond_functions=[]
   
    
    for n_func in np.arange(nmax):
        arguments = ["reference_params"]
        function_lines=''
        function_name = condition_function_base_name+"_"+str(n_func) 
        for t in conditional_tParams_dict_conversion.keys():
            targs=[]               
            for n in np.arange(n_func): #np.arange(2)= [1], so it takes the lower level automaticaly without n-1  
                arguments.append(t+str(n))
                targs.append(t+str(n))
            function_lines += '\n\t' + conditional_tParams_dict_conversion[t] + '=np.array(['+ ', '.join(targs) + '])'
        
        
        function_signture ='def ' + function_name + '(' + ', '.join(arguments) + '):'
        return_statment ='\n\treturn dict('
        for t in conditional_tParams_dict_conversion.values():
            return_statment += t +'=' + t 
            if t!= list(conditional_tParams_dict_conversion.values())[-1]:
                return_statment +=','
        return_statment +=')'     
            
        full_function = function_signture + function_lines + return_statment
        if print_out:
            print(full_function)
        if SaveTofile:    
            keep_functions_for_writing_to_file+=full_function + '\n #---------------#  \n'
                
        # Execute the function definition using exec
        exec(full_function, globals())
            
        temp_func = globals()[function_name]
            
            
        #print('creating prior number ' + str(n_func))    
        priors[param_base_name+str(n_func)]= prior_class(temp_func,**reference_params)
            #priors['alpha'+str(n_func)] = ConditionalUniformReveredGaussian(temp_funct, sigma = 1,minimum = 0, maximum=20)
    
    if SaveTofile:         
        print('creating file condition_functions.py' + '. This file will not work as a standalone, you should use it as a referance and minor modifications, if requiered '  )
        with open('condition_functions.py', 'w') as f:
            f.write(keep_functions_for_writing_to_file)
        
        
    if prior_dict_to_add!=None:
        priors.update(prior_dict_to_add)
    return priors 





def create_transdimensional_model(model_function_name, componant_functions_dict, returns_polarization=True,Complex_output=False,print_out=False,SaveTofile=False):
    '''
    

    Parameters
    ----------
    model_function_name : str
        the name of the model function that your heart desires, e.g. model  .
    componant_functions_dict : Dict
        e.g.
        
        define the functions 
        
        def gaussian(frequency_array, amplitude, f0, Q, phi0, e):    
            return frequency_array

        def sin(frequency_array,a,b):
            return frequency_array
        
        
        def poly(frequency_array,coef,deg):
            return coef frequency_array**deg 
        
        componant_functions_dict={}
        componant_functions_dict[gaussian]=(2,'amplitude','f0')
        componant_functions_dict[sin]=(2,'a')
        componant_functions_dict[poly]=(5,'coef',True,'deg')
        
        the key should be the function that serevs as a component function. The values are: the maximal number of componant function, 
        and the parameters that gets modified, the rest are condsiderd as normal params ,i.e. not trans-dimensional      
        if True is indicated, it means that this is a special function that uses the number of functions an input to the function itself, 
        the last item in the list indicates what is the name of that special discrete parameter which serves as the number of functions. 
        The universe is elegant, just like this solution, isn't it ?   
        
    returns_polarization : bool , optional
        This a bool intended for models that return plus, cross polarization strains
    print_out : TYPE, optional
         for debug information 
    SaveTofile : TYPE, optional
        for debug information, saves  a file with the model, which a name as the model  

    Returns
    -------
    model_func : function 
        the addition of the function found in the dict.

    '''
    
    # define the function locally in the enclosed function         
    for func in componant_functions_dict.keys():
        globals()[func.__name__] = func
    
    
    dict_globals={}
    print(globals().keys())
    
    model_function_args_list=[]
   
    function_body='\t'    
    #function_body+='print(globals().keys())\n\t'
    #for func in componant_functions_dict.keys():
    #    function_body+='global ' + func.__name__ + '\n\t'    
        
    
    
    # define the arraies to be iterated on
    func_deg={}
    for func in componant_functions_dict.keys():
        args_list = infer_parameters_from_function(func)
        nmax= componant_functions_dict[func][0]
        tparams= componant_functions_dict[func][1:]
        # check if this is a special function, like a polynomial or any other basis function that need the deg as input 
        is_deg_function=False 
        
        if len(tparams)>=2 and isinstance(tparams[-2], bool):
            is_deg_function=True
            deg_param = tparams[-1]
            func_deg[func] = deg_param
            # remove them from the list 
            tparams = tparams[:-2]
            
                       
        
        not_tparams = {element for element in args_list if element not in tparams}
        # check that the trans parameters are there
        if not set(tparams).issubset(args_list):
            raise Exception('wrong transdimensional params given to create the model ')
        
        for t in tparams:
            function_body+=t+'=['
            for n in np.arange(nmax):
                function_body+=t+str(n)
                model_function_args_list.append(t+str(n))
                if n !=nmax-1:
                    function_body+=','
                else:    
                    function_body+=']\n\t'
            
        model_function_args_list+= not_tparams    
        if is_deg_function:
            # make sure we dont include this paramter in teh def line of the function 
            model_function_args_list.remove(deg_param)
        
        
    polarization_modes=' '
    if returns_polarization:
        function_body+='\n\tresult={}\n\t'
        if Complex_output:
            function_body+='result[\'plus\']= np.zeros(x.shape, dtype=\'complex128\')\n\t'
            function_body+='result[\'cross\']= np.zeros(x.shape, dtype=\'complex128\')\n\t'
        else:
            function_body+='result[\'plus\']= np.zeros(x.shape)\n\t'
            function_body+='result[\'cross\']= np.zeros(x.shape)\n\t'
        polarization_modes=['[\'plus\']','[\'cross\']']        
    else:
        function_body+='\n\tresult=np.zeros(x.shape)\n\t'
    for polarization in polarization_modes:    
        
        for func in componant_functions_dict.keys():
            args_list = infer_parameters_from_function(func)
            nmax= componant_functions_dict[func][0]
            tparams= componant_functions_dict[func][1:]
            # this is a special deg function 
            if func in func_deg.keys():
                # remove the last entires from the list, they did their job  
                tparams = tparams[:-2]
                
            
            not_tparams = {element for element in args_list if element not in tparams}
            # write doen th efunction with keywords to prevent errors
            local_arg_list=list(not_tparams)
            local_arg_keywords=list(local_arg_list)
            if func in func_deg.keys():
                # lodate teh relevant index 
                inx = local_arg_list.index(func_deg[func])
                local_arg_list[inx]='n'
            for t in tparams:                                
                local_arg_list.append(t+'[n]')
                local_arg_keywords.append(t) 
            local_arg_keywords = [k+'=' for k in local_arg_keywords]       
            local_arg_list = [i + j for i, j in zip(local_arg_keywords, local_arg_list)]
            
            function_body+= '\n\tfor n in np.arange(int(np.round(' + 'n_'+func.__name__ +  '))):\n\t\t'
            function_body+= '\t\tresult'+polarization+'+='
            function_body+= 'globals()[\''+func.__name__+'\'](x,'+ ','.join(local_arg_list) +')'+polarization + '\n'
            
        
        
        
    return_statment='\n\n\treturn result'                
    
    tdegrees=''
    for componant_func in componant_functions_dict.keys():
        tdegrees+='n_'+componant_func.__name__+','
    
        
    
    function_signuture = 'def ' + model_function_name + '(x,' + tdegrees + ','.join(set(model_function_args_list)) + '):\n'
    
    function_pre_header = 'import numpy as np\n'
    #for func in componant_functions_dict.keys(): 
    #    function_pre_header += 'import '+func.__name__ +'\n'
    
    
    
    full_function = function_pre_header+function_signuture + function_body + return_statment
    
    if print_out:
        print(full_function)
    if SaveTofile:
        print('creating file '+model_function_name+'.py' + '. This file will not work as a standalone, you should use it as a referance and minor modifications, if requiered '  )
        with open(model_function_name+'.py', 'w') as f:
            f.write(full_function)
        
    locals_dict = {}
    exec(full_function,globals(), locals_dict)
    return locals_dict[model_function_name]

            
    # Execute the function definition using exec
    


def create_plain_priors(prior_class,param_base_name,nmax,prior_dict_to_add=None,**reference_params):   
    '''
    helpr function that saves you some time and copy paste, it creates a bunch of simple priors needed for trans-dimensional sampling 
    e.g. if you have a gaussian with mu and sigma, mu might be trans-dimensional conditioal parameter: mu2(mu0,mu1),
    but sigma is just a simple parameter and you need a few of these, sigma0, sigma 1,...   

    Parameters
    ----------
    prior_class : TYPE
        the type of prior you need
    param_base_name : TYPE
        the base  name of your parameter, such as sigma in teh example above 
    nmax : TYPE
        the maximal number of componant functions
    the prior dict that you want to add these priors to 
    **reference_params : TYPE
    the parameter your prior need, like minimum and maximum, etc..  


    Returns
    -------
    
    conditional prior_dict 
       

    '''
    if prior_dict_to_add==None:    
        prior_dict_to_add = bilby.core.prior.dict.ConditionalPriorDict()
    
    for n in np.arange(nmax):
        #print('creating prior number ' + str(n))   
        prior_dict_to_add[param_base_name+str(n)]= prior_class(**reference_params)
    
    
    
    return prior_dict_to_add 


  

def extract_maximal_likelihood_param_values(result,median=False,mean=False,model=None):
    '''
    

    Parameters
    ----------
    result : TYPE
        The result object received froim the run_smapler function.
    median: bool
        returns the median sample 
    mean: bool
        returns the mean sample         
    model : TYPE
        The model function used 

    Returns
    -------
    Dict[model parameters] = values which hold the highest log_likelihood value 

    '''
    
    # check input 
    if not isinstance(median, bool):
        Exception('extract_maximal_likelihood_param_values: median should be a boolean')
    if not isinstance(mean, bool):
        Exception('extract_maximal_likelihood_param_values: mean should be a boolean')    
    
    
    
    result=_fix_posterior_if_needed(result)
    
    post_sorted = result.posterior.sort_values(by='log_likelihood',ascending=False).reset_index()
    best_params_post = post_sorted.iloc[0].to_dict()
    
    if median:
        best_params_post = post_sorted.median().to_dict()
    if mean:
        best_params_post = post_sorted.mean().to_dict()    
    
    model_parameters = best_params_post
    if model is not None:
        needed_params = infer_parameters_from_function(model)
    
        model_parameters = {k: 0 for k in needed_params} # these are the ghost params + needed params 
        for k in model_parameters.keys():
            if k in best_params_post.keys():
                model_parameters[k]=best_params_post[k]
                
    return  model_parameters
    

def _fix_posterior_if_needed(result):
    # bilby throw an excpetion no matter how we approch it, let's go overboard with this  
    try:
        if result.posterior is None:
        # there was a problem
        # add the sampling 
            result.posterior=bilby.result.rejection_sample(result.nested_samples,result.nested_samples.weights)
            print('no posterior is present, applied rejection_sample to retreive it!! ')
    except:
        result.posterior=bilby.result.rejection_sample(result.nested_samples,result.nested_samples.weights)
        print('no posterior is present, applied rejection_sample to retreive it!! ')
    
    return result   
    

def preprocess_results(result_in: bilby.core.result.Result,model_dict,remove_ghost_samples=True,return_samples_of_most_freq_component_function=True): 
    '''
    

    Parameters
    ----------
    result_in : bilby.core.result.Result
        The result object received froim the run_smapler function.
    model_dict : TYPE
        The model dictionary used to build the model function.
    remove_ghost_samples : TYPE, optional
        DESCRIPTION. The default is True.
        The will set all the entries that hav higher value then the number of component to NaN
    return_samples_of_most_freq_component_function : TYPE, optional
        DESCRIPTION. The default is True.
        This option will clean the posterior from all the samples that are not in the bin with highest BF (or most freqent visited in transdimensional sampling language)

    Returns
    -------
    TYPE
        proccessed result object, columns of the posterior

    '''
    result=result_in # copy to keep it seperate 
    
    result=_fix_posterior_if_needed(result)
    
    # identify the discrete variables 
    discrete_parameters = []
    for p in result.priors.keys():
        if type(result.priors[p])==DiscreteUniform:
            discrete_parameters.append(p)
    
    if  len(discrete_parameters)==0:
        print('coudlnt find any discrete parameters, maybe thiis is not a real transdimensional sampleing ? quitting this function, nothing else matters  ')
        return result_in, list(result_in.posterior.columns)
    
    map_discrete_to_parameters={}
    discrete_parameters_max={}   
    for k in model_dict.keys():   
        dp = 'n_' + k.__name__
        map_discrete_to_parameters[dp] = model_dict[k]
        discrete_parameters_max[dp] = model_dict[k][0]
   
    
    def sublst(row):
        
        for dp in map_discrete_to_parameters.keys():
            params = map_discrete_to_parameters[dp]
            for p in params:
                if not isinstance(p,str):
                    continue
                for val in np.arange(row[dp],discrete_parameters_max[dp]):
                    row[p+str(int(val))]=np.nan # kill the extra entries         
        return row


        
    
    if remove_ghost_samples:
        result.posterior = result.posterior.apply(sublst,axis=1).reset_index(drop=True)
       
    
            
    if return_samples_of_most_freq_component_function:
        print('grouping by the discrete parameters')
        print(result.posterior.groupby(discrete_parameters).size().sort_values(ascending=False))
        vals = result.posterior.groupby(discrete_parameters).size().sort_values(ascending=False).index[0]    
        if  isinstance(vals, (float,np.float64) ):
            vals=[vals]
        
        for p_name,val in zip(discrete_parameters,vals):
            result.posterior=result.posterior[result.posterior[p_name]==int(val)]
        result.posterior.dropna(inplace=True,axis=1)
        result.posterior.reset_index(drop=True,inplace=True)
            
    full_list = list( result.priors.keys())
    for p in discrete_parameters:
        full_list.remove(p)
    
     
    cols_set =set(list(result.posterior.columns)).intersection(set(full_list))        
            
    return  result,list(cols_set)



def _recluster_one_dim_posterior_experimental_use_it_wisely(result,dict_functions_list):
    '''
    Dont use this !!!!

    Parameters
    ----------
    result : TYPE
        DESCRIPTION.
    dict_functions_list : TYPE
        DESCRIPTION.

    Returns
    -------
    res_dict : TYPE
        DESCRIPTION.

    '''
    # this probably doesnt work yet 
    from sklearn.cluster import KMeans
    #loop over the discrete variable found in param_dict
    # for each value of the discrete patameter take the values from param_dict[discrete]  = parameter
    # essamble all of them together and cluster them according to the value of the discrete parameter
    # each sample should spreadaround equally into the clusters ?? maybe not ?? not sure yet 
    res_dict={}
    for function in dict_functions_list.keys():
        discrete_param= dict_functions_list[function]
        params_to_clusetr_almost = infer_parameters_from_function(function)
            
        
        for n_clusters in set(result.posterior[discrete_param].values.astype(int)):            
            relevent_df = result.posterior[result.posterior[discrete_param]==n_clusters]
            data=[]
            
            list_of_params_to_clusetr=[]
            for param in params_to_clusetr_almost:
                params_to_clusetr=[]
                for n in np.arange(n_clusters):                                
                    if param+str(n) in relevent_df.columns:
                        params_to_clusetr.append(param+str(n)) # take only the trans-dimenstional one, the rest are common they will not provide additional infomration ot the clustering algo 
                        list_of_params_to_clusetr.append(param+str(n))
                        
                if len(params_to_clusetr) > 0:        
                    data.append(relevent_df[params_to_clusetr].values.reshape(-1,1))
                    
                # extract the data 
                
            
            data = np.squeeze(np.stack(data,axis=1))
            if len(data.shape)==1:
                data=data.reshape(-1,1)
                
            
            est = KMeans(n_clusters=n_clusters,init='k-means++', n_init="auto")
            est.fit(data)                        
            small_dict = {}
            
            params_to_iter = _group_params_by_numerical_ending(list_of_params_to_clusetr)
            
            for label,params in zip(set(est.labels_),params_to_iter):
                
                Inx = est.labels_==label                
                small_dict[label] = pd.DataFrame(columns = params , data =  data[Inx])
                
                
            res_dict[discrete_param]={(function,n_clusters): small_dict  }
            
    return res_dict       
            
            
            
def _group_params_by_numerical_ending(arr):
    groups = {}
    for word in arr:
        # Extract the last character(s) from the word
        if word[-2].isdigit():
            ending = word[-2:]
        elif word[-1].isdigit():
            ending = word[-1]
        else:
            # Skip words without numerical endings
            continue
        # If the ending is not already in the groups, create a new list for it
        if ending not in groups:
            groups[ending] = []
        # Append the word to the list corresponding to its ending
        groups[ending].append(word)
    # Convert the dictionary to a list of tuples
    result = [ words for ending, words in groups.items()]
    return result            
        
    
    
    
    
    
    
    
    
    

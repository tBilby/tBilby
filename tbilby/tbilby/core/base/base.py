import numpy as np
import pandas as pd

from bilby.core.utils import infer_parameters_from_function
import bilby
from tbilby.core.prior import DiscreteUniform





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





def create_transdimensional_model(model_function_name, componant_functions_dict, returns_polarization=True,print_out=False,SaveTofile=False):
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
        
        componant_functions_dict={}
        componant_functions_dict[gaussian]=(2,'amplitude','f0')
        componant_functions_dict[sin]=(2,'a').
        
        the key should be the function and the value are the maximal number of componant function, and the parameters that gets modified, the rest are condsiderd as normal params    
        
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
    for func in componant_functions_dict.keys():
        args_list = infer_parameters_from_function(func)
        nmax= componant_functions_dict[func][0]
        tparams= componant_functions_dict[func][1:]
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
    polarization_modes=' '
    if returns_polarization:
        function_body+='\n\tresult={}\n\t'
        function_body+='result[\'plus\']= np.zeros(x.shape, dtype=\'complex128\')\n\t'
        function_body+='result[\'cross\']= np.zeros(x.shape, dtype=\'complex128\')\n\t'
        polarization_modes=['[\'plus\']','[\'cross\']']        
    else:
        function_body+='\n\tresult=np.zeros(x.shape)\n\t'
    for polarization in polarization_modes:    
        
        for func in componant_functions_dict.keys():
            args_list = infer_parameters_from_function(func)
            nmax= componant_functions_dict[func][0]
            tparams= componant_functions_dict[func][1:]
            not_tparams = {element for element in args_list if element not in tparams}
            # write doen th efunction with keywords to prevent errors
            local_arg_list=list(not_tparams)
            local_arg_keywords=list(local_arg_list)
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


  

def extract_maximal_likelihood_param_values(result,model):
    '''
    

    Parameters
    ----------
    result : TYPE
        The result object received froim the run_smapler function.
    model : TYPE
        The model function used 

    Returns
    -------
    Dict[model parameters] = values which hold the highest log_likelihood value 

    '''
    
    result=_fix_posterior_if_needed(result)
    
    post_sorted = result.posterior.sort_values(by='log_likelihood',ascending=False).reset_index()
    best_params_post = post_sorted.iloc[0].to_dict()
    needed_params = infer_parameters_from_function(model)
    
    model_parameters = {k: 0 for k in needed_params} # these are the ghost params + needed params 
    for k in model_parameters.keys():
        if k in best_params_post.keys():
            model_parameters[k]=best_params_post[k]
    
    return  model_parameters
    

def _fix_posterior_if_needed(result):
    if result.posterior is None:
        # there was a problem
        # add the sampling 
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
        proccessed result object.

    '''
    result=result_in # copy to keep it seperate 
    
    result=_fix_posterior_if_needed(result)
    
    # identify the discrete variables 
    discrete_parameters = []
    for p in result.priors.keys():
        if type(result.priors[p])==DiscreteUniform:
            discrete_parameters.append(p)
    
    map_discrete_to_parameters={}
    for k in model_dict.keys():        
        map_discrete_to_parameters['n_' + k.__name__] = model_dict[k]
    
    discrete_parameters_max={}    
    for dp in discrete_parameters:
        discrete_parameters_max[dp] = int(result.posterior[dp].max())
    
    
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
        result.posterior = result.posterior.apply(sublst,axis=1)
       
    
            
    if return_samples_of_most_freq_component_function:
        vals = result.posterior.groupby(discrete_parameters).size().sort_values(ascending=False).index[0]    
        for p_name,val in zip(discrete_parameters,vals):
            result.posterior=result.posterior[result.posterior[p_name]==int(val)]
        result.posterior.dropna(inplace=True,axis=1)
            
    full_list = list( result.priors.keys())
    for p in discrete_parameters:
        full_list.remove(p)
    
     
    cols_set =set(list(result.posterior.columns)).intersection(set(full_list))        
            
    return  result,list(cols_set)

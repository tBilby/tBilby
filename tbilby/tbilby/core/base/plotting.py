import bilby
import re
from tbilby.core.prior import DiscreteUniform
from bilby.core.utils import infer_parameters_from_function


def corner_plot_discrete_params(result,SaveTofile=False,filename=None):
    
    

    discrete_parameters = []
    for p in result.priors.keys():
        if type(result.priors[p])==DiscreteUniform:
            discrete_parameters.append(p)
            if result.posterior[p].max()-result.posterior[p].min() ==0:
                result.posterior.iloc[0][p]=result.posterior.iloc[0][p]*1.00001
            
            
    # check for range issues, and fix it by changing a single value by 0.0001%
        
    result.plot_corner(discrete_parameters,SaveTofile=SaveTofile,filename=filename )     
    
    
def corner_plot_single_transdimenstional_param(result,param,SaveTofile=False,filename=None):
    
    cols=list(result.priors.keys())   
    
    if isinstance(param, str):
        param=[param]
        
    for p in param:
        locs_params = sorted(_extract_words_with_numeric_suffix(partial_words=[p], full_words=cols))
        if filename is not None:
            result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=filename )
        else:
            result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=p+'.png' )

def _extract_words_with_numeric_suffix(partial_words, full_words):
    extracted_words = []
    for partial in partial_words:
        for word in full_words:
            if partial in word:
                match = re.match(rf"{re.escape(partial)}(\d+)", word)
                if match:
                    extracted_words.append(word)
    return extracted_words

    
def corner_plot_single_transdimentional_component_functions(result,function,SaveTofile=False,filename=None):
   
    params = infer_parameters_from_function(function)
    params_to_plot =[]
    all_parmas = list(result.priors.keys())
    
    #extracted_parmas = [word for word in all_parmas if any(partial in word for partial in params)]
    extracted_parmas =_extract_words_with_numeric_suffix(params,all_parmas)
    
    locs_params = sorted(extracted_parmas)
    result.plot_corner(locs_params,SaveTofile=SaveTofile,filename=filename )    
    
def corner_plot_single_component_function(result,function,order,SaveTofile=False,filename=None):        
    params = infer_parameters_from_function(function)
    params = [p+str(order) for p in params]
    result.plot_corner(params,SaveTofile=SaveTofile,filename=filename )    
    
    

from abc import ABC, abstractmethod
from .TransInterped import TransInterped

import bilby
import os
import numpy as np


def create_cond_function(parameter_name,prior_class_name,componant_function_number,nested_conditional_transdimensional_params,conditional_transdimensional_params,conditional_params,SaveTofile=False):
    
    function_name="condition_func"+prior_class_name +'_' +  '_' + parameter_name # helps to seperate different priors and params  
    arguments = ["reference_params"]
    function_lines=''
    keep_functions_for_writing_to_file=''
    
    for t in nested_conditional_transdimensional_params:
        targs=[]               
        for n in np.arange(componant_function_number): #np.arange(2)= [1], so it takes the lower level automaticaly without n-1  
            arguments.append(t+str(n))
            targs.append(t+str(n))
        function_lines += '\n\t' + t + '=np.array(['+ ', '.join(targs) + '])'
        
    # take care of the non nested trans params     
    if len(conditional_transdimensional_params) > 0: # check we got something
        if isinstance(conditional_transdimensional_params,dict):
            for t in conditional_transdimensional_params.keys():
                targs=[]               
                for n in np.arange(conditional_transdimensional_params[t]): # this is independed of the number of componant functions 
                    arguments.append(t+str(n))
                    targs.append(t+str(n))
                function_lines += '\n\t' + t + '=np.array(['+ ', '.join(targs) + '])' 
                                   
        elif isinstance(conditional_transdimensional_params, list):
            for t in conditional_transdimensional_params:
                targs=[]               
                for n in np.arange(componant_function_number+1): #+1 so it takes the last  level as well
                    arguments.append(t+str(n))
                    targs.append(t+str(n))
                function_lines += '\n\t' + t + '=np.array(['+ ', '.join(targs) + '])'             
        else:
            raise Exception('conditional_transdimensional_params is not alist or a dict, not sure how to handle this, sorry.   ')

    
    
      
        
    
    # add the non transdimensions params
    arguments +=list(conditional_params)
    
    function_signture ='def ' + function_name + '(' + ', '.join(arguments) + '):'
    return_statment ='\n\treturn dict('
    # nested conditional
    for t in nested_conditional_transdimensional_params:
        return_statment += t +'=' + t 
        if t!= list(nested_conditional_transdimensional_params)[-1]:
            return_statment +=','
    
    # non - nested conditional    
    if isinstance(conditional_transdimensional_params,dict): # turn into a list, we are done with the dict 
        conditional_transdimensional_params = list(conditional_transdimensional_params.keys()).copy()
    if(len(conditional_transdimensional_params)>0):
        return_statment +=',' 
        
    for t in conditional_transdimensional_params:
        return_statment += t +'=' + t 
        if t!= list(conditional_transdimensional_params)[-1]:
            return_statment +=','            
            
    # normal conditional            
    if(len(conditional_params)>0):
        return_statment +=',' 
    
    for v in conditional_params:        
        return_statment += v +'=' + v 
        if v!= list(conditional_params)[-1]:
            return_statment +=','
            
            
            
    return_statment +=')'     
        
    full_function = function_signture + function_lines + return_statment
    
    if SaveTofile:    
        keep_functions_for_writing_to_file+=full_function + '\n #---------------#  \n'
        # check for folder 
        if os.path.isdir('condition_functions'):        
            with open(f'condition_functions/{function_name}.py', 'w') as f:
                f.write(keep_functions_for_writing_to_file)    
        else:
            print('Warning: condition function will not be written to disk, a folder named: condition_functions is missing  ')
    # Execute the function definition using exec
    exec(full_function, globals())
        
    cond_func = globals()[function_name]
    return cond_func




def transdimensional_conditional_prior_factory(conditional_prior_class):
    class TransdimensionalConditionalPrior(conditional_prior_class,ABC):
        def __init__(self, name, componant_function_number,nested_conditional_transdimensional_params,conditional_transdimensional_params,conditional_params,debug_print_out=False,latex_label=None, unit=None,
                     boundary=None, **reference_params):
            
            # this fixmakes it possible to open the results.json, casue an issue with the names..  
            self.orig_name=self.__class__.__name__   
            self.debug_print_out=debug_print_out         
            
            """

            Parameters
            ==========
            condition_func: func
                Functional form of the condition for this prior. The first function argument
                has to be a dictionary for the `reference_params` (see below). The following
                arguments are the required variables that are required before we can draw this
                prior.
                It needs to return a dictionary with the modified values for the
                `reference_params` that are being used in the next draw.
                For example if we have a Uniform prior for `x` depending on a different variable `y`
                `p(x|y)` with the boundaries linearly depending on y, then this
                could have the following form:

                .. code-block:: python

                    def condition_func(reference_params, y):
                        return dict(
                            minimum=reference_params['minimum'] + y,
                            maximum=reference_params['maximum'] + y
                        )

            name: str, optional
               See superclass
            latex_label: str, optional
                See superclass
            unit: str, optional
                See superclass
            boundary: str, optional
                See superclass
            reference_params:
                Initial values for attributes such as `minimum`, `maximum`.
                This differs on the `prior_class`, for example for the Gaussian
                prior this is `mu` and `sigma`.
            """
            # you have to run it in this ugly way since some things are not defined before the init and you these things in order to constructteh function 
            cls_name = 'Transdimensional{}'.format(conditional_prior_class.__name__)  
            cond_func =create_cond_function(name,cls_name,componant_function_number,nested_conditional_transdimensional_params,conditional_transdimensional_params,conditional_params,SaveTofile=self.debug_print_out)                                                    
            globals()[cond_func.__name__] = cond_func
            #cond_func = self.create_condition_function(name,cls_name,componant_function_number,conditional_transdimensional_params,conditional_params)
                
            super(TransdimensionalConditionalPrior,self).__init__(cond_func,name=name, latex_label=latex_label,
                                                   unit=unit, boundary=boundary, **reference_params)
            
            #tries to vercome to defining in a certain sapce issue
            self.condition_func = cond_func
            self.__class__.__name__  = self.orig_name
            self.__class__.__qualname__ = self.orig_name
            #self.__class__.__name__ = cls_name
            #self.__class__.__qualname__ = 'Transdimensional{}'.format(conditional_prior_class.__qualname__)
            self.transdimesional_params_data_holder_dict={}
            
            for i in list(nested_conditional_transdimensional_params)+list(conditional_params):
                self.transdimesional_params_data_holder_dict[i] = None
            
            self.componant_function_number=componant_function_number
            self.nested_conditional_transdimensional_params=nested_conditional_transdimensional_params
            self.conditional_transdimensional_params=conditional_transdimensional_params
            self.conditional_params=conditional_params
            
        def update_conditions(self, **required_variables):    
            # this will update the params 
            super(TransdimensionalConditionalPrior,self).update_conditions(**required_variables)
            parameters = self.transdimensional_condition_function(**required_variables) 
            for key, value in parameters.items():
                if key in self.transdimesional_params_data_holder_dict.keys():
                    self.transdimesional_params_data_holder_dict[key]=value
                else:
                    if hasattr(self,key):                       
                            setattr(self, key, value)
                    else:
                        raise Exception("Expected kwargs for {} or {}. Got kwargs for {} instead."
                                                            .format(self.required_variables,self.transdimesional_params_data_holder_dict.keys(),
                                                                    list(parameters.keys())))

        
                
        def __repr__(self):
             """Overrides the special method __repr__.

             Returns a representation of this instance that resembles how it is instantiated.
             Works correctly for all child classes

             Returns
             =======
             str: A string representation of this instance

             """
             prior_name = self.__class__.__name__
             instantiation_dict = self.get_instantiation_dict()                  
             args = ', '.join(['{}={}'.format(key, repr(instantiation_dict[key]))
                               for key in instantiation_dict])
             return "{}({})".format(prior_name, args)       
                
            
        #def create_condition_function(name,class_name,componant_function_number,conditional_transdimensional_params,conditional_params):
            
        #    return tbilby.create_cond_function(name,class_name,componant_function_number,conditional_transdimensional_params,conditional_params,SaveTofile=True)
            #def func(reference_params,**required_variables): 
                
            #    return {'m':'m'}
            #return func
        @abstractmethod
        def transdimensional_condition_function(self,**required_variables):
            pass
        
        
        
        
            #return {}
    # the rest of the method are alrady defined by conditional prior,     

    return TransdimensionalConditionalPrior



class TransdimensionalConditionalUniform(transdimensional_conditional_prior_factory(bilby.core.prior.ConditionalUniform)):
    pass


class TransdimensionalConditionalBeta(transdimensional_conditional_prior_factory(bilby.core.prior.ConditionalBeta)):
    pass

class TransdimensionalConditionalInterped(transdimensional_conditional_prior_factory(bilby.core.prior.ConditionalInterped)):
    pass

class ConditionalTransInterped(bilby.prior.conditional.conditional_prior_factory(TransInterped)):
    pass

class TransdimensionalConditionalTransInterped(transdimensional_conditional_prior_factory(ConditionalTransInterped)):
    pass



import bilby
import json 


import os

def _generate_unique_name(directory, base_name):
    # Split the base name into filename and extension
    filename, ext = os.path.splitext(base_name)
    
    # Initialize counter
    counter = 1
    
    # Generate a unique name
    while os.path.exists(os.path.join(directory, f"{filename}{counter}{ext}")):
        counter += 1
    
    # Return the unique name
    return f"{filename}{counter}{ext}"


def read_in_result(filename=None, outdir=None, label=None, extension='json', gzip=False, result_class=None):

        
    """
    Reads in a stored bilby result object  
    this is a tbilby wrapper designed to avoid issues with loading local priors that don't exist in the current script (e.g., loaded to create some plots, etc.).
    
    
    Parameters
    ==========
    filename: str
        Path to the file to be read (alternative to giving the outdir and label)
    outdir, label, extension: str
        Name of the output directory, label and extension used for the default
        naming scheme.
    result_class: bilby.core.result.Result, or child of
        The result class to use. By default, `bilby.core.result.Result` is used,
        but objects which inherit from this class can be given providing
        additional methods.
    
    """    
    
    if extension!='json':
        raise Exception('This function support json format only, sorry !!')
    
    result = None 
    try:
        # try the bilby way, if fail try another thing  
        result =  bilby.read_in_result(filename=filename,outdir=outdir,label=label,extension=extension,gzip=gzip,result_class=result_class)                
    except TypeError as e:  
        
        str_probelm= "Prior.__init__() got an unexpected keyword argument 'componant_function_number'"
        # this is the problem, we are handleing here, if this is not the problem, let the user know  
        if str(e) != str_probelm:
            
            print('we can deal with ' + str_probelm + ' but this is something else')
            raise e
        
        
        with open(filename, 'r') as file:
                data = json.load(file)
                
                
        default_kwargs= ['minimum', 'maximum', 'name', 'latex_label', 'unit', 'boundary']

        for p in data['priors'].keys():
            
            if isinstance(data['priors'][p],dict):
                if 'componant_function_number' in data['priors'][p]['kwargs'].keys(): 
                    new_dict={}
                    
                    for k in default_kwargs:
                        if k in data['priors'][p]['kwargs']:                       
                            new_dict[k]= data['priors'][p]['kwargs'][k]                    
                    data['priors'][p]['kwargs'] = new_dict
        
        
        directory, filename = os.path.split(filename)       
        base_name = "modified_bilby_to_tbilby_result_file_tmp.json"
        unique_name = _generate_unique_name(directory, base_name)
        
        full_unique_name =os.path.join(directory, unique_name)
                    
        with open(full_unique_name, 'w') as file:
                json.dump(data, file)
        try:        
            result = bilby.read_in_result(filename=full_unique_name,outdir=outdir,label=label,extension=extension,gzip=gzip,result_class=result_class)    
        except Exception as inner_exception:
            raise inner_exception
        
       
   
    finally: 
        # clean up 
        if os.path.exists(full_unique_name):
            os.remove(full_unique_name)
 
        return result
 
    
 
    
       
import json
from pathlib import Path
import os
from core.util_prior import create_prior_from_parameter

def open_config(path_config_dir):
    with open(Path(path_config_dir) / 'config.json','r') as f:
        config_file = json.loads(f.read())
    return config_file

def create_dir(path):
    
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.mkdir(OUTPUT_DIR)
        except FileExistsError:
            print('Avoided error')
    return 

def read_forward_model_from_config(config_file,params,params_names,extract_param=False):
    
    chem_params = {}
    temp_params = {}
    clouds_params = {}
    physical_params = {}
    data_params = {}
    
    RANGE={}
    LOG_PRIORS={}
    CUBE_PRIORS={}

    # free model
    if config_file['retrieval']['FM']['model'] == 'free':
        # chemistry
        if config_file['retrieval']['FM']['chemistry']['model'] == 'free':
            abundance_list = list(config_file['retrieval']['FM']['chemistry']['parameters'].keys())
            for abund in abundance_list:
                if config_file['retrieval']['FM']['chemistry']['parameters'][abund]['model'] == 'constant':
                    parameter_data=config_file['retrieval']['FM']['chemistry']['parameters'][abund]['param_0']
                    if isinstance(parameter_data,list):
                        param = '%s___%s___%s' % (abund,'constant','param_0')
                        create_prior_from_parameter(
                            param=param,
                            parameter_data=parameter_data,
                            RANGE=RANGE,
                            LOG_PRIORS=LOG_PRIORS,
                            CUBE_PRIORS=CUBE_PRIORS)
                        if extract_param:
                            chem_params[param] = params[params_names.index(param)]
                    else:
                        chem_params[param] = parameter_data
                else:
                    print('Only constant abundance profiles are currently taken into account')
        elif config_file['retrieval']['FM']['chemistry']['model'] == 'chem_equ':
            print('chem_equ model not yet taken into account')
        # p-T
        if config_file['retrieval']['FM']['p-T']['model'] == 'guillot':
            guillot_params = ['t_equ','t_int','log_gravity','log_kappa_IR','log_gamma']
            for param in guillot_params:
                if param in config_file['retrieval']['FM']['p-T']['parameters'].keys():
                    parameter_data=config_file['retrieval']['FM']['p-T']['parameters'][param]
                    if isinstance(parameter_data,list):
                        create_prior_from_parameter(
                            param=param,
                            parameter_data=parameter_data,
                            RANGE=RANGE,
                            LOG_PRIORS=LOG_PRIORS,
                            CUBE_PRIORS=CUBE_PRIORS)
                        if extract_param:
                            temp_params[param] = params[params_names.index(param)]
                    else:
                        temp_params[param] = parameter_data
        # clouds
        if config_file['retrieval']['FM']['clouds']['model'] == 'ackermann':
            print('Ackermann model not yet taken into account')
        
        # physical
        for param in config_file['retrieval']['FM']['physical'].keys():
            parameter_data=config_file['retrieval']['FM']['physical'][param]
            if isinstance(parameter_data,list):
                create_prior_from_parameter(
                    param=param,
                    parameter_data=parameter_data,
                    RANGE=RANGE,
                    LOG_PRIORS=LOG_PRIORS,
                    CUBE_PRIORS=CUBE_PRIORS)
                if extract_param:
                    physical_params[param] = params[params_names.index(param)]
            else:
                physical_params[param] = parameter_data
    else:
        print('Only free FM taken into account')
    
    # data
    for data_key in config_file['retrieval']['data'].keys():
        parameter_data=config_file['retrieval']['data'][data_key]['flux_scaling']
        if isinstance(parameter_data,list):
            param = '%s___%s' % (data_key,'flux_scaling')
            create_prior_from_parameter(
                param=param,
                parameter_data=parameter_data,
                RANGE=RANGE,
                LOG_PRIORS=LOG_PRIORS,
                CUBE_PRIORS=CUBE_PRIORS)
            if extract_param:
                data_params[param] = params[params_names.index(param)]
        else:
            data_params[param] = parameter_data
        
        parameter_data=config_file['retrieval']['data'][data_key]['error_scaling']
        if isinstance(parameter_data,list):
            param = '%s___%s' % (data_key,'error_scaling')
            create_prior_from_parameter(
                param=param,
                parameter_data=parameter_data,
                RANGE=RANGE,
                LOG_PRIORS=LOG_PRIORS,
                CUBE_PRIORS=CUBE_PRIORS)
            if extract_param:
                data_params[param] = params[params_names.index(param)]
        else:
            data_params[param] = parameter_data
    
    if extract_param:
        return chem_params,temp_params,clouds_params,physical_params,data_params
    else:
        return RANGE,LOG_PRIORS,CUBE_PRIORS
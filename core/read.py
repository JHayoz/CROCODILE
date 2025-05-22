import json
from pathlib import Path
import os
from core.util_prior import create_prior_from_parameter
import pickle
import pymultinest
import numpy as np

def open_config(path_config_dir):
    with open(Path(path_config_dir) / 'config.json','r') as f:
        config_file = json.loads(f.read())
    return config_file

def create_dir(path):
    dir_path = Path(path)
    dir_path.mkdir(exist_ok=True,parents=True)
    return 

def read_samples(filname):
    samples_path = Path(filname)
    if not samples_path.exists():
        print('No samples found for this retrieval')
        return None
    with open(samples_path,'rb') as f:
        samples = pickle.load(f)
    return samples
def retrieve_samples(config_file,output_dir):
    
    range_prior,_,_ = read_forward_model_from_config(config_file,params=[],params_names=[],extract_param=False)
    params_names = list(range_prior.keys())
    
    n_params = len(params_names)
    OUTPUT_DIR = str(Path(output_dir)) + '/'
    
    # create analyzer object
    a = pymultinest.Analyzer(n_params, outputfiles_basename = OUTPUT_DIR)
    
    stats = a.get_stats()
    bestfit_params = a.get_best_fit()
    samples = np.array(a.get_equal_weighted_posterior())[:,:-1]
    
    return samples

def save_samples(samples,output_dir,overwrite=False):
    samples_path = Path(output_dir) / 'SAMPLESpos.pickle'
    if (not samples_path.exists()) or overwrite:
        f = open(samples_path,'wb')
        pickle.dump(samples,f)
        f.close()
        print('Samples saved')
    else:
        print('Samples file exists already. Please use overwrite=True if you want to overwrite the samples.')

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
                    param = '%s___%s___%s' % (abund,'constant','param_0')
                    if isinstance(parameter_data,list):
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
            chem_equ_params = ['C/O','FeHs','log_Pquench_carbon']
            for param in chem_equ_params:
                if param in config_file['retrieval']['FM']['chemistry']['parameters'].keys():
                    parameter_data=config_file['retrieval']['FM']['chemistry']['parameters'][param]
                    if isinstance(parameter_data,list):
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
        # p-T
        if config_file['retrieval']['FM']['p-T']['model'] == 'guillot':
            guillot_params = ['t_equ','t_int','log_kappa_IR','log_gamma','P0']#,'log_gravity']
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
            ackermann_params = ['sigma_lnorm','fsed','log_kzz']
            for param in ackermann_params:
                if param in config_file['retrieval']['FM']['clouds']['parameters'].keys():
                    parameter_data=config_file['retrieval']['FM']['clouds']['parameters'][param]
                    if isinstance(parameter_data,list):
                        create_prior_from_parameter(
                            param=param,
                            parameter_data=parameter_data,
                            RANGE=RANGE,
                            LOG_PRIORS=LOG_PRIORS,
                            CUBE_PRIORS=CUBE_PRIORS)
                        if extract_param:
                            clouds_params[param] = params[params_names.index(param)]
                    else:
                        clouds_params[param] = parameter_data
            # cloud abundances
            clouds_params['cloud_abunds'] = {}
            for param in config_file['retrieval']['FM']['clouds']['opacities']:
                if param in config_file['retrieval']['FM']['clouds']['parameters'].keys():
                    cloud_param_name = param.split('_')[0] # cloud opacities are always called Mg2SiO4(c)_cm but their abundances are called Mg2SiO4(c)
                    parameter_data=config_file['retrieval']['FM']['clouds']['parameters'][param]
                    if isinstance(parameter_data,list):
                        create_prior_from_parameter(
                            param=param,
                            parameter_data=parameter_data,
                            RANGE=RANGE,
                            LOG_PRIORS=LOG_PRIORS,
                            CUBE_PRIORS=CUBE_PRIORS)
                        if extract_param:
                            clouds_params['cloud_abunds'][cloud_param_name] = params[params_names.index(param)]
                    else:
                        clouds_params['cloud_abunds'][cloud_param_name] = parameter_data
            # scattering
            clouds_params['scattering'] = config_file['retrieval']['FM']['clouds']['scattering_emission']
        elif config_file['retrieval']['FM']['clouds']['model'] == 'grey_deck':
            cloud_deck_params = ['log_Pcloud']
            for param in cloud_deck_params:
                if param in config_file['retrieval']['FM']['clouds']['parameters'].keys():
                    parameter_data=config_file['retrieval']['FM']['clouds']['parameters'][param]
                    if isinstance(parameter_data,list):
                        create_prior_from_parameter(
                            param=param,
                            parameter_data=parameter_data,
                            RANGE=RANGE,
                            LOG_PRIORS=LOG_PRIORS,
                            CUBE_PRIORS=CUBE_PRIORS)
                        if extract_param:
                            clouds_params[param] = params[params_names.index(param)]
                    else:
                        clouds_params[param] = parameter_data
        
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
        param = '%s___%s' % (data_key,'flux_scaling')
        if isinstance(parameter_data,list):
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
        param = '%s___%s' % (data_key,'error_scaling')
        if isinstance(parameter_data,list):
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
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 12:01 2025

@author: Jean Hayoz
"""
print('Starting imports')

import os
os.environ['OMP_NUM_THREADS'] = '1'
import pymultinest
import pickle
import numpy as np
import json
import sys
from pathlib import Path

# from config_petitRADTRANS import *
# os.environ["pRT_input_data_path"] = OS_ABS_PATH_TO_OPACITY_DATABASE

from core.priors import Prior
from core.data import Data
from core.read import open_config,create_dir
from core.retrievalClass import Retrieval

print('... DONE')
pc_to_meter = 30856775812799588
def main(config_file_path,continue_retrieval):
    
    config_file=open_config(config_file_path)

    # check if the retrieval should continue or not
    output_dir_path = Path(config_file['metadata']['output_dir'])
    live_points_path = output_dir_path / 'live.points'
    # if the file "live.points" already exists, then assume that it should continue and not start over
    cont_retr = live_points_path.exists()
    
    # cont_retr=bool(int(continue_retrieval))
    if cont_retr:
        print('Continuing retrieval called: %s' % config_file['metadata']['retrieval_id'])
    else:
        print('Starting retrieval called: %s' % config_file['metadata']['retrieval_id'])

    # create output directory
    OUTPUT_DIR = config_file['metadata']['output_dir']
    create_dir(OUTPUT_DIR)
    
    # load data
    data_obj = Data(
        photometry_file = config_file['data']['photometry'],
        spectroscopy_files = config_file['data']['spectroscopy']['calib'],
        contrem_spectroscopy_files = config_file['data']['spectroscopy']['contrem'],
        photometry_filter_dir = config_file['data']['filters'])
    
    if 'extinction' in config_file['data'].keys():
        ext = config_file['data']['extinction']
        if not (ext is None):
            data_obj.deredden_all_data(Av=config_file['data']['extinction'])
    
    # rescale CC data
    # scale_factor = 1e10*5e6/(config_file['retrieval']['FM']['physical']['distance']*pc_to_meter)**2
    # scale_factor = 1e20*5e6/(config_file['retrieval']['FM']['physical']['distance']*pc_to_meter)**2
    scale_factor = 1
    data_obj.rescale_CC_data(scale=scale_factor)
    
    
    fig=data_obj.plot(
        config=config_file,
        output_dir=OUTPUT_DIR,
        plot_name = 'data',
        title = 'Data for retrieval %s' % config_file['metadata']['retrieval_id'],
        inset_plot=False,
        plot_errorbars=True,
        save_plot=True)
    
    # load priors
    prior_obj = Prior(config_file)
    prior_obj.plot(output_dir= OUTPUT_DIR)
    
    # prepare retrieval object
    retrieval = Retrieval(
        data_obj=data_obj,
        prior_obj=prior_obj,
        config_file=config_file,
        for_analysis=False,
        continue_retrieval = cont_retr)
    print('Starting Bayesian inference')
    
    # save params names
    json.dump(retrieval.params_names, open(retrieval.output_path / 'params.json', 'w'))
    
    n_params = len(retrieval.params_names)
    
    pymultinest.run(retrieval.lnprob_pymultinest,
                    retrieval.Prior,
                    n_params,
                    outputfiles_basename=str(retrieval.output_path) + '/',
                    resume = cont_retr,
                    verbose = True,
                    n_live_points = config_file['hyperparameters']['multinest']['n_live_points'])
    
    print('############### FINISHING THE SAMPLER #####################')
    
    # create analyzer object
    a = pymultinest.Analyzer(n_params, outputfiles_basename = str(retrieval.output_path) + '/')
    
    stats = a.get_stats()
    bestfit_params = a.get_best_fit()
    samples = np.array(a.get_equal_weighted_posterior())[:,:-1]
    
    f = open(retrieval.output_path / 'SAMPLESpos.pickle','wb')
    pickle.dump(samples,f)
    f.close()
    
    return 

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
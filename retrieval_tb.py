# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 12:01 2025

@author: Jean Hayoz
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import pymultinest
import pickle
import numpy as np
import json
import sys

from core.priors import Prior
from core.data import Data
from core.read import open_config,create_dir
from core.retrievalClass import Retrieval

from config_petitRADTRANS import *
os.environ["pRT_input_data_path"] = OS_ABS_PATH_TO_OPACITY_DATABASE


def main(config_file_path,continue_retrieval):
    
    config_file=open_config(config_file_path)
    
    cont_retr = bool(continue_retrieval)
    
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
    
    fig=data_obj.plot(
        config=config_file,
        output_dir=OUTPUT_DIR,
        plot_name = 'data',
        title = 'Data for retrieval %s' % config_file['metadata']['retrieval_id'],
        inset_plot=False,
        plot_errorbars=False,
        save_plot=not cont_retr)
    
    # load priors
    prior_obj = Prior(config_file)
    prior_obj.plot()
    
    # prepare retrieval object
    retrieval = Retrieval(
        data_obj=data_obj,
        prior_obj=prior_obj,
        config_file=config_file,
        for_analysis=False,
        continue_retrieval = cont_retr)
    print('Starting Bayesian inference')
    
    n_params = len(retrieval.params_names)
    
    pymultinest.run(retrieval.lnprob_pymultinest,
                    retrieval.Prior,
                    n_params,
                    outputfiles_basename=str(retrieval.output_path) + '/',
                    resume = cont_retr,
                    verbose = True,
                    n_live_points = config_file['hyperparameters']['multinest']['n_live_points'])
    
    print('############### FINISHING THE SAMPLER #####################')
    # save positions
    json.dump(retrieval.params_names, open(retrieval.output_path / 'params.json', 'w'))
    
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
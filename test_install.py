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

from config_petitRADTRANS import *
os.environ["pRT_input_data_path"] = OS_ABS_PATH_TO_OPACITY_DATABASE
import petitRADTRANS

from core.priors import Prior
from core.data import Data
from core.read import open_config,create_dir
from core.retrievalClass import Retrieval

print('... DONE')

def main():
    
    target_id='hr2562b'
    data_version='v01'
    retrieval_name='all_data___free___subset_opacities'
    
    config_file_path = '/home/ipa/quanz/user_accounts/jhayoz/Projects/CO_ratio_snowlines/retrievals/%s/%s_data_%s_%s' % (
        target_id,
        target_id,
        data_version,
        retrieval_name)
    
    config_file=open_config(config_file_path)
    continue_retrieval = 'False'
    
    if continue_retrieval == 'False':
        cont_retr = False
    elif continue_retrieval == 'True':
        cont_retr = True
    else:
        print('Second argument needs to be either False or True, written as a string')
    
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
        save_plot=False)
    
    # load priors
    prior_obj = Prior(config_file)
    prior_obj.plot(output_dir= OUTPUT_DIR,save_plot=False)
    
    # prepare retrieval object
    retrieval = Retrieval(
        data_obj=data_obj,
        prior_obj=prior_obj,
        config_file=config_file,
        for_analysis=True,
        continue_retrieval = cont_retr)

    print('Code finished without error.')
    print('CROCODILE successfully installed.')
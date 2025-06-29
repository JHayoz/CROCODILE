# -*- coding: utf-8 -*-
"""
Created on Fri May 09 2025

@author: Jean Hayoz
"""
print('Starting imports')

import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from core.data import Data
from core.priors import Prior
from core.read import open_config,retrieve_samples,save_samples,create_dir,read_samples
from core.retrievalClass import Retrieval
from core.plotting_output import plot_corner,plot_CO_ratio,plot_FeH_ratio,plot_retrieved_temperature_profile,plot_retrieved_spectra,plot_retrieved_teff

print('... DONE')

def main(config_file_path,retrieval_results_dir,plot_output_dir):
    
    output_dir_path = str(Path(plot_output_dir)) + '/'
    create_dir(output_dir_path)
    
    config_file=open_config(config_file_path)
    retrieval_id = config_file['metadata']['retrieval_id']
    savefile_prefix = output_dir_path + retrieval_id + '_'
    
    print('#####################################################')
    print()
    print('Analysing results of retrieval %s' % retrieval_id)
    print('Saving plots under %s' % output_dir_path)
    print('using prefix %s' % str(Path(savefile_prefix).name))
    print()
    print('#####################################################')
    
    # load data
    print('Loading data')
    data_obj = Data(
        photometry_file = config_file['data']['photometry'],
        spectroscopy_files = config_file['data']['spectroscopy']['calib'],
        contrem_spectroscopy_files = config_file['data']['spectroscopy']['contrem'],
        photometry_filter_dir = config_file['data']['filters'])
    
    if 'extinction' in config_file['data'].keys():
        ext = config_file['data']['extinction']
        if not (ext is None):
            data_obj.deredden_all_data(Av=config_file['data']['extinction'])
    
    print('Plotting data')
    fig=data_obj.plot(
        config=config_file,
        output_dir=output_dir_path,
        plot_name = retrieval_id + '_' + '00_data',
        title = 'Data for retrieval %s' % retrieval_id,
        inset_plot=False,
        plot_errorbars=False,
        save_plot=True)
    
    prior_obj = Prior(config_file)
    samples_file_path = Path(retrieval_results_dir + 'SAMPLESpos.pickle')
    samples = []
    if samples_file_path.exists():
        samples = read_samples(samples_file_path)
    if len(samples) < 100:
        samples = retrieve_samples(config_file,retrieval_results_dir)
        
        try:
            save_samples(samples,retrieval_results_dir,overwrite=False)
        except PermissionError:
            print('You do not have the permission to save data here')
    print('Number of available samples:',len(samples))
    print('Plotting cornerplot')
    plot_corner(
        config_file,
        samples,
        param_range = None,
        percent_considered = 0.90,
        output_file = savefile_prefix + '01_corner.png',
        fontsize=12,
        include_abunds = True,
        title = 'Corner for retrieval %s' % retrieval_id,
        save_plot=True)
    
    if config_file['retrieval']['FM']['chemistry']['model'] == 'free':
        print('Plotting C/O ratio')
        plot_CO_ratio(
            config_file,
            samples,
            percent_considered = 1.,
            abundances_considered = 'all',
            output_file = savefile_prefix + '02_COratio.png',
            fontsize = 10,
            lw = 0.5,
            figsize=(3,3),
            color = 'g',
            label='C/O$=$',
            include_quantiles = True,
            title='C/O ratio for retrieval %s' % retrieval_id,
            ax = None,
            save_plot = True)
        print('Plotting Fe/H ratio')
        plot_FeH_ratio(
            config_file,
            samples,
            percent_considered = 1.,
            abundances_considered = 'all',
            output_file = savefile_prefix + '03_FeHratio.png',
            fontsize = 10,
            lw = 0.5,
            figsize=(3,3),
            color = 'g',
            label='[Fe/H]$=$',
            include_quantiles = True,
            title='Fe/H ratio for retrieval %s' % retrieval_id,
            ax = None,
            save_plot = True)
    
    print('Plotting p-T profile')
    plot_retrieved_temperature_profile(
        config_file,
        samples,
        output_file=savefile_prefix + '04_pTprofile.png',
        nb_stds = 3,
        fontsize = 10,
        lw = 0.5,
        figsize=(6,4),
        color = 'g',
        label='T$_{\mathrm{equ}}=$',
        plot_data = True,
        plot_label=True,
        title='p-T for retrieval %s' % retrieval_id,
        ax = None,
        save_plot = True)
    
    print('Loading retrieval object')
    retrieval = Retrieval(
        data_obj=data_obj,
        prior_obj=prior_obj,
        config_file=config_file,
        for_analysis=True,
        continue_retrieval = False)
    
    print('Plotting retrieved SED')
    plot_retrieved_spectra(
        config_file,
        retrieval,
        samples,
        nb_picks=5,
        title='Retrieved SED for retrieval %s' % retrieval_id,
        save_plot = True,
        output_file=savefile_prefix + '05_retrievedSED.png')
    
    # print('Computing retrieved Teff')
    # plot_retrieved_teff(
    #     config_file,
    #     retrieval,
    #     samples,
    #     nb_picks=1000,
    #     wlen_borders = [0.5,10],
    #     title='Retrieved Teff for retrieval %s' % retrieval_id,
    #     save_plot = True,
    #     output_file=savefile_prefix + '06_retrievedTeff.png'
    # )
    print('DONE')
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:40:17 2021

@author: jeanh
"""
# help functions to plot simulated spectra or the results of the retrieval

import matplotlib.pyplot as plt
from corner import corner
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from os import path
from pathlib import Path
from random import sample
from PyAstronomy.pyasl import crosscorrRV,fastRotBroad,rotBroad
import scipy.constants as cst
from seaborn import color_palette

from petitRADTRANS import physics

from core.util import get_abundance_params,calc_CO_ratio,calc_FeH_ratio_from_samples,get_std_percentiles,quantiles_to_string,nice_param_name
from core.read import read_forward_model_from_config
from core.priors import Prior
from core.model import get_temperatures

def plot_corner(
    config_file,
    samples,
    param_range = None,
    percent_considered = 0.90,
    output_file = '',
    fontsize=12,
    include_abunds = True,
    title = 'Retrieval',
    plot_format = 'png',
    save_plot=True):
    
    if plot_format not in ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']:
        plot_format = 'png'
    
    prior_obj = Prior(config_file)
    params_names = prior_obj.params_names
    abundances_names = get_abundance_params(config_file)
    
    nb_iter = len(samples)
    index_consider = int(np.ceil(nb_iter*(1.-percent_considered)))
    
    samples_cut = samples[index_consider:,:]
    
    if not include_abunds:
        samples_cut = samples_cut[:,[index for index in range(len(params_names)) if params_names[index] not in abundances_names]]
        params_names = [param for param in params_names if not param in abundances_names]
    
    n_params = len(params_names)
    corner_range=None
    if param_range is not None:
        corner_range=list([(param_range[param][0],param_range[param][1]) if (not param in abundances_names) else (param_range['abundances'][0],param_range['abundances'][1]) for param in params_names])
    labels = [nice_param_name(param,config_file) for param in params_names]
    fig = corner(samples_cut, quantiles = [(1-0.6827)/2,0.5,1-(1-0.6827)/2],show_titles=True,title_kwargs={"fontsize":fontsize},verbose=True,labels=labels,bins=20,range=corner_range)
    if title is not None:
        fig.suptitle(title,fontsize=12)
    
    axes = np.array(fig.axes).reshape((n_params, n_params))
    for i in range(n_params):
        ax = axes[i, i]
        ax.axvline(np.median(samples_cut, axis=0)[i], color='g')
        # if params_names[i] in config['DATA_PARAMS'].keys():
        #     ax.axvline(config['DATA_PARAMS'][params_names[i]],color='r',label='True: {value:.2f}'.format(value=config['DATA_PARAMS'][params_names[i]]))
        #     ax.legend(fontsize=6)
    for yi in range(n_params):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(np.median(samples_cut, axis=0)[xi], color='g')
            ax.axhline(np.median(samples_cut, axis=0)[yi], color='g')
            ax.plot(np.median(samples_cut, axis=0)[xi], np.median(samples_cut, axis=0)[yi], 'sg')
            # if params_names[xi] in config['DATA_PARAMS'].keys():
            #     ax.axvline(config['DATA_PARAMS'][params_names[xi]],color='r')
            # if params_names[yi] in config['DATA_PARAMS'].keys():
            #     ax.axhline(config['DATA_PARAMS'][params_names[yi]],color='r')
    if save_plot:
        if include_abunds:
            fig.savefig(output_file+'full_cornerplot.' + plot_format,dpi=300)
        else:
            fig.savefig(output_file+'partial_cornerplot.' + plot_format,dpi=300)
    else:
        plt.show()

def plot_CO_ratio(
                config_file,
                samples,
                percent_considered = 1.,
                abundances_considered = 'all',
                output_file = '',
                fontsize = 10,
                lw = 0.5,
                figsize=(8,4),
                color = 'g',
                label='C/O$=$',
                include_quantiles = True,
                title='C/O ratio',
                ax = None,
                save_plot = True):
    
    prior_obj = Prior(config_file)
    params_names = prior_obj.params_names
    abundances_names = get_abundance_params(config_file)
    
    CO_ratio_samples = calc_CO_ratio(
        samples, 
        params_names = params_names, 
        abundances = abundances_names, 
        percent_considered = percent_considered,
        abundances_considered = abundances_considered,
        method = 'VMRs')
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    quantiles = np.quantile(CO_ratio_samples,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
    ax.hist(CO_ratio_samples,bins = 40,color=color,density = True,alpha=0.5,label=label + quantiles_to_string(quantiles))
    ax.axvline(quantiles[1],color=color,lw=lw,ls='--')
    if include_quantiles:
        ax.axvline(quantiles[0],color=color,ls='--',lw=lw)
        ax.axvline(quantiles[2],color=color,ls='--',lw=lw)
    ax.set_xlabel('C/O ratio (MMR)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if save_plot:
        fig.savefig(output_file + 'CO_ratio.png',dpi=300)
    else:
        return ax

def plot_FeH_ratio(
                config_file,
                samples,
                percent_considered = 1.,
                abundances_considered = 'all',
                output_file = '',
                fontsize = 10,
                lw = 0.5,
                figsize=(8,4),
                color = 'g',
                label='[Fe/H]$=$',
                include_quantiles = True,
                title='[Fe/H]',
                ax = None,
                save_plot=True):
    
    prior_obj = Prior(config_file)
    params_names = prior_obj.params_names
    abundances_names = get_abundance_params(config_file)
    
    FeH_ratio_samples = calc_FeH_ratio_from_samples(
        samples, 
        params_names = params_names, 
        abundances = abundances_names,
        percent_considered = percent_considered,
        abundances_considered = abundances_considered)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    quantiles = np.quantile(FeH_ratio_samples,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
    ax.hist(FeH_ratio_samples,bins = 40,color=color,density = True,alpha=0.5,label=label + quantiles_to_string(quantiles))
    ax.axvline(quantiles[1],color=color,lw=lw,ls='--')
    if include_quantiles:
        ax.axvline(quantiles[0],color=color,ls='--',lw=lw)
        ax.axvline(quantiles[2],color=color,ls='--',lw=lw)
    ax.set_xlabel('[Fe/H] (dex)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if save_plot:
        fig.savefig(output_file + 'FeH_ratio.png',dpi=300)
    else:
        return ax

def plot_retrieved_temperature_profile(
        config_file,
        samples,
        output_file,
        nb_stds = 3,
        fontsize = 10,
        lw = 0.5,
        figsize=(8,4),
        color = 'g',
        label='T$_{\mathrm{equ}}=$',
        plot_data = True,
        plot_label=True,
        title='Thermal profile',
        ax = None,
        save_plot = True):
    print('PLOTTING THERMAL PROFILE')
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    
    temp_model = config_file['retrieval']['FM']['p-T']['model']
    
    prior_obj = Prior(config_file)
    params_names = prior_obj.params_names
    
    temperatures_all = []
    for sample_i,params in enumerate(samples):
        chem_params,temp_params,clouds_params,physical_params,data_params = read_forward_model_from_config(
            config_file=config_file,
            params=params,
            params_names=params_names,
            extract_param=True)
        pressures,temperatures=get_temperatures(
            temp_model=config_file['retrieval']['FM']['p-T']['model'],
            temp_model_params=temp_params,
            physical_model_params=physical_params)
        temperatures_all += [temperatures]
    temperatures_all_arr = np.array(temperatures_all)
        
    quantiles = get_std_percentiles(nb_stds=nb_stds)
    
    quantile_curves = np.quantile(temperatures_all_arr,q=quantiles,axis = 0)
    
    # median curve
    median_label = label + quantiles_to_string(np.quantile(samples[:,params_names.index('t_equ')],q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2]),decimals = 0) + ' K'
    ax.plot(quantile_curves[nb_stds],pressures,color=color,ls=':',label=median_label)
    
    if nb_stds > 0:
        for std_i in range(1,nb_stds+1):
            lower_curve = quantile_curves[nb_stds - std_i]
            higher_curve = quantile_curves[nb_stds + std_i]
            
            ax.fill_betweenx(pressures,lower_curve,higher_curve,color = color,alpha=1/(std_i+3))
            
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim((10**temp_params['P0'],10**-6))
    ax.set_xlabel('Temperature [K]',fontsize=fontsize)
    ax.set_ylabel('Pressure [bar]',fontsize=fontsize)
    if plot_label:
        ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if save_plot:
        fig.savefig(output_file + 'temperature_plot.png',dpi=300)
    else:
        return ax
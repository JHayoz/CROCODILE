# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:40:17 2021

@author: jeanh
"""
# help functions to plot simulated spectra or the results of the retrieval

import matplotlib.pyplot as plt
from corner import corner
import numpy as np
import os
from pathlib import Path
from PyAstronomy.pyasl import crosscorrRV
from seaborn import color_palette
import random
from scipy.constants import Stefan_Boltzmann

from core.data import Data
from core.priors import Prior
from core.forward_model import ForwardModel
from core.read import read_forward_model_from_config
from core.retrievalClass import Retrieval
from core.util import quantiles_to_string,calc_CO_ratio,calc_FeH_ratio,convert_units_to_SI,scale_flux,filter_position,get_std_percentiles,get_abundance_params,nice_param_name,calc_FeH_ratio_from_samples
from core.model import get_temperatures
from core.plotting import plot_data

file_extensions = list(map(lambda x: '.' + x,['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']))

def plot_corner(
    config_file,
    samples,
    param_range = None,
    percent_considered = 0.90,
    fontsize=12,
    include_abunds = True,
    title = 'Retrieval',
    save_plot=True,
    output_file = ''
):
    # prepare save file
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file = str(output_file_path.parent / output_file_path.stem) + plot_extension
    
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
        fig.savefig(save_file,dpi=300)
    else:
        plt.show()
def plot_histogram(
    samples_param,
    figsize,
    ax,
    label,
    color,
    lw,
    include_quantiles,
):
    quantiles = np.quantile(samples_param,q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2])
    if include_quantiles:
        label_draw = label + quantiles_to_string(quantiles)
    else:
        label_draw = label
    ax.hist(samples_param,bins = 40,color=color,density = True,alpha=0.5,label=label_draw)
    ax.axvline(quantiles[1],color=color,lw=lw,ls='--')
    if include_quantiles:
        ax.axvline(quantiles[0],color=color,ls='--',lw=lw)
        ax.axvline(quantiles[2],color=color,ls='--',lw=lw)
    return ax
    
def plot_CO_ratio(
    config_file,
    samples,
    percent_considered = 1.,
    abundances_considered = 'all',
    fontsize = 10,
    lw = 0.5,
    figsize=(8,4),
    color = 'g',
    label='C/O$=$',
    include_quantiles = True,
    ax = None,
    title='C/O ratio',
    save_plot=True,
    output_file = ''
):
    
    # prepare save file
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file = str(output_file_path.parent / output_file_path.stem) + plot_extension
    
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

    ax = plot_histogram(
        samples_param = CO_ratio_samples,
        figsize = figsize,
        ax = ax,
        label = label,
        color = color,
        lw = lw,
        include_quantiles = include_quantiles
    )
    ax.set_xlabel('C/O ratio (MMR)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    
    if save_plot:
        fig.savefig(save_file,dpi=300)
    else:
        return ax

def plot_FeH_ratio(
    config_file,
    samples,
    percent_considered = 1.,
    abundances_considered = 'all',
    fontsize = 10,
    lw = 0.5,
    figsize=(8,4),
    color = 'g',
    label='[Fe/H]$=$',
    include_quantiles = True,
    ax = None,
    title='[Fe/H]',
    save_plot=True,
    output_file = ''
):
    # prepare save file
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file = str(output_file_path.parent / output_file_path.stem) + plot_extension
    
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

    ax = plot_histogram(
        samples_param = FeH_ratio_samples,
        figsize = figsize,
        ax = ax,
        label = label,
        color = color,
        lw = lw,
        include_quantiles = include_quantiles
    )
    ax.set_xlabel('[Fe/H] (dex)',fontsize=fontsize)
    ax.set_ylabel('Probability distribution',fontsize=fontsize)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.legend(fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    if save_plot:
        fig.savefig(save_file,dpi=300)
    else:
        return ax

def plot_retrieved_temperature_profile(
    config_file,
    samples,
    nb_stds = 3,
    fontsize = 10,
    lw = 0.5,
    figsize=(8,4),
    color = 'g',
    label='T$_{\mathrm{equ}}=$',
    include_quantiles = True,
    plot_data = True,
    plot_label=True,
    ax = None,
    title='Thermal profile',
    save_plot = True,
    output_file=''
):
    # prepare save file
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file = str(output_file_path.parent / output_file_path.stem) + plot_extension
    
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
    if 't_equ' in params_names:
        median_label = label + quantiles_to_string(np.quantile(samples[:,params_names.index('t_equ')],
                                                           q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2]),decimals = 0) + ' K'
    elif 't_int' in params_names:
        median_label = label + quantiles_to_string(np.quantile(samples[:,params_names.index('t_int')],
                                                           q=[(1-0.6827)/2,0.5,1-(1-0.6827)/2]),decimals = 0) + ' K'
    else:
        median_label = label
    
    if include_quantiles:
        label_draw = median_label
    else:
        label_draw = label
    
    if nb_stds == 0:
        ax.plot(quantile_curves[nb_stds],pressures,color=color,ls=':',label=label_draw)
    else:
        ax.plot(quantile_curves[nb_stds],pressures,color=color,ls=':')

    
    if nb_stds > 0:
        for std_i in range(1,nb_stds+1):
            lower_curve = quantile_curves[nb_stds - std_i]
            higher_curve = quantile_curves[nb_stds + std_i]
            if std_i == 1:
                ax.fill_betweenx(pressures,lower_curve,higher_curve,color = color,alpha=1/(std_i+3),label=label_draw)
            else:
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
        fig.savefig(save_file,dpi=300)
    else:
        return ax

def calc_retrieved_spectra(
    retrieval,
    samples,
    nb_picks=100,
    calc_contribution=False
):
    print('Picking samples')
    # pick samples
    sim_samples = pick_sampled_spectra(samples=samples,nb_picks=nb_picks)
    # get median of posterior
    median_sample = sim_samples[-1]
    model_sim_wlen = {}
    model_sim_flux = {}
    wlen_CC,flux_CC = {},{}
    wlen_RES,flux_RES = {},{}
    model_photometry = {}
    if not retrieval.forwardmodel_ck is None:
        print('Calculating c-k spectra')
        # calculate spectra using samples
        wlen_sample_ck,flux_sample_ck,wlen_sample,flux_sample,em_contr_fct = calc_sampled_spectra_ck(
            retrieval,retrieval.forwardmodel_ck,sim_samples=sim_samples,calc_contribution=calc_contribution)
        # get posterior SED
        wlen_arr = wlen_sample[0]
        
        nb_random_samples = np.max([nb_picks,len(sim_samples)])
        flux_arr = np.zeros((nb_random_samples,len(flux_sample[0])))
        for sample_i in range(nb_random_samples):
            flux_arr[sample_i] = flux_sample[sample_i]
        quantiles = get_std_percentiles(nb_stds=1)
        flux_quantiles = np.quantile(flux_arr,q=quantiles,axis = 0)
        
        # get median SED
        median_wlen_ck,median_flux_ck = wlen_sample[len(sim_samples)-1],flux_sample[len(sim_samples)-1]
        flux_median = flux_sample[len(sim_samples)-1]
        
        # save as dictionary
        model_sim_wlen['median'] = wlen_arr
        model_sim_flux['median'] = flux_median
        for quant_i,quant in enumerate(quantiles):
            model_sim_wlen['quantile_%.2f' % quant] = wlen_arr
            model_sim_flux['quantile_%.2f' % quant] = flux_quantiles[quant_i]
        
        # get model for median posterior
        chem_params,temp_params,clouds_params,physical_params,data_params = read_forward_model_from_config(
                config_file=retrieval.config,
                params=median_sample,
                params_names=retrieval.params_names,
                extract_param=True)
        # get model low-resolution spectroscopy and photometry
        model_photometry,wlen_RES,flux_RES = calc_model_spectroscopy_ck(retrieval,median_wlen_ck,median_flux_ck,data_params)
    if not retrieval.forwardmodel_lbl is None:
        print('Calculating lbl spectrum')
        # get model high-resolution spectroscopy
        wlen_CC,flux_CC,wlen_RES,flux_RES = calc_sampled_spectra_lbl(retrieval,median_sample,wlen_RES,flux_RES)
    emission_contribution = None
    if calc_contribution:
        emission_contribution = em_contr_fct[len(sim_samples)-1]
    return model_sim_wlen,model_sim_flux,wlen_CC,flux_CC,wlen_RES,flux_RES,model_photometry,flux_quantiles,wlen_arr,flux_median,emission_contribution

def plot_retrieved_spectra(
    config_file,
    retrieval,
    samples,
    nb_picks=100,
    fontsize = 10,
    title='Retrieved spectrum',
    save_plot = True,
    output_file='',
    ax=None
):
    # prepare save file
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file_SED = str(output_file_path.parent / output_file_path.stem) + '_SED' + plot_extension
    save_file_contrem = str(output_file_path.parent / output_file_path.stem) + '_contrem' + plot_extension
    save_file_CCF = str(output_file_path.parent / output_file_path.stem) + '_CCF' + plot_extension

    wvl_label = 'Wavelength [$\mu$m]'
    filter_label = 'Filter trsm.'
    flux_label = 'Flux [Wm$^{-2}\mu$m$^{-1}$]'
    CC_flux_label = 'Residuals [Wm$^{-2}\mu$m$^{-1}$]'

    model_sim_wlen,model_sim_flux,wlen_CC,flux_CC,wlen_RES,flux_RES,model_photometry,flux_quantiles,wlen_arr,flux_median,emission_contribution = calc_retrieved_spectra(
        retrieval,
        samples,
        nb_picks=nb_picks,
    )
    
    # get all data
    PHOT_flux,PHOT_flux_err,filt,filt_func,filt_mid,filt_width = retrieval.data_obj.getPhot()
    RES_data_wlen,RES_data_flux,flux_err,inverse_cov,flux_data_std = retrieval.data_obj.getRESSpectrum()
    CC_data_wlen,CC_data_flux = retrieval.data_obj.getCCSpectrum()
    data_N,data_sf2 = retrieval.data_obj.CC_data_N,retrieval.data_obj.CC_data_sf2
    extinction = retrieval.data_obj.extinction
    
    plot_data(retrieval.config,
              CC_wlen = CC_data_wlen,
              CC_flux = CC_data_flux,
              CC_wlen_w_cont = {},
              CC_flux_w_cont = {},
              model_CC_wlen = wlen_CC,
              model_CC_flux = flux_CC,
              sgfilter = {},
              RES_wlen = RES_data_wlen,
              RES_flux = RES_data_flux,
              RES_flux_err = flux_err,
              model_RES_wlen = wlen_RES,
              model_RES_flux = flux_RES,
              PHOT_midpoint = filt_mid,
              PHOT_width = filt_width,
              PHOT_flux = PHOT_flux,
              PHOT_flux_err = PHOT_flux_err,
              PHOT_filter = filt,
              PHOT_sim_wlen = model_sim_wlen,
              PHOT_sim_flux = model_sim_flux,
              model_PHOT_flux = model_photometry,
              inset_plot = True,
              output_file = str(Path(save_file_SED).parent),
              plot_name=str(Path(save_file_SED).stem),
              title = 'Spectrum',
              fontsize=15,
              plot_errorbars=True,
              save_plot=True,
              extinction = extinction
             )
    # plot all data
    
    """
    print('Plotting')
    # start plot
    if len(PHOT_flux.keys()) > 0:
        filter_pos = filter_position(filt_mid)
        rgba = {}
        cmap = color_palette('colorblind',n_colors = len(PHOT_flux.keys()),as_cmap = True)
        
        phot_keys = list(PHOT_flux.keys())
        phot_instr = list(map(lambda x: x[x.index('_')+1:],phot_keys))
        instr_to_phot_flux_to = {phot_instr[i]:phot_keys[i] for i in range(len(phot_instr))}
        for instr in PHOT_flux.keys():
            rgba[instr] = cmap[filter_pos[instr]%len(cmap)]
    if len(RES_data_wlen.keys()) + len(PHOT_flux.keys()) > 0:
    # plot for cont-included data
    fig=plt.figure(figsize=(10,8))
    nb_plots = 4
    ax = plt.subplot(nb_plots,1,1)
    
    # ax = fig.axes[0]
    min_x,max_x = np.min(wlen_arr),np.max(wlen_arr)
    if max_x - min_x > 10:
        xscale='log'
    else:
        xscale='linear'
    
    ax.set_xlim((min_x,max_x))
    for instr in PHOT_flux.keys():
        ax.plot(filt[instr][0],filt[instr][1],color=rgba[instr])
    
    for key in np.unique(phot_instr):
        instr = instr_to_phot_flux_to[key]
        label = key[key.index('/')+1:]
        ax.annotate(label,(filt_mid[instr],0),rotation=60,color='k')
    if title is not None:
        ax.set_title(title,fontsize=fontsize)
    ax.set_xlabel(wvl_label,fontsize=fontsize)
    ax.set_ylabel(filter_label,fontsize=fontsize)
    ax.set_xscale(xscale)
    
    ax = plt.subplot(nb_plots,1,(2,3))
    ax.fill_between(x=wlen_arr,y1=flux_quantiles[0],y2=flux_quantiles[2],alpha=0.5,color='grey')
    ax.plot(wlen_arr,flux_median,'k',lw=0.5)
    ax.set_xlim((np.min(wlen_arr),np.max(wlen_arr)))
    
    for instr in RES_data_wlen.keys():
        ax.errorbar(x=RES_data_wlen[instr],y=RES_data_flux[instr],yerr=flux_data_std[instr],fmt='|',label=instr)
        ax.scatter(x=wlen_RES[instr],y=flux_RES[instr],s=1,color='r')
    
    for instr in PHOT_flux.keys():
        ax.errorbar(x=filt_mid[instr],y=PHOT_flux[instr],xerr=filt_width[instr]/2,
                    yerr=PHOT_flux_err[instr],color=rgba[instr],zorder=10,marker='o',markersize=3)
        ax.errorbar(x=filt_mid[instr],y=model_photometry[instr],
                    xerr=filt_width[instr]/2,color='r',zorder=10,marker='o',markersize=3)
    ax.legend()
    ax.set_xlabel(wvl_label,fontsize=fontsize)
    ax.set_ylabel(flux_label,fontsize=fontsize)
    ax.set_xscale(xscale)
    
    ax = plt.subplot(nb_plots,1,nb_plots)
    
    for instr in RES_data_wlen.keys():
        residual = (flux_RES[instr]-RES_data_flux[instr])/flux_data_std[instr]
        ax.scatter(x=wlen_RES[instr],y=residual,s=1,color='r',label=instr)
    for instr in PHOT_flux.keys():
        residual = (model_photometry[instr]-PHOT_flux[instr])/PHOT_flux_err[instr]
        ax.scatter(x=filt_mid[instr],y=residual,s=1,color='r')
    for std in np.arange(-3,4,3):
        ax.axhline(std,color='k',ls=':',alpha=0.5)
    ax.set_xlim((np.min(wlen_arr),np.max(wlen_arr)))
    plt.xlabel(wvl_label,fontsize=fontsize)
    plt.ylabel(r'Residual [$\sigma$]')
    ax.set_xscale(xscale)
    plt.tight_layout()
    plt.legend()
    
    
    if save_plot:
        plt.savefig(save_file_SED,dpi=300)
    plt.show()
    """
    
    # plot for cont-removed data
    if len(CC_data_wlen.keys()) > 0:
        fig=plt.figure(figsize=(15,5))
        for instr in CC_data_wlen.keys():
            plt.plot(CC_data_wlen[instr],CC_data_flux[instr]/np.std(CC_data_flux[instr]),label=instr)
            plt.plot(wlen_CC[instr],flux_CC[instr]/np.std(flux_CC[instr]),'r')
        plt.legend()
        # lim_low = np.min([CC_data_wlen[instr][0] for instr in CC_data_wlen.keys()])
        # lim_high = np.max([CC_data_wlen[instr][-1] for instr in CC_data_wlen.keys()])
        # plt.xlim((lim_low,lim_high))
        plt.xlabel(wvl_label,fontsize=fontsize)
        plt.ylabel('Flux (a.u.)',fontsize=fontsize)
        if title is not None:
            plt.title(title,fontsize=fontsize)
        if save_plot:
            plt.savefig(save_file_contrem,dpi=300)
        plt.show()
        
        # CCF
        rv_range,drv_step=500,0.5
        plt.figure()
        for instr in CC_data_wlen.keys():
            # extend range of model
            wvl_stepsize=np.mean(wlen_CC[instr][1:]-wlen_CC[instr][:-1])
            extend_wvl_bins=int(0.1/wvl_stepsize)
            wlen_low = np.arange(wlen_CC[instr][0]-extend_wvl_bins*wvl_stepsize,wlen_CC[instr][0]-wvl_stepsize,wvl_stepsize)
            wlen_high = np.arange(wlen_CC[instr][-1]+wvl_stepsize,wlen_CC[instr][-1]+(extend_wvl_bins+1)*wvl_stepsize,wvl_stepsize)
            new_wlen_CC = np.hstack([wlen_low,wlen_CC[instr],wlen_high]).flatten()
            new_flux_CC = np.zeros_like(new_wlen_CC)
            mask_CC = np.logical_and(new_wlen_CC >= wlen_CC[instr][0],new_wlen_CC <= wlen_CC[instr][-1])
            new_flux_CC[mask_CC] = flux_CC[instr]
            
            drv,ccf = crosscorrRV(CC_data_wlen[instr],CC_data_flux[instr],new_wlen_CC,new_flux_CC,rvmin=-rv_range,rvmax=rv_range,drv=drv_step)
            sf2 = np.sum(flux_CC[instr]**2)
            # plt.plot(drv,2*ccf-data_sf2[instr]-sf2,label=instr)
            plt.plot(drv,ccf/np.std(ccf),label=instr)
        plt.legend()
        plt.xlabel(wvl_label,fontsize=fontsize)
        plt.ylabel('CCF (a.u.)',fontsize=fontsize)
        if title is not None:
            plt.title(title,fontsize=fontsize)
        if save_plot:
            plt.savefig(save_file_CCF,dpi=300)
        plt.show()
    
    # one plot per RES dataset to see how good the fit is more closely

    if len(CC_data_wlen.keys()) > 0:
        
        for instr in CC_data_wlen.keys():
            fig=plt.figure(figsize=(15,5))
            plt.plot(CC_data_wlen[instr],CC_data_flux[instr]/np.std(CC_data_flux[instr]),label=instr)
            plt.plot(wlen_CC[instr],flux_CC[instr]/np.std(flux_CC[instr]),'r')
            plt.legend()
            # lim_low = np.min([CC_data_wlen[instr][0] for instr in CC_data_wlen.keys()])
            # lim_high = np.max([CC_data_wlen[instr][-1] for instr in CC_data_wlen.keys()])
            # plt.xlim((lim_low,lim_high))
            plt.xlabel(wvl_label,fontsize=fontsize)
            plt.ylabel('Flux (a.u.)',fontsize=fontsize)
            if title is not None:
                plt.title(title + instr,fontsize=fontsize)
            if save_plot:
                save_file_path = Path(save_file_contrem)
                save_file_path_new = str(save_file_path.parent / save_file_path.stem) + ('_%s' % instr) + str(save_file_path.suffix)
                plt.savefig(save_file_path_new,dpi=300)
            plt.show()
    if len(PHOT_flux.keys()) > 0:
        filter_pos = filter_position(filt_mid)
        rgba = {}
        cmap = color_palette('colorblind',n_colors = len(PHOT_flux.keys()),as_cmap = True)
        
        phot_keys = list(PHOT_flux.keys())
        phot_instr = list(map(lambda x: x[x.index('_')+1:],phot_keys))
        instr_to_phot_flux_to = {phot_instr[i]:phot_keys[i] for i in range(len(phot_instr))}
        for instr in PHOT_flux.keys():
            rgba[instr] = cmap[filter_pos[instr]%len(cmap)]
    
    for res_instr in RES_data_wlen.keys():
        fig=plt.figure(figsize=(15,5))
        ax = plt.gca()
        # data
        ax.errorbar(x=RES_data_wlen[res_instr],y=RES_data_flux[res_instr],yerr=flux_data_std[res_instr],fmt='|',label=res_instr)
        ax.scatter(x=wlen_RES[res_instr],y=flux_RES[res_instr],s=1,color='r')
        
        # photometry
        for phot_instr in PHOT_flux.keys():
            ax.errorbar(x=filt_mid[phot_instr],y=PHOT_flux[phot_instr],xerr=filt_width[phot_instr]/2,
                        yerr=PHOT_flux_err[phot_instr],color=rgba[phot_instr],zorder=10,marker='o',markersize=3)
            ax.errorbar(x=filt_mid[phot_instr],y=model_photometry[phot_instr],
                        xerr=filt_width[phot_instr]/2,color='r',zorder=10,marker='o',markersize=3)
        # model
        ax.fill_between(x=wlen_arr,y1=flux_quantiles[0],y2=flux_quantiles[2],alpha=0.5,color='grey')
        ax.plot(wlen_arr,flux_median,'k',lw=0.5)
        
        ax.legend()
        ax.set_xlabel(wvl_label,fontsize=fontsize)
        ax.set_ylabel(flux_label,fontsize=fontsize)
        ax.set_xlim((np.min(RES_data_wlen[res_instr]),np.max(RES_data_wlen[res_instr])))
        
        if save_plot:
            save_file_path = Path(save_file_SED)
            save_file_path_new = str(save_file_path.parent / save_file_path.stem) + ('_%s' % res_instr) + str(save_file_path.suffix)
            plt.savefig(save_file_path_new,dpi=300)
        plt.show()
        
    
def pick_sampled_spectra(
    samples,
    nb_picks=100
):
    sample_indices = list(np.arange(len(samples)))
    median_sample = np.median(samples,axis=0)
    
    if nb_picks > 0:
        rand_indices = random.sample(sample_indices,k=nb_picks)
        rand_samples = samples[rand_indices]
        sim_samples = np.vstack([rand_samples,median_sample])
    else:
        sim_samples = np.array([median_sample])

    return sim_samples

def calc_sampled_spectra_ck(
    retrieval,
    forwardmodel,
    sim_samples,
    calc_contribution=False
):
    
    wlen_sample_ck,flux_sample_ck = {},{}
    wlen_sample,flux_sample = {},{}
    em_contr_fct = {}
    for sample_i in range(len(sim_samples)):
        print('Sample progress: %.2f' % ((sample_i+1)*100/len(sim_samples)),end='\r')
        
        params = sim_samples[sample_i]
        
        chem_params,temp_params,clouds_params,physical_params,data_params = read_forward_model_from_config(
            config_file=retrieval.config,
            params=params.copy(),
            params_names=retrieval.params_names,
            extract_param=True)
        # evaluate log-likelihood for FM using c-k mode
        
        return_args = forwardmodel.calc_spectrum(
                  chem_model_params = chem_params,
                  temp_model_params = temp_params,
                  cloud_model_params = clouds_params,
                  physical_params = physical_params,
                  external_pt_profile = None,
                  return_profiles = False,
                  calc_contribution = calc_contribution
        )
        if calc_contribution:
            wlen_sample_ck[sample_i], flux_sample_ck[sample_i], em_contr_fct[sample_i] = return_args
        else:
            wlen_sample_ck[sample_i], flux_sample_ck[sample_i] = return_args
        wlen_ck_SI,flux_ck_SI = convert_units_to_SI(wlen_sample_ck[sample_i], flux_sample_ck[sample_i])
        flux_ck_SI_scaled = scale_flux(flux_ck_SI,radius=physical_params['R'],distance=physical_params['distance'])
        
        wlen_sample[sample_i], flux_sample[sample_i] = wlen_ck_SI,flux_ck_SI_scaled
    print()
    return wlen_sample_ck,flux_sample_ck,wlen_sample,flux_sample,em_contr_fct

def calc_model_spectroscopy_ck(
    retrieval,
    wlen_ck,
    flux_ck,
    data_params
):
    wlen_RES,flux_RES = {},{}
    model_photometry = {}
    if retrieval.data_obj.PHOTinDATA():
        # print('c-k photometry Log-L')
        log_L_PHOT,model_photometry = retrieval.calc_log_L_PHOT(
            wlen=wlen_ck,
            flux=flux_ck)
    
    RES_data_wlen,RES_data_flux,flux_err,inverse_cov,flux_data_std = retrieval.data_obj.getRESSpectrum()
    if retrieval.data_obj.RES_data_with_ck:
        for instr in RES_data_wlen.keys():
            if retrieval.data_obj.RES_data_info[instr][0] == 'c-k':
                # print('c-k RES-Log-L for',instr)
                log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = retrieval.calc_log_L_RES(
                    wlen=wlen_ck,
                    flux=flux_ck,
                    data_params=data_params,
                    data_key=instr,
                    wlen_data=RES_data_wlen[instr],
                    flux_data=RES_data_flux[instr],
                    flux_cov=flux_err[instr],
                    inverse_cov=inverse_cov[instr],
                    flux_data_std=flux_data_std[instr],
                    mode='c-k')
    return model_photometry,wlen_RES,flux_RES


def calc_sampled_spectra_lbl(
    retrieval,
    params,
    wlen_RES,
    flux_RES
):
    
    CC_data_wlen,CC_data_flux = retrieval.data_obj.getCCSpectrum()
    data_N,data_sf2 = retrieval.data_obj.CC_data_N,retrieval.data_obj.CC_data_sf2
    RES_data_wlen,RES_data_flux,flux_err,inverse_cov,flux_data_std = retrieval.data_obj.getRESSpectrum()
    
    chem_params,temp_params,clouds_params,physical_params,data_params = read_forward_model_from_config(
            config_file=retrieval.config,
            params=params,
            params_names=retrieval.params_names,
            extract_param=True)
    
    # evaluate log-likelihood for FM using lbl mode
    wlen_CC,flux_CC,sgfilter={},{},{}
    if retrieval.forwardmodel_lbl is not None:
        
        for interval_key in retrieval.lbl_itvls.keys():
            wlen_lbl,flux_lbl = retrieval.forwardmodel_lbl[interval_key].calc_spectrum(
                  chem_model_params = chem_params,
                  temp_model_params = temp_params,
                  cloud_model_params = clouds_params,
                  physical_params = physical_params,
                  external_pt_profile = None,
                  return_profiles = False)
            
            wlen_lbl_SI,flux_lbl_SI = convert_units_to_SI(wlen_lbl,flux_lbl)
            flux_lbl_SI_scaled = scale_flux(flux_lbl_SI,radius=physical_params['R'],distance=physical_params['distance'])
            
            if retrieval.data_obj.CCinDATA():
                for instr in CC_data_wlen.keys():
                    if retrieval.CC_to_lbl_itvls[instr] == interval_key:
                        # print('lbl CC-Log-L for',instr)
                        log_L_CC_temp,wlen_CC[instr],flux_CC[instr],sgfilter[instr] = retrieval.calc_log_L_CC(
                            wlen=wlen_lbl_SI,
                            flux=flux_lbl_SI_scaled,
                            data_wlen=CC_data_wlen[instr],
                            data_flux=CC_data_flux[instr],
                            data_N=data_N[instr],
                            data_sf2=data_sf2[instr],
                            data_params=data_params,
                            data_key=instr
                            )
            
            if retrieval.data_obj.RESinDATA():
                for instr in RES_data_wlen.keys():
                    if retrieval.data_obj.RES_data_info[instr][0] == 'lbl':
                        if retrieval.RES_to_lbl_itvls[instr] == interval_key:
                            # print('lbl RES-Log-L for',instr)
                            log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = retrieval.calc_log_L_RES(
                                wlen=wlen_lbl_SI,
                                flux=flux_lbl_SI_scaled,
                                data_params=data_params,
                                data_key=instr,
                                wlen_data=RES_data_wlen[instr],
                                flux_data=RES_data_flux[instr],
                                flux_cov=flux_err[instr],
                                inverse_cov=inverse_cov[instr],
                                flux_data_std=flux_data_std[instr],
                                mode='lbl')
    return wlen_CC,flux_CC,wlen_RES,flux_RES

def plot_retrieved_teff(
    config_file,
    retrieval,
    samples,
    nb_picks=5,
    wlen_borders = [0.2,15],
    title='Retrieved Teff',
    save_plot = True,
    output_file=''
):
    output_file_path = Path(output_file)
    plot_extension = output_file_path.suffix
    if plot_extension not in file_extensions:
        plot_extension = '.png'
    save_file = str(output_file_path.parent / output_file_path.stem ) + plot_extension
    
    # first need to create a new forward model that extends over a large range of wavelength
    if retrieval.config['retrieval']['FM']['clouds']['opacities'] is None:
        cloud_species = None
    else:
        cloud_species = retrieval.config['retrieval']['FM']['clouds']['opacities'].copy()
    forwardmodel_teff = ForwardModel(
         wlen_borders = wlen_borders,
         max_wlen_stepsize = 0,
         mode = 'c-k',
         line_opacities = retrieval.abundances_ck,
         cont_opacities= ['H2-H2','H2-He'],
         rayleigh_scat = ['H2','He'],
         cloud_model = retrieval.cloud_model,
         cloud_species = cloud_species,
         do_scat_emis = retrieval.config['retrieval']['FM']['clouds']['scattering_emission'],
         chem_model = retrieval.chem_model,
         temp_model = retrieval.temp_model,
         max_RV = 0,
         max_winlen = 0,
         include_H2 = True,
         only_include = 'all'
         )
    forwardmodel_teff.calc_rt_obj(lbl_sampling = 5)
    # pick samples
    print('Picking samples')
    sim_samples = pick_sampled_spectra(samples=samples,nb_picks=nb_picks)

    # create spectra from samples
    print('Calculating c-k spectra')
    wlen_sample_ck,flux_sample_ck,wlen_sample,flux_sample,em_contr_fct = calc_sampled_spectra_ck(
        retrieval,forwardmodel_teff,sim_samples=sim_samples,calc_contribution=False)
    
    # compute Teff from Stefan-Boltzmann law
    teff_samples = np.zeros((nb_picks))
    for spectrum_i in range(nb_picks):
        wlen_sample[spectrum_i]
        flux_sample[spectrum_i]
        energy = np.trapz(y=flux_sample[spectrum_i],x=wlen_sample[spectrum_i])
        teff = (energy/Stefan_Boltzmann)**0.25
        teff_samples[spectrum_i] = teff
    
    # plot histogram
    quantiles = get_std_percentiles(nb_stds=1)
    teff_quantiles = np.quantile(teff_samples,q=quantiles,axis = 0)
    plt.figure()
    ax = plt.gca()
    plt.hist(teff_samples)
    plt.axvline(teff_quantiles[0],color='k')
    for std_i in range(2):
        plt.axvline(teff_quantiles[2*std_i],ls='--',color='k')
    plt.title(r'$T_{eff}$ = ' + quantiles_to_string(teff_quantiles,decimals=1) + ' K')
    plt.xlabel(r'$T_{eff}$ [K]')
    if save_plot:
        plt.savefig(save_file,dpi=300)
    else:
        return ax
    
    
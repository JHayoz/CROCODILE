# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:45:22 2021

@author: jeanh
"""

import sys
import os
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from os import path
from petitRADTRANS import radtrans as rt
from petitRADTRANS import nat_cst as nc
import numpy as np
#from numpy.linalg import inv
#from scipy.interpolate import interp1d
#from time import time
#from PyAstronomy.pyasl import crosscorrRV
#import matplotlib.pyplot as plt
from typing import Optional,Union
from core.util import poor_mans_abunds_ck,poor_mans_abunds_lbl,convert_to_ck_names
#from core.rebin import *
from core.model2 import get_temperatures,evaluate_forward_model

class ForwardModel:
    """
    ForwardModel can be used to compute spectra for a given choice of model and model parameters.
    """
    def __init__(
            self,
            wlen_borders:list=[1,10],
            max_wlen_stepsize:float=0.1,
            mode:str='c-k',
            line_opacities:list=['H2O_main_iso'],
            cont_opacities:list = ['H2-H2','H2-He'],
            rayleigh_scat:list = ['H2','He'],
            cloud_model:str = None,
            cloud_species:list = [],
            do_scat_emis:bool = False,
            chem_model:str = 'free', # can be 'free' or 'chem_equ'
            temp_model:str = 'guillot',
            max_RV:float = 1000.,
            max_winlen:float = 201,
            include_H2:bool = True,
            only_include:Union[str,list] = 'all'
        ):
        """
        Constructor of the ForwardModel class which computes spectra for a given choice of model and model parameters

        Parameters
        ----------
        wlen_borders : str
            The wavelength range of the spectrum that needs to be computed.
        max_wlen_stepsize: float
            Maximum stepsize of the bins.
        mode:str
            Mode used to compute the spectrum using petitRADTRANS: 'c-k' for low spectral resolution or 'lbl' for high spectral resolution.
        line_opacities:list
            List of opacities included to compute the spectrum. Their names need to match that of the database.
        cont_opacities:list
            List of continuum opacities.
        rayleigh_scat:list,
            List of Rayleigh scatterers.
        clouds:str
            Name of the model used to compute clouds.
        cloud_species:list,
            List of species that make up the clouds.
        do_scat_emis:bool
            Whether to include the treatment of scattering for the clouds.
        chem_model
            Name of the chemical model used.
        temp_model:str
            Name of the p-T model used.
        max_RV:float
            Maximum radial velocity that will be used to doppler shift the spectrum.
        max_winlen:float
            Maximum window size of the filter that will be used to remove the continuum.
        include_H2:bool
            Whether to include H2 as an opacity.
        only_include:Union[str,list]
            Only relevant for chemical equilibrium model: whether to consider all opacities ('all') or list of opacities to consider.

        Returns
        -------
        NoneType
            None
        """
        
        self.wlen_borders = wlen_borders
        self.max_wlen_stepsize = max_wlen_stepsize
        self.mode = mode
        self.line_opacities = line_opacities.copy()
        self.only_include = only_include
        if include_H2:
            self.line_opacities += ['H2_main_iso']
        if cloud_model == 'grey_deck' or cloud_model is None:
            self.cloud_species =  []
            self.do_scat_emis = False
        else:
            self.cloud_species = cloud_species
            self.do_scat_emis = do_scat_emis
        self.continuum_opacities =  cont_opacities
        self.rayleigh_scatterers = rayleigh_scat
        self.cloud_model = cloud_model
        self.chem_model = chem_model
        self.temp_model = temp_model
        self.max_RV = max_RV
        self.max_winlen = max_winlen
        
        if self.chem_model == 'chem_equ':
            if self.mode == 'c-k':
                self.line_opacities = poor_mans_abunds_ck()
            else:
                self.line_opacities = poor_mans_abunds_lbl()
            if self.only_include != 'all':
                self.line_opacities = self.only_include
                if include_H2:
                    self.line_opacities += ['H2_main_iso']
        else:
            # 'free' model
            if self.mode == 'c-k':
                self.line_opacities = convert_to_ck_names(self.line_opacities)
        
        self.opa_struct_set = False
        self.rt_obj = None
        
        print('Forward model setup with {model} model and {mode} mode'.format(model=self.chem_model,mode=self.mode))
        if self.cloud_model is not None:
            print('USING CLOUDS WITH MODEL {model} AND CLOUD SPECIES {species}'.format(model = self.cloud_model,species=cloud_species))
        
        return
    
    def extend(
            self
        ):
        """
        Extends the self.wlen_borders such that the final spectrum still includes the given wavelength borders after doppler shift, rebinning, and removing its continuum
        """
        extensions = 0.
        
        # Doppler shift
        extensions += max(abs((self.wlen_borders[0])*(self.max_RV)/nc.c),
                             abs((self.wlen_borders[1])*(self.max_RV)/nc.c))*1e5
        
        # rebin
        extensions += 10*self.max_wlen_stepsize
        
        # Continuum-removed
        extensions += 2*self.max_wlen_stepsize*(self.max_winlen+3)
        
        # adjust wvl range
        self.wlen_borders[0] -= extensions
        self.wlen_borders[1] += extensions
    
    def calc_rt_obj(
            self,
            lbl_sampling:int = None
        ):
        """
        Initialises the Radtrans class from petitRADTRANS with the given parameters
        
        Parameters
        ----------
        lbl_sampling : int
            lbl_sampling-1 is the number of lines to skip at high spectral resolution ('lbl' mode) for faster computation.
        
        Returns
        -------
        NoneType
            None
        """
        # adjust wvl range
        self.extend()
        
        if self.mode == 'c-k':
            lbl_sampling = None
        print('wlen borders: ',self.wlen_borders)
        print('Line species included: ',self.line_opacities)
        line_opacities_to_use = self.line_opacities
        if 'He' in line_opacities_to_use:
            line_opacities_to_use.remove('He')
        if 'H2' in line_opacities_to_use and self.mode == 'lbl':
            line_opacities_to_use.remove('H2')
            line_opacities_to_use.append('H2_main_iso')
        self.rt_obj = rt.Radtrans(line_species = line_opacities_to_use,
                              rayleigh_species = self.rayleigh_scatterers,
                              continuum_opacities = self.continuum_opacities,
                              mode = self.mode,
                              wlen_bords_micron = self.wlen_borders,
                              cloud_species = self.cloud_species,
                              do_scat_emis = self.do_scat_emis,
                              lbl_opacity_sampling = lbl_sampling)
        
    def calc_spectrum(self,
                      chem_model_params:dict = {},
                      temp_model_params:dict = {},
                      cloud_model_params:dict = {},
                      physical_params:dict = {},
                      external_pt_profile:Union[np.array,list] = None,
                      return_profiles:bool = False
                      ):
        """
        Calculates the spectrum from the given parameter of chemical, thermal, or cloud model.
        
        Parameters
        ----------
        chem_model_params : dict,
            Dictionary containing the values of the parameters for the chemical model chosen when initialising the class.
        temp_model_params: dict,
            Dictionary containing the values of the parameters for the thermal model chosen when initialising the class.
        cloud_model_params:dict,
            Dictionary containing the values of the parameters for the cloud model chosen when initialising the class.
        physical_params:dict,
            Dictionary containing the values of the physical parameters.
        external_pt_profile:np.array or list,
            p-T profile (pressures,temperatures) calculated externally
        return_profiles:bool,
            Whether to return pressures,temperatures,abundances or not in addition to wlen and flux.
        

        Returns
        -------
        np.array,np.array
            Wavelength and flux calculated
        np.array,np.array,np.array,np.array,dict
            Wavelength and flux calculated, pressures, temperatures, and abundance dictionary if return_profiles is True
        """
        if self.only_include != 'all':
            self.only_include = list(dict.fromkeys(self.only_include))
        
        # temperature-pressure profile
        if external_pt_profile is not None:
            pressures,temperatures = external_pt_profile
        else:
            print(self.temp_model)
            print(temp_model_params)
            pressures,temperatures=get_temperatures(temp_model=self.temp_model,temp_model_params=temp_model_params)
        
        # setup up opacity structure if not yet done so
        if not(self.opa_struct_set):
            self.rt_obj.setup_opa_structure(pressures)
            self.opa_struct_set = True
        
        
        wlen,flux,pressures,temperatures,abundances = evaluate_forward_model(
            rt_object=self.rt_obj,
            chem_model=self.chem_model,
            chem_model_params=chem_model_params,
            temp_model=self.temp_model,
            temp_model_params=temp_model_params,
            cloud_model=self.cloud_model,
            cloud_model_params=cloud_model_params,
            physical_params=physical_params,
            mode=self.mode,
            only_include=self.only_include)
        
        if self.only_include != 'all':
            self.only_include = list(dict.fromkeys(self.only_include))
            assert(len(abundances.keys())==len(self.only_include)+2)
        
        if return_profiles:
            return wlen,flux,pressures,temperatures,abundances
        else:
            return wlen,flux
"""
    def calc_emission_contribution(self):
        
        return self.rt_obj.contr_em
    
    def calc_pressure_distribution(
        self,
        config,
        contr_em_fct,
        ab_metals,
        temp_params,
        wlen,
        flux,
        which_data_format = 'CC',
        which_em = 'molecules', # 'retrieved' or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
        which_abund = 'retr_abund', # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
        which_included = 'included', # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
        output_dir = '',
        plot_distr = True
        ):
        
        if which_em == 'retrieved':
            contribution = False
        else:
            contribution = True
        
        #wlen_lbl_ref = nc.c/self.rt_obj.freq*1e4
        
        wlen_mol,flux_mol,flux_diff,flux_diff_interped,pressure_distr = {},{},{},{},{}
        abundances_considered = ab_metals.keys()
        
        if plot_distr:
            figs,axs = plt.subplots(nrows=len(abundances_considered),ncols=2,figsize=(2*5,len(abundances_considered)*5))
        
        for mol_i,mol in enumerate(abundances_considered):
            print('Considering '+mol)
            mol_ab_metals = {}
            for key in ab_metals.keys():
                # copy retrieved abundances
                if which_included == 'excluded':
                    if which_abund == 'retr_abund':
                        mol_ab_metals[key] = ab_metals[key]
                    else:
                        # 'high_abund'
                        mol_ab_metals[key] = -3.5
                else:
                    # 'included'
                    mol_ab_metals[key] = -20
            
            # remove one molecule
            if which_included == 'excluded':
                mol_ab_metals[mol] = -20
            else:
                # 'included'
                if which_abund == 'retr_abund':
                    mol_ab_metals[mol] = ab_metals[mol]
                else:
                    # 'high_abund'
                    mol_ab_metals[mol] = -3.5
            print(mol_ab_metals)
            print(contribution)
            wlen_lbl_mol,flux_lbl_mol = self.calc_spectrum(
                    ab_metals = mol_ab_metals,
                    temp_params = temp_params,
                    clouds_params = {},
                    external_pt_profile = None,
                    return_profiles = False,
                    contribution = contribution)
            if which_em == 'molecules':
                contr_em_fct = self.calc_emission_contribution()
                print(np.shape(contr_em_fct))
            if which_data_format == 'CC':
                wlen_mol[mol],flux_mol[mol],calc_filter,wlen_rebin_datalike,flux_rebin_datalike = rebin_to_CC(wlen_lbl_mol,flux_lbl_mol,wlen,win_len=config['WIN_LEN'],method='datalike',filter_method = 'only_gaussian',nb_sigma=5,convert = True,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            else:
                # 'RES'
                # need to take the spectrum difference between retrieved and spectrum where we exclude each molecule, but we still need to take the emission contribution function of the molecules
                mol_ab_metals = {key:ab_metals[key] for key in ab_metals.keys()}
                mol_ab_metals[mol] = -20
                wlen_lbl_mol_RES,flux_lbl_mol_RES = self.calc_spectrum(
                    ab_metals = mol_ab_metals,
                    temp_params = temp_params,
                    clouds_params = {},
                    external_pt_profile = None,
                    return_profiles = False,
                    contribution = False)
                
                wlen_mol[mol],flux_mol[mol] = rebin_to_RES(wlen_lbl_mol_RES,flux_lbl_mol_RES,wlen,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            wlen_lbl_mol = 1e4*wlen_lbl_mol
            if which_included == 'excluded' or which_data_format == 'RES':
                flux_diff[mol] = np.abs(flux_mol[mol]-flux)
            else:
                # 'included'
                flux_diff[mol] = np.abs(flux_mol[mol])
            
            flux_diff_interp = interp1d(wlen_mol[mol],flux_diff[mol],bounds_error=False,fill_value=0)
            flux_diff_interped[mol] = flux_diff_interp(wlen_lbl_mol)
            
            pressure_distr[mol] = np.dot(contr_em_fct,flux_diff_interped[mol])/sum(flux_diff_interped[mol])
            
            if plot_distr:
                print(np.shape(contr_em_fct))
                pressures = np.logspace(-6,temp_params['P0'],100)
                X,Y = np.meshgrid(wlen_lbl_mol[::100], pressures)
                axs[mol_i,0].contourf(X, Y,contr_em_fct[:,::100],cmap=plt.cm.bone_r)
                axs[mol_i,0].set_xlim([np.min(wlen_lbl_mol),np.max(wlen_lbl_mol)])
                axs[mol_i,0].set_title('Contribution emission function',fontsize=12)
                axs[mol_i,0].set_yscale('log')
                axs[mol_i,0].set_ylim([1e2,1e-6])
                axs[mol_i,1].plot(pressure_distr[mol],pressures)
                axs[mol_i,1].set_yscale('log')
                axs[mol_i,1].set_ylim([1e2,1e-6])
                
                axs[mol_i,0].set_ylabel(mol)
        if plot_distr:
            figs.savefig(output_dir + 'em_contr_fct_VS_press_distr_'+which_em[:4] + '_AB' + which_abund[:4] + '_' + which_included[:4] +'.png',dpi=600)
        
        return pressure_distr,wlen_lbl_mol,flux_diff_interped,contr_em_fct
    
    def calc_em_contr_pressure_distr(
            self,
            config,
            samples,
            data_obj,
            contribution = True,
            which_em = 'molecules', # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
            which_abund = 'retr_abund', # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
            which_included = 'included', # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
            output_dir = '',
            plot_distr = True
            ):
        
        
        ab_metals,temp_params = calc_retrieved_params(config,samples)
        
        wlen_lbl,flux_lbl = self.calc_spectrum(
                ab_metals = ab_metals,
                temp_params = temp_params,
                clouds_params = {},
                external_pt_profile = None,
                return_profiles = False,
                contribution=contribution)
        contr_em_fct = None
        if which_em == 'retrieved':
            contr_em_fct = np.array(self.calc_emission_contribution())
        wlen_included,flux_included = None,None
        which_data_format = None
        if data_obj.CCinDATA():
            CC_wlen_data_dic,CC_flux_data_dic = data_obj.getCCSpectrum()
            for key in CC_wlen_data_dic.keys():
                CC_wlen_data,CC_flux_data = CC_wlen_data_dic[key],CC_flux_data_dic[key]
            
            CC_wlen,CC_flux,calc_filter,wlen_rebin_datalike,flux_rebin_datalike = rebin_to_CC(wlen_lbl,flux_lbl,CC_wlen_data,win_len=config['WIN_LEN'],method='datalike',filter_method = 'only_gaussian',nb_sigma=5,convert = True,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            wlen_included,flux_included=CC_wlen,CC_flux
            which_data_format = 'CC'
            
        if data_obj.RESinDATA():
            RES_data_wlen,RES_data_flux,RES_data_err,RES_inv_cov,RES_data_flux_err = data_obj.getRESSpectrum()
            for key in RES_data_wlen.keys():
                RES_wlen_data,RES_flux_data = RES_data_wlen[key],RES_data_flux[key]
            
            RES_wlen,RES_flux = rebin_to_RES(wlen_lbl,flux_lbl,RES_wlen_data,log_R=temp_params['log_R'],distance=config['DISTANCE'])
            if not data_obj.CCinDATA():
                wlen_included,flux_included=RES_wlen,RES_flux
                which_data_format = 'RES'
        
        pressure_distr,wlen_lbl_ref,CC_flux_diff_interped,contr_em_fct = self.calc_pressure_distribution(
                config,
                contr_em_fct,
                ab_metals,
                temp_params,
                wlen_included,
                flux_included,
                which_data_format = which_data_format,
                which_em = which_em, # or 'molecules' refering to whether we take em contr fct from retrieved spectrum or from each molecule
                which_abund = which_abund, # 'high_abund' or 'retr_abund' refering to the amount of the molecules to include when calculating em contr fct and flux
                which_included = which_included, # 'excluded' or 'included' refering to whether we only include one molecule at a time, or if we only exclude one molecule at a time but include all others
                output_dir = output_dir,
                plot_distr = False)
        
        return wlen_lbl_ref,contr_em_fct,CC_flux_diff_interped,pressure_distr
    
"""
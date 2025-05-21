# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:45:22 2021

@author: jeanh
"""

import sys
import os

from os import path
from petitRADTRANS import radtrans as rt
from petitRADTRANS import nat_cst as nc
import numpy as np
from typing import Optional,Union
from core.util import poor_mans_abunds_ck,poor_mans_abunds_lbl,convert_to_ck_names
from core.model import get_temperatures,evaluate_forward_model

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
        
        if 'chem_equ' in self.chem_model:
            if self.mode == 'c-k':
                self.line_opacities = poor_mans_abunds_ck().copy()
            else:
                self.line_opacities = poor_mans_abunds_lbl().copy()
            if self.only_include != 'all':
                self.line_opacities = self.only_include
                if include_H2:
                    self.line_opacities += ['H2_main_iso']
        else:
            # 'free' model
            if self.mode == 'c-k':
                self.line_opacities = list(np.unique(convert_to_ck_names(self.line_opacities)))
        
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
                      chem_model_params:dict,
                      temp_model_params:dict,
                      cloud_model_params:dict,
                      physical_params:dict,
                      external_pt_profile:Union[np.array,list] = None,
                      return_profiles:bool = False,
                      calc_contribution:bool = False
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
            pressures,temperatures=get_temperatures(temp_model=self.temp_model,temp_model_params=temp_model_params,physical_model_params=physical_params)
        
        # setup up opacity structure if not yet done so
        if not(self.opa_struct_set):
            self.rt_obj.setup_opa_structure(pressures)
            self.opa_struct_set = True
        
        wlen,flux,pressures,temperatures,abundances,contr_em = evaluate_forward_model(
            rt_object=self.rt_obj,
            chem_model=self.chem_model,
            chem_model_params=chem_model_params,
            temp_model=self.temp_model,
            temp_model_params=temp_model_params,
            cloud_model=self.cloud_model,
            cloud_model_params=cloud_model_params,
            physical_params=physical_params,
            mode=self.mode,
            only_include=self.only_include,
            external_pt_profile=external_pt_profile,
            calc_contribution=calc_contribution)
        
        if self.only_include != 'all':
            self.only_include = list(dict.fromkeys(self.only_include))
            assert(len(abundances.keys())==len(self.only_include)+2)
        
        if return_profiles:
            if calc_contribution:
                return wlen,flux,pressures,temperatures,abundances,contr_em
            else:
                return wlen,flux,pressures,temperatures,abundances
        else:
            if calc_contribution:
                return wlen,flux,contr_em
            else:
                return wlen,flux
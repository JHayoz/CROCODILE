# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:15:36 2021

@author: jeanh
"""

import numpy as np
import sys
import petitRADTRANS as rt
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import physics
from petitRADTRANS.poor_mans_nonequ_chem import *

from core.util import filter_relevant_mass_fractions,get_MMWs,name_ck
import sys

def guillot_temp_model(
        temp_model_params:dict
    ):
    """
    Calculates the p-T profile from the Guillot model.

    Parameters
    ----------
    temp_model_params : dict
        Parameters to evaluate the Guillot model.
        Need the following keys: P0,log_kappa_IR,log_gamma,log_gravity,t_int,t_equ
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    temperatures:dict
        Array containing the temperature of each atmospheric layer.
    """
    pressures = np.logspace(-6, temp_model_params['P0'], 100)
    temperatures = physics.guillot_global(
                pressures,
                1e1**temp_model_params['log_kappa_IR'],
                1e1**temp_model_params['log_gamma'],
                1e1**temp_model_params['log_gravity'],
                temp_model_params['t_int'],
                temp_model_params['t_equ'])
    return pressures,temperatures

def free_chem_model(
        chem_model_params:dict,
        pressures:np.array,
        temperatures:np.array,
        mode:str
    ):
    """
    Calculates the abundances from the free model, i.e. vertically constant molecular abundances.

    Parameters
    ----------
    chem_model_params : dict
        Dictionary containing the parameters needed to evaluate the chemical model, i.e. the mass fractions of each molecule
    pressures:np.array
        Array containing the pressure layers to calculate the abundances at.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    mode:str
        Mode
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    abundances:dict
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    """
    abundances = {}
    metal_sum = 0
    for name in chem_model_params.keys():
        if mode =='c-k':
            mol_name = name_ck(name)
        else:
            mol_name = name
        abundances[mol_name] = np.ones_like(pressures)*1e1**chem_model_params[name]
        metal_sum += 1e1**chem_model_params[name]
    
    abH2He = 1. - metal_sum
    
    if mode == 'lbl':
        abundances['H2_main_iso'] = abH2He*0.75 * np.ones_like(pressures)
        abundances['H2'] = abH2He*0.75 * np.ones_like(pressures)
    else:
        abundances['H2'] = abH2He*0.75 * np.ones_like(pressures)
    
    abundances['He'] = abH2He*0.25 * np.ones_like(pressures)
    return pressures,abundances

def chem_equ_model(
        chem_model_params:dict,
        pressures:np.array,
        temperatures:np.array,
        mode:str
    ):
    """
    Calculates the abundances from the chemical equilibrium model, i.e. for given p-T profile, calculate the equilibrium chemistry.

    Parameters
    ----------
    chem_model_params : dict
        Dictionary containing the parameters needed to evaluate the chemical model, i.e. C/O ratio and Fe/H
    pressures:np.array
        Array containing the pressure layers to calculate the abundances at.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    mode:str
        Mode
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    abundances:dict
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    """
    COs = chem_model_params['C/O']*np.ones_like(pressures)
    FeHs = chem_model_params['FeHs']*np.ones_like(pressures)
    
    mass_fractions = poor_mans_nonequ_chem.interpol_abundances(
        COs,
        FeHs,
        temperatures,
        pressures,
        Pquench_carbon = chem_model_params['Pquench_carbon'])
    
    abundances = filter_relevant_mass_fractions(mass_fractions,mode)
    
    return pressures,abundances

def get_abundances(
        chem_model:str,
        chem_model_params:dict,
        pressures:np.array,
        temperatures:np.array,
        mode:str
    ):
    """
    Calculates the abundances from the chemical model and parameters given.

    Parameters
    ----------
    chem_model : str
        Name of the chemical model to use.
    chem_model_params : dict
        Dictionary containing the parameters needed to evaluate the p-T model
    pressures:np.array
        Array containing the pressure layers to calculate the abundances at.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    mode:str
        Mode
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    abundances:dict
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    """
    if chem_model == 'free':
        print('Evaluating free chemical model')
        pressures,abundances = free_chem_model(chem_model_params,pressures,temperatures,mode=mode)
    elif chem_model=='chem_equ':
        print('Evaluating chemical equilibrium model')
        pressures,abundances = chem_equ_model(chem_model_params,pressures,temperatures,mode=mode)
    else:
        pressures,abundances =None,None
    return pressures,abundances
    
def get_temperatures(
        temp_model:str,
        temp_model_params:dict
    ):
    """
    Calculates the p-T profile from the model and parameters given.

    Parameters
    ----------
    temp_model : str
        Name of the p-T model to use.
    temp_model_params : dict
        Dictionary containing the parameters needed to evaluate the p-T model
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    """
    if temp_model == 'guillot':
        pressures,temperatures = guillot_temp_model(temp_model_params)
    else:
        pressures,temperatures =None, None
    
    return pressures,temperatures
    
def evaluate_forward_model(
        rt_object:rt.Radtrans,
        chem_model:str,
        chem_model_params:dict,
        temp_model:str,
        temp_model_params:dict,
        cloud_model:str,
        cloud_model_params:dict,
        physical_params:dict,
        mode:str,
        only_include:list
    ):
    """
    Calculates the spectrum from the models and model parameters given.

    Parameters
    ----------
    chem_model : str
        Name of the chemical model to use.
    chem_model_params : dict
        Dictionary containing the parameters needed to evaluate the p-T model
    pressures:np.array
        Array containing the pressure layers to calculate the abundances at.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    mode:str
        Mode for the spectral resolution to calculate the spectrum with, i.e. 'c-k' or 'lbl'.
    
    Returns
    -------
    pressures:np.array
        Array containing the pressure layers of the atmosphere.
    abundances:dict
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    """
    pressures,temperatures = get_temperatures(temp_model,temp_model_params)
    pressures,abundances = get_abundances(chem_model,chem_model_params,pressures,temperatures,mode=mode)
    
    if only_include != 'all':
        for mol in only_include:
            print(mol)
            assert(mol in abundances.keys() or mol == 'H2_main_iso')
        new_abundances = {key:abundances[key] for key in only_include if key!= 'H2_main_iso'}
        new_abundances['H2_main_iso'] = abundances['H2']
        new_abundances['H2'] = abundances['H2']
        new_abundances['He'] = abundances['He']
        abundances = {}
        abundances = new_abundances
    if mode == 'lbl':
        abundances['H2_main_iso'] = abundances['H2']
    
    
    wlen,flux=rt_obj_calc_flux(rt_object,
                     temperatures,
                     abundances,
                     gravity=1e1**physical_params['log_gravity'],
                     clouds_params=cloud_model_params,
                     contribution=False)
    return wlen,flux,pressures,temperatures,abundances

def calc_MMW(
        abundances:dict
    ):
    """
    Calculates the mean molecular weight of the atmosphere.

    Parameters
    ----------
    abundances : str
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    
    Returns
    -------
    float mean molecular weight
    """
    MMW = 0.
    for key in abundances.keys():
        MMW += abundances[key]/get_MMWs(key)
    return 1./MMW


def rt_obj_calc_flux(
        rt_object,
        temperatures:np.array,
        abundances:dict,
        gravity:float,
        clouds_params:dict,
        contribution:bool=False
    ):
    """
    Calculates the spectrum.

    Parameters
    ----------
    rt_object : str
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    temperatures:np.array
        Array containing the temperature of each atmospheric layer.
    abundances:dict
        dictionary containing the abundance of each molecule/opacity at each atmospheric layer.
    gravity:float
        surface gravity
    clouds_params:dict
        dictionary containing the parameters for the cloud model
    contribution:bool
        Whether to compute the contribution emission function
    
    Returns
    -------
    wlen:np.array
        wavelength bins of the computed spectrum
    flux:np.array
        flux bins of the computed spectrum
    """
    MMW = calc_MMW(abundances)
    
    Pcloud = None
    if 'log_Pcloud' in clouds_params.keys():
        print('CAREFUL: YOU ARE USING CLOUDS')
        Pcloud = 10**clouds_params['log_Pcloud']
    
    kzz,fsed,sigma_lnorm = None,None,None
    add_cloud_scat_as_abs = False
    if 'log_kzz' in clouds_params.keys() and 'fsed' in clouds_params.keys() and 'sigma_lnorm' in clouds_params.keys() and 'cloud_abunds' in clouds_params.keys():
        print('CAREFUL: YOU ARE USING CLOUDS')
        kzz = 10**clouds_params['log_kzz']*np.ones_like(temperatures)
        fsed = clouds_params['fsed']
        sigma_lnorm = clouds_params['sigma_lnorm']
        
        for cloud_abund in clouds_params['cloud_abunds'].keys():
            abundances[cloud_abund] = 10**clouds_params['cloud_abunds'][cloud_abund] * np.ones_like(temperatures)
        
        if 'scattering' in clouds_params.keys():
            add_cloud_scat_as_abs = clouds_params['scattering']
            print('adding scattering')
    
    rt_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        Pcloud = Pcloud,
                        contribution = contribution,
                        sigma_lnorm = sigma_lnorm,
                        fsed = fsed,
                        Kzz = kzz,
                        add_cloud_scat_as_abs = add_cloud_scat_as_abs
                        )
    wlen,flux = nc.c/rt_object.freq, rt_object.flux
    return wlen,flux
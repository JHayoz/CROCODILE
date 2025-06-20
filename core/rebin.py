# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:39:55 2021

@author: jeanh
"""
import numpy as np
from PyAstronomy.pyasl import dopplerShift
from spectres import spectres
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter,median_filter
from time import time
from core.util import synthetic_photometry,calc_median_filter,effective_width_filter

# help functions to rebin the flux
def rebin(wlen,flux,wlen_data,flux_err = None, method='linear'):
    #wlen larger than wlen_data
    
    
    #if method == 'linear':
    #extends wlen linearly outside of wlen_data using the spacing on each side
    if method == 'linear':
        stepsize_left = abs(wlen_data[1]-wlen_data[0])
        
        N_left = int((wlen_data[0]-wlen[0])/stepsize_left)-1
        wlen_left = np.linspace(wlen_data[0]-N_left*stepsize_left,
                                wlen_data[0],
                                N_left,
                                endpoint=False)
        
        stepsize_right = wlen_data[-1]-wlen_data[-2]
        
        N_right = int((wlen[-1]-wlen_data[-1])/stepsize_right)-1
        wlen_right = np.linspace(wlen_data[-1]+stepsize_right,
                                wlen_data[-1]+(N_right+1)*stepsize_right,
                                N_right,
                                endpoint=False)
        
        wlen_temp = np.concatenate((wlen_left,wlen_data,wlen_right))
    elif method == 'datalike':
        wlen_temp = wlen_data
    if flux_err is not None:
        assert(np.shape(flux_err)==np.shape(flux))
        flux_temp,flux_new_err = spectres(wlen_temp,wlen,flux,spec_errs = flux_err)
        return wlen_temp,flux_temp,flux_new_err
    else:
        flux_temp = spectres(wlen_temp,wlen,flux)
        return wlen_temp,flux_temp

def remove_cont(wlen,flux,win_len,wlen_after = None):
    
    sgfilter = savgol_filter(flux, window_length=win_len, polyorder=3)
    
    if wlen_after is not None:
        wvl_indices = [i for i in range(len(wlen)) if wlen[i] >= wlen_after[0] and wlen[i] <= wlen_after[-1]]
    else:
        wvl_indices = range(int((win_len-1)/2),int(len(wlen)-(win_len-1)/2))
    
    wlen_temp,flux_temp,temp_sgfilter = np.transpose(
        [
            [wlen[i],flux[i]-sgfilter[i],sgfilter[i]] for i in wvl_indices
        ])
    return wlen_temp,flux_temp,temp_sgfilter

def doppler_shift(wlen,flux,RV):
    flux_temp,wlen_prime = dopplerShift(wlen,
                             flux,
                             RV,
                             edgeHandling='firstlast')
    
    wlen_temp,flux_temp = np.transpose(
    [
        [wlen[i],flux_temp[i]] for i in range(len(wlen))
        if (wlen[i] >= wlen_prime[0] and wlen[i] <= wlen_prime[-1])
    ])
    
    return wlen_temp,flux_temp

def rebin_to_CC(wlen,flux,wlen_data,win_len,method='linear',filter_method = 'only_gaussian',nb_sigma=5):
    t0 = time()
    # rebinning
    wlen_rebin,flux_rebin = rebin(wlen,flux,wlen_data)
    wlen_rebin_datalike,flux_rebin_datalike = np.transpose([[wlen_rebin[i],flux_rebin[i]] for i in range(len(wlen_rebin)) if wlen_rebin[i] >= wlen_data[0] and wlen_rebin[i] <= wlen_data[-1]])
    
    # remove continuum with savitzky-golay filter
    if filter_method == 'sgfilter':
        if method == 'linear':
            wlen_removed,flux_removed,calc_filter = remove_cont(wlen_rebin,flux_rebin,win_len)
        else:
            wlen_removed,flux_removed,calc_filter = remove_cont(wlen_rebin,flux_rebin,win_len,wlen_after = wlen_data)
    elif filter_method == '5sigma_sgfilter':
        # if filter_method == '5sigma_sgfilter'
        if method == 'linear':
            wlen_removed,flux_removed,calc_filter = sigma_clipping_SG_filter(wlen_rebin,flux_rebin,win_len,wlen_after=None,nb_sigma = nb_sigma)
        else:
            wlen_removed,flux_removed,calc_filter = sigma_clipping_SG_filter(wlen_rebin,flux_rebin,win_len,wlen_after=wlen_data,nb_sigma = nb_sigma)
    elif filter_method == 'gaussian':
        # if filter_method == 'gaussian'
        if method == 'linear':
            wlen_removed,flux_removed,calc_filter = sigma_clipping_gauss_filter(wlen_rebin,flux_rebin,win_len,order_filter=0,wlen_after=None,nb_sigma = 5)
        else:
            wlen_removed,flux_removed,calc_filter = sigma_clipping_gauss_filter(wlen_rebin,flux_rebin,win_len,order_filter=0,wlen_after=wlen_data,nb_sigma = 5)
    elif filter_method == 'only_gaussian':
        if method == 'linear':
            wlen_removed,flux_removed,calc_filter = only_gaussian_filter(wlen_rebin,flux_rebin,sigma=win_len,wlen_after=None)
        else:
            wlen_removed,flux_removed,calc_filter = only_gaussian_filter(wlen_rebin,flux_rebin,sigma=win_len,wlen_after=wlen_data)
    else:
        # print('Using median-gaussian filter')
        # if filter_method == 'median_filter'
        if method == 'linear':
            wlen_removed,flux_removed,calc_filter = median_gaussian_filter(wlen_rebin,flux_rebin,win_len,nb_sigma=nb_sigma,wlen_after=None)
        else:
            wlen_removed,flux_removed,calc_filter = median_gaussian_filter(wlen_rebin,flux_rebin,win_len,nb_sigma=nb_sigma,wlen_after=wlen_data)
    
    return wlen_removed,flux_removed,calc_filter,wlen_rebin_datalike,flux_rebin_datalike

def rebin_to_RES(wlen,flux,wlen_data):
    
    wlen_rebin,flux_rebin = rebin(wlen,flux,wlen_data,method = 'datalike')
    
    return wlen_rebin,flux_rebin
    
def rebin_to_PHOT(wlen,flux,filt_func,phot_flux_data = None,phot_flux_err_data = None):
    
    # calculate photometry for forward model
    model_photometry = {}
    model_photometry_err = {}
    phot_midpoint={}
    phot_width={}
    for instr in filt_func.keys():
        model_photometry[instr] = synthetic_photometry(wlen,flux,filt_func[instr])
        if phot_flux_data is not None and phot_flux_err_data is not None:
            model_photometry_err[instr] = phot_flux_err_data[instr]/phot_flux_data[instr]*model_photometry[instr]
            phot_midpoint[instr] = calc_median_filter(filt_func[instr],N_points=2000)
            phot_width[instr] = effective_width_filter(filt_func[instr],N_points=2000)
    if phot_flux_data is not None and phot_flux_err_data is not None:
        return model_photometry,model_photometry_err,phot_midpoint,phot_width
    else:
        return model_photometry

def only_gaussian_filter(wlen,flux,sigma=None,wlen_after=None):
    if sigma is None:
        resolution = wlen[0]/(wlen[1]-wlen[0])
        sigma = 2*resolution/1000
    
    filt = gaussian_filter(flux,sigma=sigma,mode='nearest')
    
    if wlen_after is not None:
        wvl_indices = [i for i in range(len(wlen)) if wlen[i] >= wlen_after[0] and wlen[i] <= wlen_after[-1]]
    else:
        wvl_indices = range(int(2*sigma),int(len(wlen)-2*sigma))
    
    # remove the continuum using the filter calculated from the spectrum without the outliers
    wlen_temp,flux_temp,temp_filt = np.transpose(
        [
            [wlen[i],flux[i]-filt[i],filt[i]] for i in wvl_indices
        ])
    
    return wlen_temp,flux_temp,temp_filt


def sigma_clipping_SG_filter(wlen,flux,win_len,wlen_after=None,nb_sigma = 5):
    
    # first calculate normal s-g filter
    sgfilter_first = savgol_filter(flux, window_length=3*win_len, polyorder=3)
    
    flux_outlier_removed = remove_outliers(flux,sgfilter_first,sigma = nb_sigma,win_len_outliers = 31)
    
    # re-calculate the s-g filter, but this time applied to the spectrum where the outliers have already been ignored
    sgfilter = savgol_filter(flux_outlier_removed, window_length=win_len, polyorder=3)
    
    if wlen_after is not None:
        wvl_indices = [i for i in range(len(wlen)) if wlen[i] >= wlen_after[0] and wlen[i] <= wlen_after[-1]]
    else:
        wvl_indices = range(int((win_len-1)/2),int(len(wlen)-(win_len-1)/2))
    # remove the continuum using the filter calculated from the spectrum without the outliers
    wlen_temp,flux_temp,temp_sgfilter = np.transpose(
        [
            [wlen[i],flux[i]-sgfilter[i],sgfilter[i]] for i in wvl_indices
        ])
    
    return wlen_temp,flux_temp,temp_sgfilter

def rolling_median_filter(wlen,flux,win_len):
    if win_len%2 == 1:
        print('NEED TO USE EVEN WINDOW LENGTH')
    nb_calculations = len(flux)-win_len + 1
    rolling_median = np.zeros((nb_calculations,))
    for i in range(nb_calculations):
        rolling_median[i] = np.median(flux[i:i+win_len+1])
    assert(len(wlen[int((win_len-1)/2)+1:-int((win_len-1)/2)+1])==len(rolling_median))
    return wlen[int((win_len-1)/2)+1:-int((win_len-1)/2)+1],rolling_median

def sigma_clipping_gauss_filter(wlen,flux,sigma_filter,order_filter=0,wlen_after=None,nb_sigma = 4):
    # first calculate normal gaussian filter
    t0 = time()
    gaussfilter_first = gaussian_filter(flux,sigma = sigma_filter*3, order=order_filter, mode = 'nearest')
    t1 = time()
    flux_outlier_removed = remove_outliers(flux,gaussfilter_first,sigma = nb_sigma,win_len_outliers = 51)
    t2 = time()
    gaussfilter = gaussian_filter(flux_outlier_removed,sigma = sigma_filter, order=order_filter, mode = 'nearest')
    t3 = time()
    print('Time total: {T:0.3f}, outliers {t2:0.3f}, filters {t3:0.3f}'.format(T=t3-t0,t2=t2-t1,t3=t3-t0-t2+t1))
    nb_bins_ignored = int(sigma_filter*2)
    if wlen_after is not None:
        wvl_indices = [i for i in range(len(wlen)) if wlen[i] >= wlen_after[0] and wlen[i] <= wlen_after[-1]]
    else:
        wvl_indices = range(nb_bins_ignored,len(wlen)-nb_bins_ignored)
    
    # remove the continuum using the filter calculated from the spectrum without the outliers
    wlen_temp,flux_temp,temp_gaussfilter = np.transpose(
        [
            [wlen[i],flux[i]-gaussfilter[i],gaussfilter[i]] for i in wvl_indices
        ])
    
    return wlen_temp,flux_temp,temp_gaussfilter

def median_gaussian_filter(wlen,flux,win_len=64,nb_sigma=16,wlen_after=None):
    # first calculate normal gaussian filter
    """
    gaussfilter_first = gaussian_filter(flux,sigma = sigma_filter*3, order=order_filter, mode = 'nearest')
    
    flux_outlier_removed = remove_outliers(flux,gaussfilter_first,sigma = nb_sigma,win_len_outliers = 51)
    """
    t0 = time()
    med_filter = median_filter(flux,size = win_len, mode = 'nearest')
    t1=time()
    # print('Median filter',t1-t0)
    smooth_filter = gaussian_filter(med_filter,sigma = nb_sigma, order=0, mode = 'nearest')
    t2=time()
    # print('Gaussian filter',t2-t1)
    nb_bins_ignored = int(win_len/2)
    if wlen_after is not None:
        wvl_indices = [i for i in range(len(wlen)) if wlen[i] >= wlen_after[0] and wlen[i] <= wlen_after[-1]]
    else:
        wvl_indices = range(nb_bins_ignored,len(wlen)-nb_bins_ignored)
    
    # remove the continuum using the filter calculated from the spectrum without the outliers
    wlen_temp,flux_temp,temp_gaussfilter = np.transpose(
        [
            [wlen[i],flux[i]-smooth_filter[i],smooth_filter[i]] for i in wvl_indices
        ])
    return wlen_temp,flux_temp,temp_gaussfilter

def remove_outliers(flux,smoothed_flux,sigma = 5,win_len_outliers = 31):
    flux_removed = flux - smoothed_flux
    # calculate 16-th, 50-th, and 84-th percentile (like sigma)
    WL = win_len_outliers
    q1,q2,q3=np.quantile([flux_removed[bin_i:bin_i+WL] for bin_i in range(len(flux_removed)-WL+1)],q=[0.16,0.5,0.84],axis=1)
    sigma_bottom,sigma_top = np.abs(q2-q1),np.abs(q3-q2)
    sigma_bottom = np.hstack((sigma_bottom[0]*np.ones((int((WL-1)/2))),sigma_bottom,sigma_bottom[-1]*np.ones((int((WL-1)/2)))))
    sigma_top = np.hstack((sigma_top[0]*np.ones((int((WL-1)/2))),sigma_top,sigma_top[-1]*np.ones((int((WL-1)/2)))))
    
    assert(len(sigma_top)==len(flux_removed))
    outliers_top = flux_removed > sigma*sigma_top
    outliers_bot = flux_removed < -sigma*sigma_bottom
    outliers = [outliers_top[i] or outliers_bot[i] for i in range(len(outliers_top))]
    
    #flux_outlier_removed = np.array([flux[i] if not outliers[i] else smoothed_flux[i] for i in range(len(flux))])
    
    median_flux = [np.median(flux[bin_i - winlenfunc(bin_i,len(flux_removed),WL):1+bin_i+winlenfunc(bin_i,len(flux_removed),WL)]) for bin_i in range(len(flux_removed))]
    
    
    flux_outlier_removed = np.array([flux[i] if not outliers[i] else median_flux[i] for i in range(len(flux))])
    
    return flux_outlier_removed

def winlenfunc(bin_i,arr_len,wl):
    return min(min(bin_i,arr_len-1-bin_i),int((wl-1)/2))

def remove_outliers2(flux,smoothed_flux,sigma = 5,nb_extended = 10):
    flux_removed = flux - smoothed_flux
    # calculate 16-th, 50-th, and 84-th percentile (like sigma)
    q1,q2,q3=np.quantile(flux_removed,q=[0.16,0.5,0.84])
    sigma_bottom,sigma_top = abs(q2-q1),abs(q3-q2)
    # ignore the bins that are outside of 5 sigma from the filter, or respectively replace those bins with the s-g filter
    outliers_top = flux_removed > sigma*sigma_top
    outliers_bot = flux_removed < -sigma*sigma_bottom
    outliers = [outliers_top[i] or outliers_bot[i] for i in range(len(outliers_top))]
    
    #first try was to replace outliers with filter at their location, but the filter also has bumps so not ideal
    #flux_outlier_removed = np.array([flux[i] if not outliers[i] else smoothed_flux[i] for i in range(len(flux))])
    
    # first extend flux with values at edge to avoid looking at inexistent flux values
    bin_extended = nb_extended
    flux_extended = np.hstack((flux[0]*np.ones((bin_extended)),flux,flux[-1]*np.ones((bin_extended))))
    outliers_extended = np.hstack((np.zeros((bin_extended)),outliers,np.zeros((bin_extended))))
    flux_outlier_removed = np.zeros_like(flux)
    for bin_i in range(bin_extended,bin_extended+len(flux)):
        flux_bin_i = bin_i-bin_extended
        flux_i = 0
        
        if sum(outliers_extended[bin_i-bin_extended:bin_i+bin_extended]) > bin_extended*3/4:
            flux_i = smoothed_flux[flux_bin_i]
        elif sum(outliers_extended[bin_i-bin_extended:bin_i+bin_extended]) >= 1:
            flux_i = np.median([flux_extended[bin_i-bin_extended:bin_i+bin_extended][j] for j in range(2*bin_extended) if outliers_extended[bin_i-bin_extended:bin_i+bin_extended][j] == False])
        else:
            flux_i = flux[flux_bin_i]
        flux_outlier_removed[flux_bin_i] = flux_i
    return flux_outlier_removed
    
    
from PyAstronomy.pyaC import pyaErrors as PE
from PyAstronomy.pyasl import _ic

def crosscorrRVnorm(w, f, tw, tf, rvmin, rvmax, drv, mode="doppler", skipedge=0, edgeTapering=None):
    """
    Cross-correlate a spectrum with a template.
    
    
    The algorithm implemented here works as follows: For
    each RV shift to be considered, the wavelength axis
    of the template is shifted, either linearly or using
    a proper Doppler shift depending on the `mode`. The
    shifted template is then linearly interpolated at
    the wavelength points of the observation
    (spectrum) to calculate the cross-correlation function.
    
    Parameters
    ----------
    w : array
        The wavelength axis of the observation.
    f : array
        The flux axis of the observation.
    tw : array
        The wavelength axis of the template.
    tf : array
        The flux axis of the template.
    rvmin : float
        Minimum radial velocity for which to calculate
        the cross-correlation function [km/s].
    rvmax : float
        Maximum radial velocity for which to calculate
        the cross-correlation function [km/s].
    drv : float
        The width of the radial-velocity steps to be applied
        in the calculation of the cross-correlation
        function [km/s].
    mode : string, {lin, doppler}, optional
        The mode determines how the wavelength axis will be
        modified to represent a RV shift. If "lin" is specified,
        a mean wavelength shift will be calculated based on the
        mean wavelength of the observation. The wavelength axis
        will then be shifted by that amount. If "doppler" is
        specified (the default), the wavelength axis will
        properly be Doppler-shifted.
    skipedge : int, optional
        If larger zero, the specified number of bins will be
        skipped from the begin and end of the observation. This
        may be useful if the template does not provide sufficient
        coverage of the observation.
    edgeTapering : float or tuple of two floats
        If not None, the method will "taper off" the edges of the
        observed spectrum by multiplying with a sine function. If a float number
        is specified, this will define the width (in wavelength units)
        to be used for tapering on both sides. If different tapering
        widths shall be used, a tuple with two (positive) numbers
        must be given, specifying the width to be used on the low- and
        high wavelength end. If a nonzero 'skipedge' is given, it
        will be applied first. Edge tapering can help to avoid
        edge effects (see, e.g., Gullberg and Lindegren 2002, A&A 390).
    
    Returns
    -------
    dRV : array
        The RV axis of the cross-correlation function. The radial
        velocity refer to a shift of the template, i.e., positive
        values indicate that the template has been red-shifted and
        negative numbers indicate a blue-shift of the template.
        The numbers are given in km/s.
    CC : array
        The cross-correlation function.
    """
    if not _ic.check["scipy"]:
        raise(PE.PyARequiredImport("This routine needs scipy (.interpolate.interp1d).", \
                                 where="crosscorrRV", \
                                 solution="Install scipy"))
    import scipy.interpolate as sci
    # Copy and cut wavelength and flux arrays
    w, f = w.copy(), f.copy()
    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]
    
    if edgeTapering is not None:
        # Smooth the edges using a sine
        if isinstance(edgeTapering, float):
            edgeTapering = [edgeTapering, edgeTapering]
        if len(edgeTapering) != 2:
            raise(PE.PyAValError("'edgeTapering' must be a float or a list of two floats.", \
                               where="crosscorrRV"))
        if edgeTapering[0] < 0.0 or edgeTapering[1] < 0.0:
            raise(PE.PyAValError("'edgeTapering' must be (a) number(s) >= 0.0.", \
                               where="crosscorrRV"))
        # Carry out edge tapering (left edge)
        indi = np.where(w < w[0]+edgeTapering[0])[0]
        f[indi] *= np.sin((w[indi] - w[0])/edgeTapering[0]*np.pi/2.0)
        # Carry out edge tapering (right edge)
        indi = np.where(w > (w[-1]-edgeTapering[1]))[0]
        f[indi] *= np.sin((w[indi] - w[indi[0]])/edgeTapering[1]*np.pi/2.0 + np.pi/2.0)
    
    # Speed of light in km/s
    c = 299792.458
    # Check order of rvmin and rvmax
    if rvmax <= rvmin:
        raise(PE.PyAValError("rvmin needs to be smaller than rvmax.",
                           where="crosscorrRV", \
                           solution="Change the order of the parameters."))
    # Check whether template is large enough
    if mode == "lin":
        meanWl = np.mean(w)
        dwlmax = meanWl * (rvmax/c)
        dwlmin = meanWl * (rvmin/c)
        if (tw[0] + dwlmax) > w[0]:
            raise(PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                             where="crosscorrRV", \
                             solution=["Provide a larger template", "Try to use skipedge"]))
        if (tw[-1] + dwlmin) < w[-1]:
            raise(PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                             where="crosscorrRV", \
                             solution=["Provide a larger template", "Try to use skipedge"]))
    elif mode == "doppler":
        # Ensure that the template covers the entire observation for all shifts
        maxwl = tw[-1] * (1.0+rvmin/c)
        minwl = tw[0] * (1.0+rvmax/c)
        if minwl > w[0]:
            raise(PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                             where="crosscorrRV", \
                             solution=["Provide a larger template", "Try to use skipedge"]))
        if maxwl < w[-1]:
            raise(PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                             where="crosscorrRV", \
                             solution=["Provide a larger template", "Try to use skipedge"]))
    else:
        raise(PE.PyAValError("Unknown mode: " + str(mode), \
                           where="crosscorrRV", \
                           solution="See documentation for available modes."))
    # Calculate the cross correlation
    drvs = np.arange(rvmin, rvmax, drv)
    cc = np.zeros(len(drvs))
    for i, rv in enumerate(drvs):
        if mode == "lin":
            # Shift the template linearly
            fi = sci.interp1d(tw+meanWl*(rv/c), tf)
        elif mode == "doppler":
            # Apply the Doppler shift
            fi = sci.interp1d(tw*(1.0 + rv/c), tf)
        # Shifted template evaluated at location of spectrum
        cc[i] = np.sum(f * fi(w))/np.sqrt(np.sum(f*f)*np.sum(fi(w)*fi(w)))
    return cc, drvs
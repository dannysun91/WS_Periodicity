'''
This file defines and runs plotting routines for the wavelet/FFT/peak finder summary plots shown on
my AAS/SPD 54 poster for the PSP E14 intervals 2022.12.13 19:30-21:00 and 2022.12.14 04:40-05:50.
'''

#Imports
from datetime import datetime, timezone, timedelta
import numpy as np
import pycwt
from scipy import fft
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

import gather_functions as gf
import lightcurve_class as l
import utility as u

from os import listdir
import pandas as pd


def summary_plot(lc, bool_show=False, bool_save=False, smoothing):
    '''
    Generates a standard wavelet plot for the given lightcurve with the following changes:
    - FFT plot in the top right
    - Peak finder result as a line on the global wavelet power spectrum
    - Customized axis label text and sizes
    -----------
    Parameters:
    -----------
    lc : LightCurve
        The LightCurve for which to do the plotting.
    bool_save : Boolean (optional)
        Toggles whether the figure is saved.
    bool_show : Boolean (optional)
        Toggles whether the figure is shown.
    '''
    #Creating lists with the time and power data (mean subtracted from power)
    power = [p - np.mean(lc.power) for p in lc.power]
    time = lc.timestamps
    t_step = lc.timestep
    #Normalizing the power data by the standard deviation
    std = np.std(power)
    var = std ** 2
    power_norm = power / std
    #Setting the parameters of our wavelet analysis
    mother = pycwt.Morlet(6) #Morlet with w_0 = 6
    #Starting scale set to twice the timestep to detect single-point spikes
    scale_0 = 2 * t_step
    #Specifying frequency resolution at 16 sub-octaves per octave
    freq_step = 1 / 16
    num_octaves = 8
    #Estimating the Lag-1 autocorrelation for red noise modeling
    try:
        alpha = pycwt.ar1(power)[0]
    except Warning:
        print(f'Unable to perform the wavelet on {lc} {lc.get_time_str()}. Series is ' + \
                'too short or trend is too large.')
        return 0
    #Performing the wavelet transform according to the parameters above
    wave, scales, freqs, coi, _, _ = pycwt.cwt(power_norm, t_step, freq_step, scale_0,
                                                num_octaves/freq_step, mother)
    #Calculating the wavelet and Fourier power spectra
    wave_power = (np.abs(wave)) ** 2
    periods = 1 / freqs
    periods_min = [p / 60 for p in periods]
    #Generating significance levels for the wavelet power spectrum
    #The power is significant when wave_power / sig95 > 1
    sig = pycwt.significance(1.0, t_step, scales, 0, alpha, significance_level=0.95,
                                wavelet=mother)[0]
    sig95 = np.ones([1, len(power)]) * sig[:, None]
    sig95 = wave_power / sig95
    #Calculating the global wavelet spectrum and significance levels for red and white noise
    glbl_power = wave_power.mean(axis=1)
    dof = len(power) - scales #Correction for edge effects
    red_sig = pycwt.significance(var, t_step, scales, 1, alpha, significance_level=0.95,
                                    dof=dof, wavelet=mother)[0]
    white_sig = pycwt.significance(var, t_step, scales, 1, 0, significance_level = 0.95,
                                    dof=dof, wavelet=mother)[0]
    #Creating the figure combining the raw lightcurve, wavelet power spectrum, and global wps
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 8), dpi=721)
    _ = plt.figure(**figprops, facecolor='white')
    #Plotting the raw lightcurve as the first subplot
    axx = plt.axes([0.05, 0.55, 0.58, 0.35])
    axx.plot(time, power_norm, color='k')
    title_str = str(lc) + '' + lc.get_time_str()
    axx.set_title(title_str, fontsize=20)
    axx.set_ylabel('Normalized Power', fontsize=18)
    axx.tick_params(axis='x', which='major', labelsize=14)
    #Plotting the normalized wavelet power spectrum with significance level contour
    #lines and the cone of influence marked as the second subplot
    bxx = plt.axes([0.05, 0.05, 0.58, 0.4], sharex=axx)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bxx.contourf(time, np.log2(periods_min), np.log2(wave_power), np.log2(levels),
                    extend='both', cmap=plt.cm.inferno)
    #bxx.contourf(time, np.log2(periods)
    extent = [min(time), max(time), 0, np.log2(max(periods_min))]
    bxx.contour(time, np.log2(periods_min), sig95, [-99, 1], colors='k', linewidths=2,
                extent=extent)
    bxx.fill(np.concatenate([time, time[-1:] + t_step, time[-1:] + t_step, time[:1] - t_step,
                                time[:1] - t_step]),
            np.concatenate([np.log2(coi/60), [1e-9], np.log2(periods_min[-1:]),
                            np.log2(periods_min[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    bxx.set_title('Morlet Wavelet Power Spectrum', fontsize=24)
    bxx.set_ylabel('Period (min)', fontsize=18)
    bxx.axvspan(2022-11-25 9:25:0, 2022-11-25 9:36:0, color="blue", alpha=0.3)
    ##Getting evenly spaced times in the interval for time axis labeling
    t_ticks = [time[i] for i in range(len(time)) if i % (len(time) // 5) == 0]
    t_labels = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%H:%M') for t in t_ticks]
    bxx.set_xticks(t_ticks)
    bxx.set_xticklabels(t_labels)
    ##Labeling the log-scale period axis with the corresponding periods in minutes
    p_labels = 2 ** np.arange(np.ceil(np.log2(min(periods_min))),
                            np.ceil(np.log2(max(periods_min))))
    bxx.set_yticks(np.log2(p_labels))
    bxx.set_yticklabels(p_labels)
    period_ax_range = [np.log2(min(periods_min)), np.log2(max(periods_min))]
    bxx.set_ylim(period_ax_range)
    bxx.tick_params(axis='both', which='major', labelsize=14)
    #Finding the peaks in the global wavelet power spectrum
    all_peak_indices = [i for i in find_peaks(glbl_power)[0]]
    red_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > red_sig[i]]
    white_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > white_sig[i]]
    #Plotting the global wavelet power spectrum with its significance levels as the 3rd subplot
    cxx = plt.axes([0.74, 0.05, 0.21, 0.4], sharey = bxx)
    for i in white_peak_indices:
        cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color='#696969')
    for i in red_peak_indices:
        cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color = '#FF6666')
    line1, = cxx.plot(var * glbl_power, np.log2(periods_min), 'k-')
    line2, = cxx.plot(red_sig, np.log2(periods_min), '--', color='#FF6666')
    line3, = cxx.plot(white_sig, np.log2(periods_min), '--', color='#696969')
    #Adding in the peak finder results
    _, _, pf_period, pf_std = lc.peak_finder(180, show_results=False)
    log_period_min = np.log2(pf_period / 60)
    log_plus = np.log2((pf_period + pf_std) / 60) - log_period_min
    log_minus = log_period_min - np.log2((pf_period - pf_std) / 60)
    cxx.set_xlim([-0.1 * var * max(glbl_power), 1.2 * var * max(glbl_power)])
    line4, = cxx.plot(cxx.get_xlim(), [log_period_min, log_period_min], 'b')
    cxx.errorbar(0, log_period_min, [[log_minus], [log_plus]], ecolor='b', capsize=5, markersize=0)
    if lc.source[:3] == 'PSP':
        loc = 'upper left'
    else:
        loc = 'upper right'
    cxx.legend([line1, line4, line2, line3], ['Global Wavelet Power', 'Peak Finder Period', 
                                                'Red Noise', 'White Noise'],
                loc=loc, prop={'size':6})
    cxx.tick_params(axis='y', which='major', labelsize=14)
    #Calculating the FFT
    #Subtracting the mean to remove the 0-frequency component
    balanced_data = lc.power - np.mean(lc.power)
    #Applying a Hanning window to the data
    window = np.hanning(len(lc.power))
    data_windowed = np.multiply(window, balanced_data)
    #Symmetrically zero padding the windowed data to the next fast length
    num_elements = fft.next_fast_len(len(lc.power), real=True)
    num_0s_added = num_elements - len(lc.power)
    #Adding half of the 0s before the data, and half after
    data_processed = []
    for _ in range(num_0s_added // 2):
        data_processed.append(0)
    for val in data_windowed:
        data_processed.append(val)
    for _ in range(num_0s_added - (num_0s_added // 2)):
        data_processed.append(0)
    #Now that preprocessing is done, the power spectrum is calculated
    data_fft = fft.rfft(data_processed)
    pos_freq_abs_fft = np.abs(data_fft[:len(data_processed) // 2])
    fft_power = [val ** 2 for val in pos_freq_abs_fft]
    freq_bin = fft.rfftfreq(len(data_processed))[:len(data_processed) // 2]
    freq_bin /= lc.timestep #Converting the fft frequencies to meaningful values
    periods = 1 / freq_bin[1:] #seconds
    periods /= 60 #minutes
    #Plotting the FFT spectrum in the top right
    fxx = plt.axes([0.74, 0.55, 0.21, 0.35])
    fxx.plot(fft_power[1:], np.log2(periods))
    fxx.set_title("FFT Power", fontsize=18)
    fxx.set_ylabel("Period (min)", fontsize=18)
    #Labeling the log-scale period axis nicely
    p_tick_labels = 2 ** np.arange(np.ceil(np.log2(min(periods))),
                                    np.ceil(np.log2(max(periods))))
    fxx.set_yticks(np.log2(p_tick_labels))
    fxx.set_yticklabels([f'{f:.1f}' for f in p_tick_labels])
    fxx.set_ylim([np.log2(min(periods_min)), np.log2(max(periods_min))])
    fxx.tick_params(axis='y', which='major', labelsize=14)
    #Saving the finished plot
    if bool_save:
        fname = u.write_fnames(lc, 'summary',smoothing)[1]
        plt.savefig(fname, bbox_inches='tight')
    if bool_show:
        plt.show()
    plt.close()

def psp13():
    '''
    Gathers PSP data for the 12.13 interval and makes the plot.
    '''
    start = datetime(2022, 12, 13, 19, 30, tzinfo=timezone.utc)
    end = start + timedelta(minutes=80)
    all_lc = gf.gather_psp_local_files(start, end, '03', 'combination')
    lc = u.get_lc_at_freq(all_lc, 4)
    summary_plot(lc, 'PSP 4.022 MHz 2022-12-13', '../../Desktop/poster_figures/psp13.png')

def psp14():
    '''
    Gathers PSP data for the 12.14 interval and makes the plot.
    '''
    start = datetime(2022, 12, 14, 4, 40, tzinfo=timezone.utc)
    end = start + timedelta(minutes=60)
    all_lc = gf.gather_psp_local_files(start, end, '03', 'combination')
    lc = u.get_lc_at_freq(all_lc, 2)
    summary_plot(lc, 'PSP 2.025 MHz 2022-12-14', '../../Desktop/poster_figures/psp14.png')

def aia13():
    '''
    Gathers AIA data for the 12.13 interval and makes the plot.
    '''
    start = datetime(2022, 12, 13, 19, 20, tzinfo=timezone.utc)
    end = datetime(2022, 12, 13, 21, 10, tzinfo=timezone.utc)
    lc = gf.gather_aia(start, end, 171, 1) #Waiting on file from Reed
    #Splitting the lightcurve up to get the interval from 19:37-20:57
    split_t = datetime(2022, 12, 13, 19, 37, tzinfo=timezone.utc)
    lc = l.split_lc(lc, split_t)[1]
    split_t = datetime(2022, 12, 13, 20, 57, tzinfo=timezone.utc)
    lc = l.split_lc(lc, split_t)[0]
    summary_plot(lc, 'AIA Reg 1 171A 2022-12-13', '../../Desktop/poster_figures/aia13.png')

def aia14():
    '''
    Gathers AIA data for the 12.14 interval and makes the plot.
    '''
    start = datetime(2022, 12, 14, 4, 35, tzinfo=timezone.utc)
    end = datetime(2022, 12, 14, 7, 10, tzinfo=timezone.utc)
    lc = gf.gather_aia(start, end, 171, 3)
    #Splitting the lightcurve up to get the interval from 4:47-5:47
    split_t = datetime(2022, 12, 14, 4, 47, tzinfo=timezone.utc)
    lc = l.split_lc(lc, split_t)[1]
    split_t = datetime(2022, 12, 14, 5, 47, tzinfo=timezone.utc)
    lc = l.split_lc(lc, split_t)[0]
    summary_plot(lc, 'AIA Reg 3 171A 2022-12-14', '../../Desktop/poster_figures/aia14.png')
    



if __name__ == '__main__':


'''
This document defines the LightCurve class used to organize and analyze intensity time series data.
Also defines functions used to split LightCurves into 2 subinterval LightCurves.
'''

#Imports
from datetime import datetime
from datetime import timezone
import numpy as np
from scipy import fft
from scipy.signal import resample, find_peaks
from matplotlib import pyplot as plt
from matplotlib import dates
import pycwt

from utility import write_fnames, write_txt_file

from whittaker_smooth import whittaker_smooth as ws

import csv
#whittaker Eilers Smoothing implemented with Martinus Werts's code on Github, by Yuni Kim

class LightCurve:
  '''
  Provides an interface to represent, manipulate, and analyze the intensity time series of a 
  certain frequency of electromagnetic radiation within a certain time range.
  -----------
  Attributes:
  -----------
  timestamps : list of int
    The Unix timestamps of the data.
  power : list of float
    The intensity or power data making up the light curve. 
  units : string
    The units associated with the quantities in power.
  timestep : float
    The average time between data points. (s)
  start : datetime.datetime
    Offset-aware UTC datetime representation of the beginning of the interval described by the
    LightCurve object.
  end : datetime.datetime
    Offset-aware UTC datetime representation of the end of the interval described by the
    LightCurve object.
  source : String
    The instrument that produced the represented data.
  channel : string
    The channel and the power units (for PSP data)
  frequency : float
    The frequency of the represented electromagnetic radiation. (MHz)
  wavelength : int
    In the case of AIA, the wavelength of the represented electromagnetic radiation. (Angstrom)
  region : int
    In the case of AIA, specifies which region from which the data came.
  regname : string
    States more specifically where the data came from (especially useful if it's NOT from AIA).
  type : int
    In the case of HMI, specific which instrument from which the data came.
  ws_lambda : float
    If WE Smoothing, the lambda value is stored here. Otherwise, the default value is zero.
  '''

  def __init__(self, input_time, input_power, input_source, input_regname='', input_region='',
               input_channel='',input_wavelength = 0, input_type=0, input_frequency=0,lam=0.,
               input_dist=180):
    '''
    Creates an instance of the LightCurve class.
    -----------
    Parameters:
    -----------
    input_time : list of int
      The UTC timestamps of the data.
    input_power : list of float
      The power or intensity data of the light curve.
    input_source : String
      The instrument that produced the data.
    input_region : int (optional)
      The region from which the data came.
    input_regname : string (optional)
      A description of the region from which the data came.
    input_channel: string (optional)
      Relevant for certain PSP csv files
    input_wavelength : int (optional)
      If the source is AIA or NuSTAR, wavelength is used to characterize the LightCurve
      instead of frequency. (Angstrom)
    input_type : int (optional)
      The type of data that is being inputted if the source is HMI.
    input_frequency : float (optional)
      The frequency of the data being used to construct a LightCurve. (MHz)
    '''
    self.power = input_power
    self.source = input_source.upper()
    self.region = input_region
    self.regname = input_regname
    self.channel = input_channel   # May be acquired from PSP file, if applicable.
    self.ws_lambda = lam  # stays at 0 if no WESmoothing applied later
    self.min_dist=input_dist
    if input_source[:3] == 'PSP':
      self.units = 'V^2/Hz'
      if (input_channel != ''):
        ch_parts=input_channel.split('(')
        unit_str=ch_parts[1]
        self.units='('+unit_str
        print(self.units)
    elif input_source == 'AIA':
      self.units = 'Detrended DN'
    elif input_source == 'HMI':
      self.units = 'Magnetic Activity'
    elif input_source == 'WIND':
      self.units = 'V'
    elif input_source[:6] == 'STEREO':
      self.units = 'dB above background'
    elif input_source[:6] == 'NUSTAR':
      self.units = 'Count Rate'
    else:
      self.units = 'Unitless'
    #Calculating the timestep
    intervals = [(input_time[ii+1] - t) for ii, t in enumerate(input_time[:-1])]
    if input_source[:6] == 'STEREO':
      #the timestamps for certain files are in ms from 0AD. So that needs to be accounted.
      intervals = [(input_time[i+1] - t)*1e-3 for i, t in enumerate(input_time[:-1])]
      self.timestamps = [t * 1e-3 - 6.2168e+10 for t in input_time]
      self.timestep = np.average(intervals)
      self.start = datetime.fromtimestamp(int(self.timestamps[0]), tz=timezone.utc)
      self.end = datetime.fromtimestamp(int(self.timestamps[-1]), tz=timezone.utc)
      #previous code for STEREO timestamps. 
      '''self.timestep = int(sum(intervals)/len(intervals)*10**(-9))
      time = list((i - np.datetime64('1970-01-01T00:00:00'))/ np.timedelta64(1, 's')
        for i in self.timestamps)
      time_start = time[0]
      time_end = time[-1]
      self.timestamps =  list(datetime.fromtimestamp(i, tz=timezone.utc) for i in time)
      self.timestamps = list(dt.replace(tzinfo=timezone.utc).timestamp() for dt in self.timestamps)
      self.start = datetime.fromtimestamp(time_start, tz=timezone.utc)
      self.end = datetime.fromtimestamp(time_end, tz=timezone.utc)'''
    elif input_source == 'HMI':
      self.timestamps = input_time
      self.timestep = np.average(intervals)
      self.start = datetime.fromtimestamp(self.timestamps[0])
      self.end = datetime.fromtimestamp(self.timestamps[-1])
      self.start = self.start.replace(tzinfo = timezone.utc)
      self.end = self.end.replace(tzinfo =timezone.utc )
    else:
      self.timestamps = input_time
      self.timestep = np.average(intervals)
      self.start = datetime.fromtimestamp(int(self.timestamps[0]), tz=timezone.utc)
      self.end = datetime.fromtimestamp(int(self.timestamps[-1]), tz=timezone.utc)
    #Setting the identifying attributes of the light curve
    if input_frequency != 0:
      self.frequency = input_frequency
      #Generating a placeholder wavelength to be able to convert between the two
      self.wavelength = self.frequency ** -1
    elif input_wavelength != 0:
      #In the case of NuSTAR, 'wavelength' is passed in as a String
      self.wavelength = input_wavelength
      if input_source[:6] == 'NUSTAR':
        self.frequency = 1
      else:
        self.frequency = self.wavelength ** -1
    else:
      print('Please enter either a frequency or wavelength.')
      return
    #adding the type of the HMI data collected
    if input_type!= 0:
      self.type = input_type
    else:
      self.type = 0

  def __str__(self):
    '''
    Produces a String representation of the LightCurve object.
    --------
    Returns:
    --------
    self_str : String
      The String representation.
    '''
    if self.source[0:3] == 'AIA':
      if self.region != '':
        self_str = f'{self.source} Region {self.region} ({self.regname}) '+\
          f'{self.wavelength:.3f} Angstrom'
      else:
        self_str = f'{self.source} {self.regname} {self.wavelength:.3f} Angstrom'
    elif self.source[0:3] == 'PSP':
      self_str = f'{self.source} {self.frequency:.3f} MHz'
    elif self.source[:6] == 'NuSTAR':
      self_str = f'{self.source} {self.region} {self.regname} {self.wavelength}'
    elif self.source[0:3] == 'HMI':
      self_str = f'{self.source} {self.type} Region {self.region} ({self.regname})'+\
        f' {self.wavelength:.3f} Angstrom'
    elif self.source[:6] == 'STEREO':
      self_str = f'{self.source} Region {self.region} ({self.regname}) {self.frequency:.3f} kHz'
    elif self.source[:6] == 'STA' or self.source[:6] == 'STB':
      self_str = f'{self.source} {self.frequency:.3f} MHz'
    else:
      if self.region != '':
        self_str = f'{self.source} Region {self.region} ({self.regname}) {self.frequency:.3f} MHz'
      else:
        self_str = f'{self.source} {self.regname} {self.frequency:.3f} MHz'
    return self_str

  def __add__(self, other):
    '''
    Creates a new LightCurve object with all of the timestamps and associated power
    measurements of self and other. It is necessary that there are no duplicate timestamps
    between the 2 Light Curves, and that they are from the same instrument and frequency
    (because otherwise it makes no sense).
    -----------
    Parameters:
    -----------
    other : LightCurve
      The Light Curve whose measurements are combined with self.
    '''
    if self.source != other.source:
      print('These Light Curve objects can not be combined')
      return
    if self.wavelength != other.wavelength:
      print('These Light Curve objects can not be combined')
      return
    if self.region != other.region:
      print('These Light Curve objects can not be combined')
      return
    times_and_powers = {}
    for ii, timestamp in enumerate(self.timestamps):
      times_and_powers[timestamp] = self.power[ii]
    for ii, timestamp in enumerate(other.timestamps):
      times_and_powers[timestamp] = other.power[ii]
    combined_timestamps = self.timestamps + other.timestamps
    combined_timestamps.sort()
    ordered_powers = []
    for timestamp in combined_timestamps:
      if combined_timestamps.count(timestamp) != 1:
        print('Ensure there are no duplicated times')
        return
      ordered_powers.append(times_and_powers[timestamp])
    lc_comb = LightCurve(combined_timestamps, ordered_powers, self.source,
      input_wavelength=self.wavelength, input_region=self.region,input_regname=self.regname)
    return lc_comb

  def get_time_str(self):
    '''
    Generates a string describing the time range of the LightCurve's data.
    --------
    Returns:
    --------
    time_str : String
      The time string.
    '''
    start_date = self.start.strftime('%Y-%m-%d')
    start_time = self.start.strftime('%H:%M:%S')
    end_date = self.end.strftime('%Y-%m-%d')
    end_time = self.end.strftime('%H:%M:%S')
    if start_date == end_date:
      time_str = start_date + " / " + start_time + " - " + end_time
    else:
      time_str = start_date + "/" + start_time + " - " + end_date + "/" + end_time
    return time_str

  def resample(self, new_timestep):
    '''
    Resamples the data to a new data rate given by new_timestep.
    -----------
    Parameters:
    -----------
    new_timestep : int
      The new timestep for the resampled data. (s)
    '''
    self.timestep = new_timestep
    #Identifying the sample positions at the new data rate
    num_steps = int((self.timestamps[-1]-self.timestamps[0])/self.timestep)
    #Gathering the resampled power and new timestamp locations at the resampled data rate
    self.power, self.timestamps = resample(np.array(self.power, dtype='float64'), num_steps,
                        t=self.timestamps)

  def get_param_msg(self):
    '''
    Generates a string describing the time range of the LightCurve's data.
    --------
    Returns:
    --------
    lam_msg : String
      A string message giving lambda or stating n/a.
    '''
    if self.ws_lambda > 0:
     lam_msg="Lambda={:1.0e}".format(self.ws_lambda)
    else:
      lam_msg="Lambda=N/A"
    param_msg=lam_msg+"; Dist="+str(int(self.min_dist))
    return param_msg

  def get_param_tag(self):
    '''
    Generates a string describing the time range of the LightCurve's data.
    --------
    Returns:
    --------
    param_tag : String
      A string message giving lambda or stating n/a.
    '''
    if self.ws_lambda > 0:
     lam_tag="_lambda{:.0e}".format(self.ws_lambda)
    else:
      lam_tag="_lambda-NA"
    param_tag=lam_tag+"_dist"+str(int(self.min_dist))
    return param_tag


  def detrend(self,lam, smoothing,hmi = 0,psp=0, aia = 0):
    '''
    Detrends the input lightcurve by either the Whittaker function or the boxcar function.
    --------
    Parameters:
    --------
    lam : float
      The lambda that will determine the strength of detrending.
    smoothing : bool
      Boolean that toggles if the Whittaker or boxcar method will be chosen
      (0: boxcar, 1: whittaker)
    hmi, psp, aia : bool
      Boolean that indicates if the lightcurve type is hmi, psp, or aia.
      Changes the data analysis accordingly.
    '''
    hmi = bool(hmi)
    psp = bool(psp)
    aia = bool(aia)
    self.ws_lambda=lam
    if smoothing:
      if hmi: 
        #needed due to the way HMI data formatted from the pickle files 
        self.power = np.array(self.power,dtype =float)[0]
      else:
        self.power = np.array(self.power,dtype =float)
      #Smoothing the power signal using the whittaker-Eilers smoothing function  
      smoothed_power = ws(self.power, lam, d=2)
      #Entire set of data is utilized as no noise occurs at edge.  
      self.power = (self.power - smoothed_power)[:]
    else:
      if hmi:
        self.power = np.array(self.power,dtype =float)[0]
        window = 10
        smoothed_power = np.convolve(self.power, np.ones(window), mode='same') / window
        #The detrended power is the original signal minus the boxcar average
        #One window worth of data on either end is discarded to account for edge effects
        self.power = (self.power - smoothed_power)[window:-window]
        self.timestamps = self.timestamps[window:-window]
      elif psp:
        # NO boxcar smoothing for PSP
        # In which case one must ask why put it through this function at all??
        self.power = np.array(self.power,dtype =float)
        window = 10
      elif aia: 
        #no boxcar smoothing for AIA
        self.power = np.array(self.power,dtype =float)
        window = 10
        smoothed_power = np.convolve(self.power, np.ones(window), mode='same') / window
        #The detrended power is the original signal minus the boxcar average
        #One window worth of data on either end is discarded to account for edge effects
        self.power = (self.power - smoothed_power)[window:-window]
        self.timestamps = self.timestamps[window:-window]
      else:
        #if unspecified, the lightcurve will always be detrended using the boxcar method. 
        self.power = np.array(self.power,dtype =float)
        window = 51
        smoothed_power = np.convolve(self.power, np.ones(window), mode='same') / window
        #The detrended power is the original signal minus the boxcar average
        #One window worth of data on either end is discarded to account for edge effects
        self.power = (self.power - smoothed_power)[window:-window]
        self.timestamps = self.timestamps[window:-window]
      self.start = datetime.fromtimestamp(self.timestamps[0], tz=timezone.utc)
      self.end = datetime.fromtimestamp(self.timestamps[-1], tz=timezone.utc)
    
  def peak_finder(self,smoothing, input_dist,input_height=1,
    show_results=True,save_results=False):
    '''
    Computes the peak times, peak locations, and average periods (s) between the peaks of a
    LightCurve signal. Uses a simple identification of peaks as local maxima.
    -----------
    Parameters:
    -----------
    input_dist : int
      The minimum allowed time in between peaks. (s)
    input_height : float (optional)
      The height multiplier for choosing the minimum allowed height of peaks with relation to
      the mean.
    show_results : Boolean (optional)
      Toggles whether or not to show the text report.
    save_results : Boolean (optional)
      Toggles whether or not to save the text report.
    smoothing:
      Toggles whether or not the WE smoothing is done or not. 
    --------
    Returns:
    --------
    peak_times : list of int
      The timestamps of peak locations in the signal.
    peak_vals : list of float
      The power values of the peak locations for the signal.
    peak_period : float
      The average time between peaks of the signal. (s)
    peak_std : float
      The standard deviation of the time between peaks in the signal. (s)
    '''
    #Finding the indices of the peaks in the power time series
    mean_abs = input_height * np.mean(np.abs(self.power))
    peak_locs = list(find_peaks(self.power, mean_abs, distance=input_dist / self.timestep)[0])
    #Finding the times and power values at the peak indices
    peak_times = np.array(self.timestamps)[peak_locs]
    peak_vals = np.array(self.power)[peak_locs]
    # Calculating the time intervals between adjacent peaks
    peak_differences = [peak_times[i+1] - t for i, t in enumerate(peak_times[:-1])]
    peak_period = np.mean(peak_differences)
    peak_std = np.std(peak_differences)
    #Producing a txt report of the calculated period. Identifying info is in the file path/name
    txt = f"Peak finder period: {peak_period/60:0.1f} +/- {peak_std/60:0.1f} minutes"
    #Saving results if requested
    if save_results:
      txt_fname, _ = write_fnames(self, 'peakFinder',smoothing)
      write_txt_file(txt, txt_fname)
    #Showing results if requested
    if show_results:
      print(txt)
    return smoothing, peak_times, peak_vals, peak_period, peak_std

  def fft_periodicity(self,smoothing, show_results=True, save_results=False,
    period_ax_range=0,save_txt=False):
    '''
    Performs an FFT on the power data to plot a power spectrum and identify the frequencies
    with the most oscillatory power.
    -----------
    Parameters:
    -----------
    show_results : Boolean (optional)
      Specifies whether or not to show the plot and txt output.
    save_results : Boolean (optional)
      Specifies whether or not the power spectrum and txt output are saved.
    period_ax_range : list of 2 floats (optional)
      Sets the range of the period axis. Used to match FFT plots with wavelet plots.
    save_txt : Boolean (optional)
      Toggles whether a text file that summarizes the results are saved. 
    --------
    Returns:
    --------
    Nothing. Plot and txt output can be displayed and/or saved.
    '''
    #Subtracting the mean to remove the 0-frequency component
    balanced_data = self.power - np.mean(self.power)
    #Applying a Hanning window to the data
    window = np.hanning(len(self.power))
    data_windowed = np.multiply(window, balanced_data)
    #Symmetrically zero padding the windowed data to the next fast length
    num_elements = fft.next_fast_len(len(self.power), real=True)
    num_0s_added = num_elements - len(self.power)
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
    freq_bin /= self.timestep #Converting the fft frequencies to meaningful values
    periods = 1 / freq_bin[1:] #seconds
    periods /= 60 #minutes
    #Finding the frequency that corresponds to the highest peak in the power spectrum
    fft_peak_locs = find_peaks(fft_power)[0]
    peak_vals = np.array(fft_power)[fft_peak_locs]
    idx = 0
    #Controlling to avoid the error thrown when there are no peaks present
    try:
      while peak_vals[idx] < np.amax(peak_vals):
        idx += 1
    except IndexError:
      print(f'No peaks found in the FFT power spectrum for {self} '+\
        '({self.get_param_msg()}) {self.get_time_str()}.')
      return
    dom_freq = freq_bin[fft_peak_locs[idx]]
    peak_period = (1 / dom_freq) / 60 #minutes
    #Plotting the frequency spectrum and saving it if requested
    _, axx = plt.subplots(figsize=[4.8, 6.4])
    #We plot the log of the period on the vertical axis and FFT power on the horizontal
    axx.plot(fft_power[1:], np.log2(periods))
    axx.scatter(peak_vals[idx], np.log2((1 / dom_freq) / 60), color = 'r', marker = 'X')
    axx.set_xlabel("FFT Power", fontsize=10)
    axx.set_ylabel("Period (min)", fontsize=10)
    plt.title(str(self) + "; "+self.get_param_msg()+"\n" + self.get_time_str(), fontsize=8)
    plt.ioff()
    plt.suptitle("FFT Power Spectrum", fontsize=12, y=1)
    plt.ioff()
    #Labeling the log-scale period axis nicely
    p_tick_labels = 2 ** np.arange(np.ceil(np.log2(min(periods))),
                    np.ceil(np.log2(max(periods))))
    axx.set_yticks(np.log2(p_tick_labels))
    axx.set_yticklabels([f'{f:.1f}' for f in p_tick_labels])
    #Setting the period axis range to rougly resemble that of wavelet plots
    if period_ax_range != 0:
      axx.set_ylim(period_ax_range)
    else:
      axx.set_ylim(np.log2(min(periods)), np.log2(max(periods)))
    plt.tight_layout()
    #Producing a txt report. All identifying information is in the file name/path
    txt = f"Periodicity with highest power: {peak_period:.3f} minutes"
    #Saving results if requested
    if save_results:
      #write_fnames handles directory creation
      txt_fname, fig_fname = write_fnames(self, 'FFT',smoothing)
      write_txt_file(txt, txt_fname)
      plt.savefig(fig_fname)
    #Showing the results if requested
    if show_results:
      print(txt)
      plt.show()
    if save_txt:
      txt_fname, _ = write_fnames(self, 'FFT',smoothing)
      write_txt_file(txt, txt_fname)
    plt.close()

  def wavelet(self, smoothing,show_results=True, save_results=False, wavelet_type='Morlet',
        bool_flatten=False,save_txt=False):
    '''
    Performs a wavelet transform on the power data. Significance testing on power spectrum
    results against red and white noise backgrounds is done to determine the existence of
    significant periodicities. Plots and text output can be shown and saved as requested.

    Further information regarding wavelet transforms and red noise/white noise detection is 
    elaborated in this link: https://pycwt.readthedocs.io/en/latest/reference/
    -----------
    Parameters:
    -----------
    show_results : Boolean (optional)
      Toggles whether or not to show the figure and text report.
    save_results : Boolean (optional)
      Toggles whether or not to save the figure and text report.
    wavelet_type : String (optional)
      Specifies the mother wavelet to be used. ('morlet' or 'DOG')
    bool_flatten : Boolean (optional)
      Toggles whether all peaks are truncated to the power of the smallest peak.
    save_txt : Boolean (optional)
      Toggles whether a text file that summarizes the results are saved. 
    --------
    Returns:
    --------
    period_ax_range : list of 2 floats
      The minimum and maximum values plotted on the period axis. Returned so that FFT plot
      period axes can be matched with wavelet plots.
    '''
    #Creating lists with the time and power data (mean subtracted from power)
    power = [p - np.mean(self.power) for p in self.power]
    time = self.timestamps
    t_step = int(self.timestep)
    #Finds the power of the smallest peak and sets all larger values to that power if requested
    if bool_flatten:
      peak_indices = find_peaks(power, distance=int(180/t_step), height=max(power) / 20)[0]
      peak_vals = [power[i] for i in peak_indices]
      smallest_peak = min(peak_vals)
      for i, val in enumerate(power):
        if val > smallest_peak:
          power[i] = smallest_peak
    #Normalizing the power data by the standard deviation
    std = np.std(power)
    var = std ** 2
    power_norm = power / std
    #Setting the parameters of our wavelet analysis
    if wavelet_type == 'Morlet':
      mother = pycwt.Morlet(6) #Morlet with w_0 = 6
    elif wavelet_type == 'DOG':
      mother = pycwt.DOG(2) #DOG with m = 2 (Mexican Hat)
    else:
      print("Please choose either 'morlet' or 'DOG' as the mother wavelet")
      return
    #Starting scale set to twice the timestep to detect single-point spikes
    scale_0 = 2 * t_step
    #Specifying frequency resolution at 16 sub-octaves per octave
    freq_step = 1 / 16
    num_octaves = 8
    #Estimating the Lag-1 autocorrelation for red noise modeling
    try:
      alpha = pycwt.ar1(power)[0]
      red_noise_ok=bool(1)
    except Warning:
      print(f'Unable to obtain Lag-1 Autocorrelation for red noise modeling'+\
          f' of the wavelet for {self} ({self.get_param_msg()}) {self.get_time_str()}.\n'+\
          f'The series is too short or the trend is too large.')
      alpha=0.
      red_noise_ok=bool(0)
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
    white_sig = pycwt.significance(var, t_step, scales, 1, 0, significance_level = 0.95,
                    dof=dof, wavelet=mother)[0]
    if red_noise_ok:
      red_sig = pycwt.significance(var, t_step, scales, 1, alpha, significance_level=0.95,
                    dof=dof, wavelet=mother)[0]
    #Creating the figure combining the raw lightcurve, wavelet power spectrum, and global wps
    plt.close('all')
    plt.ioff()
    print("SECOND-DANNY")
    figprops = dict(figsize=(11, 8), dpi=72)
    _ = plt.figure(**figprops, facecolor='white')
    #Plotting the raw lightcurve as the first subplot
    axx = plt.axes([0.1, 0.65, 0.65, 0.3])
    axx.plot(time, power_norm, color='k')
    axx.set_title(f'{self}; {self.get_param_msg()}; {self.get_time_str()}\nLightcurve')
    axx.set_ylabel('Normalized Power')
    #Plotting the normalized wavelet power spectrum with significance level contour
    #lines and the cone of influence marked as the second subplot
    bxx = plt.axes([0.1, 0.1, 0.65, 0.45], sharex=axx)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bxx.contourf(time, np.log2(periods_min), np.log2(wave_power), np.log2(levels),
            extend='both', cmap=plt.cm.inferno)
    extent = [min(time), max(time), 0, np.log2(max(periods_min))]
    bxx.contour(time, np.log2(periods_min), sig95, [-99, 1], colors='k', linewidths=2,
          extent=extent)
    bxx.fill(np.concatenate([time, [time[-1] + t_step], [time[-1] + t_step], [time[1] - t_step],
                  [time[1] - t_step]]),
        np.concatenate([np.log2(coi/60), [1e-9], np.log2(periods_min[-1:]),
                np.log2(periods_min[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
    bxx.set_title(f'{wavelet_type} Wavelet power spectrum')
    bxx.set_xlabel(datetime.fromtimestamp(time[0], tz=timezone.utc).strftime('%Y-%m-%d'))
    bxx.set_ylabel('Period (min)')
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
    #Finding the peaks in the global wavelet power spectrum
    all_peak_indices = [i for i in find_peaks(glbl_power)[0]]
    white_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > white_sig[i]]
    #Plotting the global wavelet power spectrum with its significance levels as the 3rd subplot
    cxx = plt.axes([0.8, 0.1, 0.15, 0.45], sharey = bxx)
    for i in white_peak_indices:
      cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color='#696969')
    line1, = cxx.plot(var * glbl_power, np.log2(periods_min), 'k-')
    line3, = cxx.plot(white_sig, np.log2(periods_min), '--', color='#696969')
    if red_noise_ok:
      red_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > red_sig[i]]
      for i in red_peak_indices:
        cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color = '#FF6666')
      line2, = cxx.plot(red_sig, np.log2(periods_min), '--', color='#FF6666')
      cxx.legend([line1, line2, line3], ['Global Wavelet Power', 'Red Noise', 'White Noise'],
          loc='lower right', fontsize='xx-small')
    else:
      cxx.legend([line1, line3], ['Global Wavelet Power', 'White Noise'],
          loc='lower right', fontsize='xx-small')
    cxx.set_title('Global wavelet power spectrum')
    cxx.set_xlabel('Power')
    #Producing a txt report. All identifying information is in the file name/path
    if wavelet_type == 'Morlet':
      txt = "Morlet wavelet, w_0 = 6"
    else:
      txt = "DOG wavelet, m = 2 (ie Mexican Hat)"
    #Adding the estimated AR1 coefficient to the txt for reference
    if red_noise_ok:
      txt += f"\nEstimated AR1 coefficient: {alpha:0.2f}"
      #Reporting the periods with significant peaks in the global wavelet power spectrum
      txt += "\nGLOBAL PERIODICITIES: Period; Wavelet power/WN power; Peak power/RN power"
    else:
      txt += f"\nEstimated AR1 coefficient: INVALID"
      #Reporting the periods with significant peaks in the global wavelet power spectrum
      txt += "\nGLOBAL PERIODICITIES: Period; Wavelet power/WN power"
    if len(all_peak_indices) == 0:
      txt += '\nNone'
    else:
      for i in all_peak_indices:
        period = periods_min[i]
        wn_ratio = var * glbl_power[i] / white_sig[i]
        if red_noise_ok:
          rn_ratio = var * glbl_power[i] / red_sig[i]
          txt += f"\n{period:0.2f} min; {wn_ratio:0.2f}; {rn_ratio:0.2f}"
        else:
          txt += f"\n{period:0.2f} min; {wn_ratio:0.2f}"
    #Saving results if requested
    if save_results:
      txt_fname, plot_fname = write_fnames(self, f'{wavelet_type}Wavelet',smoothing)
      write_txt_file(txt, txt_fname)
      plt.savefig(plot_fname)
    #Showing results if requested
    if save_txt:
      txt_fname, _= write_fnames(self, f'{wavelet_type}Wavelet',smoothing)
      write_txt_file(txt, txt_fname)
    if show_results:
      print(txt)
      plt.show()
    plt.close()
    return period_ax_range

  def plot_time_series(self,smoothing, show_peaks=False, input_dist=0, input_height=1,
    input_start_time=None, input_end_time=None, show_plot=True,save_plot=False):
    '''
    Plots the time series representation of a LightCurve object. If requested, the peak finder
    can be used to highlight the identified peaks in the plot following specifications that are
    passed in. The plot can be shown and/or saved, as requested.
    -----------
    Parameters:
    -----------
    show_peaks : Boolean (optional)
       Specifies whether or not to show peaks in the final plot.
    input_dist, input_height : See peak_finder DocString
    input_start_time : datetime.datetime (optional)
      Beginning of the time range to show in the plot. Defaults to self.start if not given.
    input_end_time : datetime.datetime (optional)
      End of the time range to show in the plot. Defaults to self.end if not given.
    '''
    # Converting timestamps to Datetime objects for better plotting
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in self.timestamps]
    if show_peaks:
      _ ,peak_times, peak_vals, _, _ = self.peak_finder(smoothing, input_dist, input_height,
                              show_results=False,save_results=False)
      peak_dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in peak_times]
    _, axx = plt.subplots()
    axx.plot_date(dts, self.power, ms=1.5, linestyle='-', lw=0.5, color='b')
    if show_peaks:
      axx.plot_date(peak_dts, peak_vals, ms=3, color='#FFA500')
    if input_start_time is None:
      input_start_time = self.start
    if input_end_time is None:
      input_end_time = self.end
    axx.set_xlim(input_start_time, input_end_time)
    axx.set_xlabel(dts[0].date())
    axx.set_ylabel(self.units)
    #Date formatting for x-axis of plots.
    axx.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    axx.set_title(f'{self}; {self.get_param_msg()}; {self.get_time_str()}\nLightcurve')
    #Saving the plot if requested
    if save_plot:
      _, plot_fname = write_fnames(self, 'timeseries',smoothing)
      plt.savefig(plot_fname)
    #Showing the plot if requested
    if show_plot:
      plt.show()
    plt.close()

  def summary_plot(self,smoothing, bool_show=False, bool_save=False):
    '''
    Generates a standard wavelet plot for the given lightcurve with the following changes:
    - FFT plot in the top right
    - Peak finder result as a line on the global wavelet power spectrum
    - Customized axis label text and sizes

    Further information regarding wavelet transforms and red noise/white noise detection is 
    elaborated in this link: https://pycwt.readthedocs.io/en/latest/reference/
    -----------
    Parameters:
    -----------
    smoothing : Boolean 
      Indicates to the program if the WS smoothing was done to the data set.
    bool_save : Boolean (optional)
      Toggles whether the figure is saved.
    bool_show : Boolean (optional)
      Toggles whether the figure is shown.    
    '''
    #Creating lists with the time and power data (mean subtracted from power)
    power = list(p - np.mean(self.power) for p in self.power)
    time = self.timestamps
    if bool_show:
      print(f"self")
      print(self)
      print(f"self.timestamps")
      print(self.timestamps)
      print(f"time")
      print(time)
    t_step = int(self.timestep)
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
      red_noise_ok=bool(1)
    except Warning:
      print(f'Unable to obtain Lag-1 Autocorrelation for red noise modeling' + \
          f' of the wavelet for {self} ({self.get_param_msg()}) {self.get_time_str()}.\n'+\
          f'Series is too short or trend is too large.')
      alpha=0.
      red_noise_ok=bool(0)
    power_norm = np.transpose(power_norm)

    #Performing the wavelet transform according to the parameters above
    if bool_show:
      print(f"t_step")
      print(t_step)
      print(f"freq_step")
      print(freq_step)
      print(f"scale_0")
      print(scale_0)
      print(f"num_octaves")
      print(num_octaves)
      print(f"mother")
      print(mother)
    wave, scales, freqs, coi, _, _ = pycwt.cwt(power_norm, t_step, freq_step, scale_0,
                          num_octaves/freq_step, mother)
    #Calculating the wavelet and Fourier power spectra
    wave_power = (np.abs(wave)) ** 2
    periods = 1 / freqs
    periods_min = [p / 60 for p in periods]

    # DANNY
    # print(wave_power.shape) # (129, 374)
    # print(coi.shape) # (374,)
    # print(periods.shape) # (129,)
    wave_power_COImask = (np.abs(wave)) ** 2
    coi_matrix = np.tile(coi, (len(periods), 1)) # make an array coi 
    periods_matrix = np.tile(periods[:, np.newaxis], (1, len(coi))) # make array of periods
    mask = periods_matrix >= coi_matrix
    wave_power_COImask[mask] = np.nan


    #Generating significance levels for the wavelet power spectrum
    #The power is significant when wave_power / sig95 > 1
    sig = pycwt.significance(1.0, t_step, scales, 0, alpha, significance_level=0.95,
                  wavelet=mother)[0]
    sig95 = np.ones([1, len(power)]) * sig[:, None]
    sig95 = wave_power / sig95
    #Calculating the global wavelet spectrum and significance levels for red and white noise
    glbl_power = np.nanmean(wave_power,axis=1)
    glbl_power_COImask = np.nanmean(wave_power_COImask,axis=1)
    

    dof = len(power) - scales #Correction for edge effects
    white_sig = pycwt.significance(var, t_step, scales, 1, 0, significance_level = 0.95,
                    dof=dof, wavelet=mother)[0]
    if red_noise_ok:
      red_sig = pycwt.significance(var, t_step, scales, 1, alpha, significance_level=0.95,
                    dof=dof, wavelet=mother)[0]
    #Creating the figure combining the raw lightcurve, wavelet power spectrum, and global wps
    plt.close('all')
    plt.ioff()
    print("FIRST-DANNY")
    figprops = dict(figsize=(15, 8), dpi=72)
    _ = plt.figure(**figprops, facecolor='white')
    #Plotting the raw lightcurve as the first subplot
    axx = plt.axes([0.05, 0.55, 0.5, 0.35])
    axx.plot(time, power_norm, color='k')
    if smoothing:
      if self.source[:3] == 'AIA':
        title_str = self.source +' reg'+ str(self.region)+' '+str(self.wavelength)+'$\AA$ '+\
          self.get_time_str() + ' WS ('+self.get_param_msg()+')'
      elif self.source[:6] == 'STEREO':
        title_str = self.source +' reg'+ str(self.region)+' '+str(self.frequency)+'kHz '+\
          self.get_time_str() + ' WS ('+self.get_param_msg()+')'
      else:
        title_str = str(self) + ' ' + self.get_time_str() + ' WS ('+self.get_param_msg()+')'
    else:
      if self.source[:3] == 'AIA':
        title_str = self.source +' reg'+ str(self.region)+' '+str(self.wavelength)+'$\AA$ '+\
          self.get_time_str()+"; "+self.get_param_msg()
      elif self.source[:6] == 'STEREO':
        title_str = self.source +' reg'+ str(self.region)+' '+str(self.frequency)+'kHz '+\
          self.get_time_str()+"; "+self.get_param_msg()
      else:
        title_str = str(self) + ' ' + self.get_time_str()+"; "+self.get_param_msg()
    axx.set_title(title_str, fontsize=16)
    if self.source =='AIA':
      axx.set_ylabel('Intensity', fontsize=14)
    else:
      axx.set_ylabel('Normalized Power', fontsize=14)
    axx.tick_params(axis='x', which='major', labelsize=12)
    #Putting the mean and standard deviation of the lightcurve in the legend
    line1, = axx.plot([time[0]], [np.max(power_norm)], 'w.', ms=0,
              label=f'Mean: {np.mean(self.power):0.2E} {self.units}')
    line2, = axx.plot([time[0]], [np.max(power_norm)], 'w.', ms=0,
              label=f'St. Dev: {std:0.2E} {self.units}')
    axx.legend()
    #Plotting the normalized wavelet power spectrum with significance level contour
    #lines and the cone of influence marked as the second subplot
    bxx = plt.axes([0.05, 0.05, 0.5, 0.4], sharex=axx)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bxx.contourf(time, np.log2(periods_min), np.log2(wave_power), np.log2(levels),
            extend='both', cmap=plt.cm.inferno)
    extent = [min(time), max(time), 0, np.log2(max(periods_min))]
    bxx.contour(time, np.log2(periods_min), sig95, [-99, 1], colors='k', linewidths=2,
          extent=extent)
    bxx.fill(np.concatenate([time, [time[-1] + t_step], [time[-1] + t_step], [time[1] - t_step],
                  [time[1] - t_step]]),
        np.concatenate([np.log2(coi/60), [1e-9], np.log2(periods_min[-1:]),
                np.log2(periods_min[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
    bxx.set_title('Morlet Wavelet Power Spectrum', fontsize=20)
    bxx.set_ylabel('Period (min)', fontsize=14)
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
    bxx.tick_params(axis='both', which='major', labelsize=12)
    #Finding the peaks in the global wavelet power spectrum
    all_peak_indices = [i for i in find_peaks(glbl_power)[0]]
    white_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > white_sig[i]]
    #Plotting the global wavelet power spectrum with its significance levels as the 3rd subplot
    cxx = plt.axes([0.64, 0.05, 0.15, 0.4], sharey = bxx)
    for i in white_peak_indices:
      cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color='#696969')
    line1, = cxx.plot(var * glbl_power, np.log2(periods_min), 'k-')
    line3, = cxx.plot(white_sig, np.log2(periods_min), '--', color='#696969')
    if red_noise_ok:
      red_peak_indices = [i for i in all_peak_indices if var * glbl_power[i] > red_sig[i]]
      for i in red_peak_indices:
        cxx.scatter(var * glbl_power[i], np.log2(periods_min[i]), marker='o', color = '#FF6666')
      line2, = cxx.plot(red_sig, np.log2(periods_min), '--', color='#FF6666')

    #Finding the peaks in the global wavelet power spectrum
    all_peak_indices_COImask = [i for i in find_peaks(glbl_power_COImask)[0]]
    white_peak_indices_COImask = [i for i in all_peak_indices_COImask if var * glbl_power_COImask[i] > white_sig[i]]
    #Plotting the global wavelet power spectrum with its significance levels as the 3rd subplot
    dxx = plt.axes([0.82, 0.05, 0.15, 0.4], sharey=cxx)
    for i in white_peak_indices_COImask:
      dxx.scatter(var * glbl_power_COImask[i], np.log2(periods_min[i]), marker='o', color='#696969')
    line1_COImask, = dxx.plot(var * glbl_power_COImask, np.log2(periods_min), 'k-')
    line3_COImask, = dxx.plot(white_sig, np.log2(periods_min), '--', color='#696969')
    if red_noise_ok:
      red_peak_indices_COImask = [i for i in all_peak_indices_COImask if var * glbl_power_COImask[i] > red_sig[i]]
      for i in red_peak_indices_COImask:
        dxx.scatter(var * glbl_power_COImask[i], np.log2(periods_min[i]), marker='o', color = '#FF6666')
      line2_COImask, = dxx.plot(red_sig, np.log2(periods_min), '--', color='#FF6666')


    #Adding in the peak finder results
    _,_,_, pf_period, pf_std = self.peak_finder(smoothing,180,show_results=False)
    log_period_min = np.log2(pf_period / 60)
    log_plus = np.log2((pf_period + pf_std) / 60) - log_period_min
    log_minus = log_period_min - np.log2((pf_period - pf_std) / 60)
    cxx.set_xlim([-0.1 * var * max(glbl_power), 1.2 * var * max(glbl_power)])
    dxx.set_xlim([-0.1 * var * max(glbl_power_COImask), 1.2 * var * max(glbl_power_COImask)])
    line4, = cxx.plot(cxx.get_xlim(), [log_period_min, log_period_min], 'b')
    line4_COImask, = dxx.plot(dxx.get_xlim(), [log_period_min, log_period_min], 'b')
    cxx.errorbar(0, log_period_min, [[log_minus], [log_plus]], ecolor='b', capsize=5, markersize=0)
    dxx.errorbar(0, log_period_min, [[log_minus], [log_plus]], ecolor='b', capsize=5, markersize=0)
    if self.source[:3] == 'PSP':
      loc = 'upper left'
    else:
      loc = 'upper right'
    if red_noise_ok:
      cxx.legend([line1, line4, line2, line3], ['Global Wavelet Power', 'Peak Finder Period', 
                          'Red Noise', 'White Noise'],
          loc=loc, prop={'size':6})
      dxx.legend([line1_COImask, line4_COImask, line2_COImask, line3_COImask], ['Global Wavelet Power', 'Peak Finder Period', 
                          'Red Noise', 'White Noise'],
          loc=loc, prop={'size':6})
    else:
      cxx.legend([line1, line4, line3], ['Global Wavelet Power', 'Peak Finder Period',
                          'White Noise'],
          loc=loc, prop={'size':6})
      cxx.legend([line1_COImask, line4_COImask, line3_COImask], ['Global Wavelet Power', 'Peak Finder Period',
                          'White Noise'],
          loc=loc, prop={'size':6})
    cxx.tick_params(axis='y', which='major', labelsize=12)
    dxx.tick_params(axis='y', which='major', labelsize=12)
    #Calculating the FFT
    #Subtracting the mean to remove the 0-frequency component
    balanced_data = self.power - np.mean(self.power)
    #Applying a Hanning window to the data
    window = np.hanning(len(self.power))
    data_windowed = np.multiply(window, balanced_data)
    #Symmetrically zero padding the windowed data to the next fast length
    num_elements = fft.next_fast_len(len(self.power), real=True)
    num_0s_added = num_elements - len(self.power)
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
    freq_bin /= self.timestep #Converting the fft frequencies to meaningful values
    periods = 1 / freq_bin[1:] #seconds
    periods /= 60 #minutes
    #Plotting the FFT spectrum in the top right
    fxx = plt.axes([0.64, 0.55, 0.21, 0.35])
    fxx.plot(fft_power[1:], np.log2(periods))
    fxx.set_title("FFT Power", fontsize=14)
    fxx.set_ylabel("Period (min)", fontsize=14)
    #Labeling the log-scale period axis nicely
    p_tick_labels = 2 ** np.arange(np.ceil(np.log2(min(periods))),
                    np.ceil(np.log2(max(periods))))
    fxx.set_yticks(np.log2(p_tick_labels))
    fxx.set_yticklabels([f'{f:.1f}' for f in p_tick_labels])
    fxx.set_ylim([np.log2(min(periods_min)), np.log2(max(periods_min))])
    fxx.tick_params(axis='y', which='major', labelsize=12)
    #Saving the finished plot
    if bool_save:
      fname = write_fnames(self, 'summary', smoothing)[1]
      plt.savefig(fname, bbox_inches='tight')
    if bool_show:
      plt.show()
    plt.close()

def split_lc(full_lc, split_dt):
  '''
  Splits a LightCurve into 2 LightCurves representing the subintervals before
    and after the given time.
  -----------
  Parameters:
  -----------
  full_lc : LightCurve
    The LightCurve instance to be split.
  split_dt : datetime.Datetime
    The time about which the LightCurve is to be split.
    Assumed to be between the start and end times of full_lc.
  --------
  Returns:
  --------
  (before, after) : tuple of 2 LightCurve
    The 2 LightCurves produced by splitting full_lc.
  '''
  #Saving all the identifying characteristics of full_lc for the creation of new LightCurves
  timestamps = full_lc.timestamps
  power = full_lc.power
  source = full_lc.source
  region = full_lc.region
  #NuSTAR and AIA LightCurves need to be created using the wavelength parameter
  if source[:6] == 'NuSTAR' or source[:3] == 'AIA':
    wavelength = full_lc.wavelength
    frequency = 0
  else:
    frequency = full_lc.frequency
    wavelength = 0
  #Getting the lists of timestamps before and after split_dt
  ts_before = []
  ts_after = []
  for timestamp in timestamps:
    if timestamp < split_dt.timestamp():
      ts_before.append(timestamp)
    else:
      ts_after.append(timestamp)
  #Getting the power before and after split_dt
  power_before = [p for i, p in enumerate(power) if timestamps[i] in ts_before]
  power_after = [p for i, p in enumerate(power) if timestamps[i] in ts_after]
  #Making the before and after LightCurves
  before = LightCurve(ts_before, power_before, source, full_lc.region ,wavelength,
    full_lc.type , frequency)
  after = LightCurve(ts_after, power_after, source, full_lc.region ,wavelength,
    full_lc.type , frequency)
  return before, after

def split_lc_list(lc_list, split_dt):
  '''
  Takes a list of LightCurves and splits them all about the time given.
  -----------
  Parameters:
  -----------
  lc_list : list of LightCurve
    The list of LightCurves to be split.
  split_dt : datetime.Datetime
    The time about which to split the LightCurves.
  --------
  Returns:
  --------
  lc_list_split : list of LightCurve
    All of the subinterval LightCurves.
  '''
  lc_list_split = []
  for lightcurve in lc_list:
    subs = split_lc(lightcurve, split_dt)
    lc_list_split.append(subs[0])
    lc_list_split.append(subs[1])
  return lc_list_split

def summary_txt(self, bool_smoothing):
  return 

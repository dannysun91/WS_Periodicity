#Imports
import os

from datetime import datetime
from datetime import timedelta
from datetime import timezone
import numpy as np
from csv import DictWriter
from scipy import fft
from scipy.signal import peak_widths, find_peaks
import fm_utilities as gf
import pycwt

from lightcurve_class import LightCurve
from utility import read_l2_hres, lc_sort_func

from copy import deepcopy

os.environ['CDF_LIB'] = './lib'

#Constants
SPEED_OF_LIGHT = 3e8 #m/s
METERS_PER_AU = 149597870700

def fft_lc(lc):

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
        #Finding the frequency that corresponds to the highest peak in the power spectrum

    fft_peak_locs = find_peaks(fft_power)[0]
    peak_vals = np.array(fft_power)[fft_peak_locs]
    idx = 0
        #Controlling to avoid the error thrown when there are no peaks present
    try:
        while peak_vals[idx] < np.amax(peak_vals):
            idx += 1
    except IndexError:
        print(f'No peaks found in the FFT power spectrum for {lc} {lc.get_time_str()}.')
        return
    dom_freq = freq_bin[fft_peak_locs[idx]]
    peak_period = (1 / dom_freq) / 60 #minutes
        
    fft_peak_width = peak_widths(fft_power,fft_peak_locs,rel_height = 0.5)
    fft_peak_left = int(fft_peak_width[2][idx])
    fft_peak_right = int(fft_peak_width[3][idx])
    freq_right= (1/freq_bin[fft_peak_right])/60
    freq_left = (1/freq_bin[fft_peak_left])/60
    peak_width = abs(freq_right-freq_left)
    return peak_period, peak_width

def wavelet(self):
    #Creating lists with the time and power data (mean subtracted from power)
    power = [p - np.mean(self.power) for p in self.power]
    length = len(power)
    t_step = self.timestep
    #Normalizing the power data by the standard deviation
    stdv = np.std(power)
    power_norm = power / stdv
    #Setting the parameters of our wavelet analysis
    mother = pycwt.Morlet(6) #Morlet with w_0 = 6
    #Starting scale set to twice the timestep to detect single-point spikes
    scale_0 = 2 * t_step
    #Specifying frequency resolution at 16 sub-octaves per octave
    freq_step = 1 / 16
    num_octaves = 8
    try:
        alpha = pycwt.ar1(power)[0]
    except Warning:
            print(f'Unable to perform the wavelet on {self} {self.get_time_str()}. Series is ' + \
                    'too short or trend is too large.')
            alpha = 0
        #Performing the wavelet transform according to the parameters above
    wave, scales, freqs, _, _, _ = pycwt.cwt(power_norm, t_step, freq_step, scale_0,
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
    wavelet_peak_locs = find_peaks(glbl_power)[0]
    peak_vals = np.array(glbl_power)[wavelet_peak_locs]
    idx = 0
        #Controlling to avoid the error thrown when there are no peaks present
    try:
        while peak_vals[idx] < np.amax(peak_vals):
            idx += 1
    except IndexError:
        print(f'No peaks found in the Wavelet power spectrum for {self} {self.get_time_str()}.')
        return
    wavelet_peak_time = np.array(periods_min)[wavelet_peak_locs] 
    wavelet_time = np.log2(np.linspace(0,max(periods_min),length))
    wavelet_width = peak_widths(glbl_power,wavelet_peak_locs,rel_height = 0.50)
    wavelet_peak_left = int(wavelet_width[2][idx])
    wavelet_peak_right = int(wavelet_width[3][idx])
    peak_width = wavelet_time[wavelet_peak_right] - wavelet_time[wavelet_peak_left]
    '''wavelet_peak_width = peak_widths(glbl_power,wavelet_peak_locs,rel_height = 0.5)
    peak_width_loc = int(wavelet_peak_width[0][idx])
    peak_width = wavelet_time[wavelet_peak_locs[idx]+peak_width_loc] - wavelet_time[wavelet_peak_locs[idx]-peak_width_loc]'''
    peak_period = wavelet_peak_time[idx]

    return peak_period, peak_width

def str_to_dt(dt_str):
    '''
    Creates a UTC datetime object from a specifically formatted String.
    -----------
    Parameters:
    -----------
    dt_str : String
        The string encoding the time information. Formatted as 'YYYY.mm.dd HH:MM' or
        'YYYY.mm.dd HH:MM:SS'
    --------
    Returns:
    --------
    utc_dt : datetime.Datetime
        The offset-aware datetime object created from the string
    '''
    date = dt_str.split()[0]
    date_info = date.split('.')
    year = int(date_info[0])
    month = int(date_info[1])
    day = int(date_info[2])
    time = dt_str.split()[1]
    time_info = time.split(':')
    hour = int(time_info[0])
    minute = int(time_info[1])
    #Allowing for roll-over to the next day with hours above 23
    if hour >= 24:
        bool_next_day = True
    else:
        bool_next_day = False
    if len(time_info) == 3:
        second = int(time_info[2])
    else:
        second = 0
    if bool_next_day:
        utc_dt = datetime(year, month, day, hour-24, minute, second, tzinfo=timezone.utc) + \
                    timedelta(days=1)
    else:
        utc_dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return utc_dt

def write_txt_file(wavelength, other_var,fname,type):
    '''
    Writes the given text to a file given by the input path.
    -----------
    Parameters:
    -----------
    wavelength : list
        list containing all the wavelengths
    other_var : list
        list containing all the peak and peak widths
    fname : string
        string holding the file name for the CSV.
    type : string
        input source
    -----------
    Returns:
    -----------
    arr : np.list
        numpy list containing all the peak widths and variables sorted by wavelength.
    '''
    
    for n, lam in enumerate(wavelength):
        if type == 'AIA':
            
 
            # list of column names
            field_names = [ 'wavelength (AA) ', 'Period of FFT','Width of Period for FFT', 'Period of Wavelet',
                          'Width of Period for Wavelet','Period of FFT w/ WS','Width of Period for FFT w/ WS', 
                          'Period of Wavelet w/ WS','Width of Period for Wavelet w/ WS']

            
            # Dictionary that we want to add as a new row
            dict = {'wavelength (AA) ':lam , 'Period of FFT':other_var[n][0] , 'Width of Period for FFT': other_var[n][1],
            'Period of Wavelet': other_var[n][2], 'Width of Period for Wavelet': other_var[n][3],'Period of FFT w/ WS':other_var[n][4],
            'Width of Period for FFT w/ WS':other_var[n][5],'Period of Wavelet w/ WS':other_var[n][6],'Width of Period for Wavelet w/ WS':other_var[n][7]}

            # Open CSV file in append mode
            # Create a file object for this file
            with open('.\\'+f'{fname}'+'.csv', 'a') as f_object:
            
                # Pass the file object and a list
                # of column names to DictWriter()
                # You will get a object of DictWriter
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(dict)
            
                # Close the file object
        f_object.close()
        if type == 'PSP':
            arr2 = [f'{wavelength[n]} MHz']
            for i,_ in enumerate(other_var):
                arr2.append(other_var[i])
            arr.append(arr1)
        if type == 'PSP':
            arr3 = [f'region {wavelength[n]}']
            for i,_ in enumerate(other_var):
                arr3.append(other_var[i])
            arr.append(arr1)
    
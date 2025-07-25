from os import listdir
import gather_functions as gf
from lightcurve_class import split_lc, LightCurve

from cdasws import CdasWs
import cdflib
import numpy as np
import csv


from datetime import datetime
from datetime import timezone



def AIA_gathering(split,split_time):
    fname_list = listdir('./DATA/AIA/') #input must be a file in the AIA folder. Can be edited. 
    lc_list = []
    lc_listWS = []
    
    for lc in fname_list:
        with open("./DATA/AIA/" + lc, 'r', newline='', encoding='utf8') as file:
            reader = csv.reader(file)
            data = list(reader)
            data = np.transpose(data)
            timestamps = [int(time.split('.')[0]) for time in data[0]]
            power = [int(val.split('.')[0]) for val in data[1]]
            #Getting the wavelength (in A) from the file name
            fname_parts = lc.split('_')
            wavelength = int(fname_parts[-3])
            #Creating a LightCurve object from the timestamps and detrended power values
            lc_aia = LightCurve(timestamps, power, "AIA",input_region=lc[-5], input_wavelength=wavelength)
            lc_aiaws = LightCurve(timestamps, power, "AIA",input_region=lc[-5], input_wavelength=wavelength)
            lc_aia.detrend(1e3,bool(0),0,0,1)
            lc_aiaws.detrend(1e3,bool(1),0,0,1)
            lc_list.append(lc_aia)
            lc_listWS.append(lc_aiaws)
   

    '''lc_list_new = []
    if bool(split):
        split_time = gf.str_to_dt(f'{date} {split_time}')
        for lc in lc_list:
            lc_list_split = split_lc(lc, split_time)
            lc_list_new.append(lc_list_split[0])

        lc_list = lc_list_new    '''
       
    return lc_list, lc_listWS

def HMI_gathering(split,split_time):
    fname_list = listdir('./DATA/HMI/data') #input must be a file in the hmi folder. Can be edited. 
    lc_listWS = []
    lc_list = []
    
    for file in fname_list:
        lc_list.append(gf.gather_hmi_fname(file, bool(0)))
        lc_listWS.append(gf.gather_hmi_fname(file, bool(1)))
    
    curve = gf.gather_hmi_fname(fname_list[0], bool(0))
    date = curve.start
    date = date.strftime('%Y.%m.%d')
   
    lc_list_new = []
    if bool(split):
        split_time = gf.str_to_dt(f'{date} {split_time}')
        for lc in lc_list:
            lc_list_split = split_lc(lc, split_time)
            lc_list_new.append(lc_list_split[0])

        lc_list = lc_list_new   
       
    return lc_list, lc_listWS

def PSP_gathering(split,split_time, date,start_time, end_time):
    #you must manually choose which file you would like to analyze
    file_location= "./DATA/PSP/"
    power = []
    timestamps = []
    lc_list = []
    lc_listWS = []
    list1 = []
    freq_range = 'hfr'

    

    year = date.split('.')[0]
    month_day = date.split('.')[1]
    month = month_day.split(':')[0]
    day = month_day.split(':')[1]
    s_hour,s_min = start_time(':')[:1]
    e_hour,e_min = end_time(':')[:1]

    start_time = datetime(int(year),int(month),int(day),int(s_hour),int(s_min),0, tzinfo=timezone.utc)
    end_time = datetime(int(year),int(month),int(day),int(e_hour),int(e_min),0, tzinfo=timezone.utc)

    cdf_file = cdflib.CDF(file_location + f"psp_fld_l3_rfs_hfr_"+year+month+day+"_v02.cdf")
    #Gathering relevant variables from the CDF file
    v12_data = cdf_file.varget(f'psp_fld_l3_rfs_{freq_range}_auto_averages_ch0_V1V2')
    v34_data = cdf_file.varget(f'psp_fld_l3_rfs_{freq_range}_auto_averages_ch1_V3V4')
    freqs = cdf_file.varget(f'frequency_{freq_range}_auto_averages_ch1_V3V4')[0][32:]
    for i, _ in enumerate(v12_data):
        vector_sum_magnitude = np.sqrt((v12_data[i] ** 2) + (v34_data[i] ** 2))
        power.append(vector_sum_magnitude)
    
    epoch_times = cdf_file.varget('epoch_' + freq_range)
    offset = datetime(2000,1,1,11,58,55,816000, tzinfo=timezone.utc).timestamp() #in seconds
    for epoch in epoch_times:
        timestamps.append((epoch/1e9) + offset)

    dts = [datetime.fromtimestamp(int(timestamp), tz=timezone.utc) for timestamp in timestamps]
    constricted_dts = [dt for dt in dts if (dt > start_time and dt < end_time)]
    constricted_timestamps = [dt.timestamp() for dt in constricted_dts]
    #The power must be accordingly restricted to match up with the timestamps correctly.
    constricted_power = np.array([power[idx] for idx, dt in enumerate(dts) \
                                        if (dt > start_time and dt < end_time)])
    
   #the lightcurve is created three times as to not repeat the detrending. 
    for idx, freq in enumerate(freqs):
        for freq_val in freq_list:
            if round(freq*1e-6) == freq_val:
                lc = LightCurve(constricted_timestamps,constricted_power[:,idx],'PSPComb',0,0,0,
                                    input_frequency=freq * 1e-6)
                lc.detrend(1000,1,0,1)
                lc_listWS.append(lc)
    
    for idx, freq in enumerate(freqs):
        lc = LightCurve(constricted_timestamps,constricted_power[:,idx],'PSPComb',0,0,0,
                            input_frequency=freq * 1e-6)
        lc.detrend(1000,0,0,1)
        lc_list.append(lc)
        
    for idx, freq in enumerate(freqs):
        lc = LightCurve(constricted_timestamps,constricted_power[:,idx],'PSPComb',0,0,0,
                            input_frequency=freq * 1e-6)
        list1.append(lc)

    curve = lc_list[0]

    return lc_list, lc_listWS, list1, curve

def STEREO_gathering(split,split_time):
    power = []
    timestamps = []
    lc_list = []
    lc_listWS = []
    list1 = []

    start_time = datetime(2019,4,12,17,0,0, tzinfo=timezone.utc)
    end_time = datetime(2019,4,12,19,0,0, tzinfo=timezone.utc)

    cdas = CdasWs()
    lc_list = []
    #Getting the data from cdaweb
    status, data = cdas.get_data('STEREO_LEVEL2_SWAVES', ['avg_intens_ahead'], start_time,
                                    end_time)
    
    

    return lc_list, lc_listWS, list1, curv
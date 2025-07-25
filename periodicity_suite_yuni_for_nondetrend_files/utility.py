'''
This document defines utility functions used for various purposes.
'''

#Imports
import struct
from os import mkdir
from os.path import exists
from datetime import datetime
from datetime import timezone
from datetime import timedelta

def find_range_above(val, ref, idx):
    '''
    Wavelet helper function for finding uncertainty bounds on significant periods in the global
    spectrum. Starts at a peak value in val (index=idx) and finds the range of indices around idx
    for which val > ref.
    -----------
    Parameters:
    -----------
    val : list of float
        The dataset with the peak we are interested in finding the bounds of.
    ref : list of float
        The reference values that val is compared to.
    idx : int
        The index of a known peak in val for which it is known that val[idx] > ref[idx].
    --------
    Returns:
    --------
    low : int
        The lowest index for which val > ref in the range including idx.
    high : int
        The highest index for which val > ref in the range including idx.
    '''
    low = 0
    high = -1
    #Finding the upper bound
    for i in range(len(val[idx:])):
        if val[idx + i] <= ref[idx + i]:
            high = idx + i - 1
            break
    #Finding the lower bound
    for i in range(len(val[:idx + 1])):
        if val[idx - i] <= ref[idx - i]:
            low = idx - i + 1
            break
    return low, high

def read_l2_hres(filepath):
    '''
    Unpacks a Wind/Waves l2 high resolution data file.
    -----------
    Parameters:
    -----------
    filepath : String
        Relative path to the Wind Waves L2 high resolution binary file.
    --------
    Returns:
    --------
    header : list of dictionaries
        The header info from the file.
    data : list of dictionaries
        The data from the file.
    '''
    header_fields = ('P_FIELD', 'JULIAN_DAY_B1', 'JULIAN_DAY_B2', 'JULIAN_DAY_B3', 'MSEC_OF_DAY',
                    'RECEIVER_CODE', 'JULIAN_SEC_FRAC', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE',
                    'SECOND', 'JULIAN_SEC_FRAC', 'ISWEEP', 'IUNIT', 'NPBS', 'SUN_ANGLE',
                    'SPIN_RATE', 'KSPIN', 'MODE', 'LISTFR', 'NFREQ', 'ICAL', 'IANTEN', 'IPOLA',
                    'IDIPXY', 'SDURCY', 'SDURPA', 'NPALCY', 'NFRPAL', 'NPALIF', 'NSPALF', 'NZPALF')
    header_dtype = '>bbbbihLhhhhhhfihhffhhhhhhhhffhhhhh'
    header = []
    data = []
    nsweep = 1
    nsample = 0
    with open(filepath, 'rb') as frb:
        while True:
            try:
                # Reading number of octets in the current sweep
                block = frb.read(4)
                if len(block) == 0:
                    break
                loctets1 = struct.unpack('>i', block)[0]
                # Reading header parameters in the current sweep
                block = frb.read(80)
                header_i = dict(
                    zip(header_fields, struct.unpack(header_dtype, block)))
                npalf = header_i['NPALIF']
                nspal = header_i['NSPALF']
                nzpal = header_i['NZPALF']
                # Reading frequency list (kHz) in the current sweep
                block = frb.read(4 * npalf)
                ## > denotes big endian byte order, f denotes float data type
                ## multiplication of 'f' by npalf signifies the number of data
                ## points to be unpacked
                freq = struct.unpack('>' + 'f' * npalf, block)
                # Reading intensity and time values for S/SP in the current
                # sweep
                block = frb.read(4 * npalf * nspal)
                vspal = struct.unpack('>' + 'f' * npalf * nspal, block)
                block = frb.read(4 * npalf * nspal)
                tspal = struct.unpack('>' + 'f' * npalf * nspal, block)
                # Reading intensity and time values for Z in the current sweep
                block = frb.read(4 * npalf * nzpal)
                vzpal = struct.unpack('>' + 'f' * npalf * nzpal, block)
                block = frb.read(4 * npalf * nzpal)
                tzpal = struct.unpack('>' + 'f' * npalf * nzpal, block)
                # Reading number of octets in the current sweep
                block = frb.read(4)
                loctets2 = struct.unpack('>i', block)[0]
                if loctets2 != loctets1:
                    print('Error reading file!')
                    return None
            except EOFError:
                print('End of file reached')
                break
            else:
                header.append(header_i)
                data.append({'FREQ': freq, 'VSPAL': vspal, 'VZPAL': vzpal, 'TSPAL': tspal,
                                'TZPAL': tzpal})
                nsweep += 1
                nsample += len(freq)
    return header, data

def lc_sort_func(lightcurve):
    '''
    Generates a sort key for LightCurve objects based on their source.
    -----------
    Parameters:
    -----------
    lc : LightCurve
        The LightCurve we aim to sort in a list
    '''
    if lightcurve.source == 'AIA':
        return lightcurve.wavelength
    else:
        return lightcurve.frequency

def make_dir(dir_name):
    '''
    Creates a directory, if it does not already exist, according to the path given in the input
    String. If necessary, parent directories will be created first via recursion.
    -----------
    Parameters:
    -----------
    dir_name : String
        The relative paths to the directory we wish to make.
    '''
    #Getting rid of the slash on the end, if there is one (it ruins the recursion)
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    if not exists(dir_name):
        #Trying to just make the directory
        try:
            #This will throw a FileNotFoundError if directories above do not exist
            mkdir(dir_name)
        except FileNotFoundError:
            #Getting the path to the directory above the one given by dir_name
            all_dirs = dir_name.split('/')
            dir_directly_above = ''
            for parent_dir in all_dirs[:-1]:
                dir_directly_above += parent_dir
                dir_directly_above += '/'
            #Recursive step
            make_dir(dir_directly_above)
            #Actually making the desired directory once all necessary parent directories are made
            mkdir(dir_name)

def write_fnames(lightcurve, method,smoothing):
    '''
    Creates the directory and file names for the results of a given method being performed on a
    given LightCurve object.
    -----------
    Parameters:
    -----------
    lightcurve : LightCurve
        The specific LightCurve for which results are being produced.
    method : String
        Specifies which periodicity analysis method was performed so that files can be accurately
        named.
    --------
    Returns:
    --------
    txt_fname : String
        The full relative path to the text output file of the given method being run on the given
        LightCurve.
    plot_fname : String
        The full relative path to the plot made by the given method being run on the given
        LightCurve.
    '''
    #Getting the identifying characteristics of the LightCurve data
    source = lightcurve.source.upper()
    wavelength = lightcurve.wavelength
    frequency = lightcurve.frequency
    region = lightcurve.region
    regname=lightcurve.regname
#    print('Region',region,'   Regname',regname)
    #Getting the time information for filenames
    start = lightcurve.start.strftime('%Y%m%d_%H%M')
    end = lightcurve.end.strftime('%Y%m%d_%H%M')
#    #Getting the date information for making directories
#    date = lightcurve.start.strftime('%Y%m%d')
    #Making directories
#    txt_dir_name = f"results/txt/{date}/"
#    plots_dir_name = f"results/plots/{date}/"
    txt_dir_name = f"results/txt/"
    plots_dir_name = f"results/plots/"
    if lightcurve.type != 0:
        type_HMI = lightcurve.type
    #We don't need a txt directory for timeseries or summary plots
    if method != 'timeseries' and method != 'summary':
        make_dir(txt_dir_name)
    #We don't need a plots directory for the peak finder
    if method != 'peakFinder':
        make_dir(plots_dir_name)
    #Creating the txt filename
    if source == 'AIA':
        if smoothing == True:
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}A_reg{region}_{regname}"+\
              f"_{method}Results_WS{lightcurve.get_param_tag()}.txt"
        else: 
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}A_reg{region}_{regname}"+\
              f"_{method}Results{lightcurve.get_param_tag()}.txt"
    elif source[:6] == 'NuSTAR':
        if smoothing == True:
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}_reg{region}_{regname}"+\
              f"_{method}Results_WS{lightcurve.get_param_tag()}.txt"
        else:
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}_reg{region}_{regname}"+\
              f"_{method}Results{lightcurve.get_param_tag()}.txt"
    elif source[:6] == 'STEREO':
        if smoothing == True:
            txt_fname = f"{source}_{start}_to_{end}_{frequency:.3f}kHz_reg{region}_{regname}"+\
              f"_{method}Results_WS{lightcurve.get_param_tag()}.txt"
        else:
            txt_fname = f"{source}_{start}_to_{end}_{frequency:.3f}kHz_reg{region}_{regname}"+\
              f"_{method}Results{lightcurve.get_param_tag()}.txt"
    elif source == 'HMI':
        if smoothing == True:
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}MHz_reg{region}_{regname}"+\
              f"_{method}Results_{type_HMI}_WS{lightcurve.get_param_tag()}.txt"
        else: 
            txt_fname = f"{source}_{start}_to_{end}_{wavelength}MHz_reg{region}_{regname}"+\
              f"_{method}Results_{type_HMI}{lightcurve.get_param_tag()}.txt"
    else:
        if smoothing == True:
            txt_fname = f"{source}_{start}_to_{end}_{frequency:.3f}MHz_{regname}"+\
              f"_{method}results_WS{lightcurve.get_param_tag()}.txt"
        else:
            txt_fname = f"{source}_{start}_to_{end}_{frequency:.3f}MHz_{regname}"+\
              f"_{method}results{lightcurve.get_param_tag()}.txt"
    #Creating the plot filename
    if source == 'AIA':
        if smoothing == True:
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}A_reg{region}_{regname}"+\
              f"_{method}Plot_WS{lightcurve.get_param_tag()}.png"
        else:  
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}A_reg{region}_{regname}"+\
              f"_{method}Plot{lightcurve.get_param_tag()}.png"
    elif source[:6] == 'NuSTAR':
        if smoothing == True:
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}_reg{region}_{regname}"+\
              f"_{method}Plot_WS{lightcurve.get_param_tag()}.png"
        else:
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}_reg{region}_{regname}"+\
              f"_{method}Plot{lightcurve.get_param_tag()}.png"
    elif source == 'HMI':
        if smoothing == True:
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}MHz_reg{region}_{regname}"+\
              f"_{method}Plot_{type_HMI}_WS{lightcurve.get_param_tag()}.png"
        else: 
            plot_fname = f"{source}_{start}_to_{end}_{wavelength}MHz_reg{region}_{regname}"+\
              f"_{method}Plot_{type_HMI}{lightcurve.get_param_tag()}.png"
    elif source[:6] == 'STEREO':
        if smoothing == True:
            plot_fname = f"{source}_{start}_to_{end}_{frequency:.3f}kHz_reg{region}_{regname}"+\
              f"_{method}Plot_WS{lightcurve.get_param_tag()}.png"
        else:
            plot_fname = f"{source}_{start}_to_{end}_{frequency:.3f}kHz_reg{region}_{regname}"+\
              f"_{method}Plot{lightcurve.get_param_tag()}.png"
    else:
       if smoothing == True:
            plot_fname = f"{source}_{start}_to_{end}_{frequency:.3f}MHz_{regname}"+\
              f"_{method}Plot_WS{lightcurve.get_param_tag()}.png"
       else:
            plot_fname = f"{source}_{start}_to_{end}_{frequency:.3f}MHz_{regname}"+\
              f"_{method}Plot{lightcurve.get_param_tag()}.png"
    #Combining the filenames with their directory paths to return the full path
    full_txt_fname = txt_dir_name + txt_fname
    full_plot_fname = plots_dir_name + plot_fname
    return full_txt_fname, full_plot_fname

def write_txt_file(txt, txt_fname):
    '''
    Writes the given text to a file given by the input path.
    -----------
    Parameters:
    -----------
    txt : String
        The text to be written to a file.
    txt_fname : String
        The relative path to the file to which we want to write the text.
    '''
    with open(txt_fname, 'w', encoding='utf8') as file:
        file.write(txt)

def make_dts_from_input(date, time_1, time_2):
    '''
    Makes two datetime objects according to the input Strings. Both have the same date, with times
    given by 'time_1' and 'time_2'. The output datetimes are offset-aware, in the UTC timezone.
    Capable of handling roll-over from one day to the next.
    -----------
    Parameters:
    -----------
    date : String
        The date for which we wish to create datetime objects. Formatted YYYY.mm.dd
    time_1 : String
        The time for the first datetime object. Formated HH:MM or HH:MM:SS
    time_2 : String
        The time for the second datetime object. Formated HH:MM or HH:MM:SS
    --------
    Returns:
    --------
    dts : list of 2 datetime.Datetime
        The two datetime objects created from the input. time_1 is first, time_2 is second.
    '''
    #Getting the date information
    date_info = date.split('.')
    year = int(date_info[0])
    month = int(date_info[1])
    day = int(date_info[2])
    #Getting the time_1 information
    time_1_info = time_1.split(':')
    hour_1 = int(time_1_info[0])
    #If the hour input is greater than or equal to 24, we roll over to the next day
    next_day_1 = False
    if hour_1 >= 24:
        hour_1 -= 24
        next_day_1 = True
    minute_1 = int(time_1_info[1])
    #Checking if seconds have been given in time_1 or not, and acting accordingly
    if len(time_1_info) == 3:
        second_1 = int(time_1_info[2])
    else:
        second_1 = 0
    #Getting the time_2 information
    time_2_info = time_2.split(':')
    hour_2 = int(time_2_info[0])
    next_day_2 = False
    if hour_2 >= 24:
        hour_2 -= 24
        next_day_2 = True
    minute_2 = int(time_2_info[1])
    #Checking if seconds have been given in time_2 or not, and acting accordingly
    if len(time_2_info) == 3:
        second_2 = int(time_2_info[2])
    else:
        second_2 = 0
    #Creating the datetime objects and storing them
    dts = []
    try:
        if next_day_1:
            dt_1 = datetime(year, month, day, hour_1, minute_1, second_1, tzinfo=timezone.utc) + \
                    timedelta(days=1)
        else:
            dt_1 = datetime(year, month, day, hour_1, minute_1, second_1, tzinfo=timezone.utc)
        if next_day_2:
            dt_2 = datetime(year, month, day, hour_2, minute_2, second_2, tzinfo=timezone.utc) + \
                    timedelta(days=1)
        else:
            dt_2 = datetime(year, month, day, hour_2, minute_2, second_2, tzinfo=timezone.utc)
        dts.append(dt_1)
        dts.append(dt_2)
    except ValueError:
        print('Make sure you are entering appropriate date/time value in a valid format. Exitting')
        return
    return dts

def get_lc_at_freq(lc_list, freq):
    '''
    Looks through a list of LightCurves and returns the LightCurve instance at or directly above
    the input frequency.
    -----------
    Parameters:
    -----------
    lc_list : list of LightCurve
        The list of LightCurves we search through. All from the same source over the same interval.
    freq : float
        The approximate frequency we are looking for.
    '''
    #Sorting the list just in case is hasn't been already
    lc_list.sort(key=lc_sort_func)
    #Looping through the list to get the index of the correct LightCuve
    idx = 0
    while lc_list[idx].frequency < freq:
        idx += 1
    return lc_list[idx]

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

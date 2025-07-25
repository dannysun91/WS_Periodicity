'''
This documents defines the methods used the control the flow of information given to the main
function, based on the source we wish to analyze.
'''

#
import os
import gather_functions as gf
from utility import make_dts_from_input, get_lc_at_freq, str_to_dt
from lightcurve_class import split_lc_list
from datetime import date

def prompt_time():
  '''
  Asks questions about the start and end times of intervals for creating LightCurves. Capable of
  creating multiple intervals on the same date and handling roll-over from one day to the next.
  Also, if only one interval is give, the user is prompted with the option to split that interval
  into 2 subintervals about a time in the middle.
  --------
  Returns:
  --------
  intervals : list of [datetime.Datetime, datetime.Datetime] lists
    The datetime objects describing the start and end times of each interval given.
  split_info : list
    Contains a Boolean toggling whether or not splitting is to be performed, and the time about
    which the splitting is to happen (if necessary).
  '''
  date = input('Enter date as YYYY.mm.dd\n')
  intervals = []
  move_on = ''
  while move_on != 'done':
    start = input('Enter start time (UTC) as HH:MM or HH:MM:SS\n')
    end = input('Enter end time (UTC) as HH:MM or HH:MM:SS\n')
    intervals.append(make_dts_from_input(date, start, end))
    move_on = input("Press Enter to give another time interval, or enter 'done' to move on\n")
  #Offering the option to split an interval into subintervals, if only one is given
  if len(intervals) == 1:
    bool_split = bool(int(input('Would you like to split this interval into 2 ' + \
                                'subintervals? Enter 0 for no, 1 for yes\n')))
    split_info = [bool_split]
    if bool_split:
      split_str = input('Enter the time about which to split the interval as HH:MM or ' + \
                        'HH:MM:SS\n')
      split_dt = str_to_dt(f'{date} {split_str}')
      if split_dt <= intervals[0][0] or split_dt >= intervals[0][1]:
        print('Enter a time that is between the inteval start and end times. Exitting.')
        return
      split_info.append(split_dt)
    else:
      split_info = [False]
  return intervals, split_info

def prompt_split(lc_list):
  '''
  Asks questions about whether the light curve appended would like to be divded into sub-intervals.
  --------
  Returns:
  --------
  split_info : list
    Contains a Boolean toggling whether or not splitting is to be performed, and the time about
    which the splitting is to happen (if necessary).
  '''
  intervals=[lc_list.start,lc_list.end]
  date = intervals[0].strftime('%Y.%m.%d')
  bool_split = bool(int(input('Would you like to split this interval into 2 ' + \
                              'subintervals? Enter 0 for no, 1 for yes\n')))
  split_info = [bool_split]
  if bool_split:
    split_str = input('Enter the time about which to split the interval as HH:MM or ' + \
                      'HH:MM:SS\n')
    split_dt = str_to_dt(f'{date} {split_str}')
    if split_dt <= intervals[0] or split_dt >= intervals[1]:
      print('Enter a time that is between the inteval start and end times. Exitting.')
      return
    split_info.append(split_dt)
  else:
    split_info = [False]
  return split_info

def prompt_psp(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on PSP data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  all_lcs = []
  intervals, split_info = prompt_time()
  variable = input("Which PSP variable would you like? Enter 'V1V2', 'V3V4', or 'combination'" + \
                   "\n").upper()
  method = int(input('Enter 0 to gather data from cdaweb, or 1 to gather from a local file\n'))
  #The following questions depend on which method is to be performed
  if method == 0:
    auto = input("Would you like averages or peaks data? Enter 'averages' or 'peaks'\n")
    bool_shift = bool(int(input('Would you like to adjust the times to account for the ' + \
                                'Earth-PSP distance? Enter 0 for no, 1 for yes\n')))
    for interval in intervals:
      all_lcs.append(gf.gather_psp(interval[0], interval[1], variable, auto, bool_shift,bool_smoothing))
  elif method == 1:
    v_str = input("What number appears after v in the file name? Enter as '01', '02', etc.\n")
    for interval in intervals:
      try:
        all_lcs.append(gf.gather_psp_local_files(interval[0], interval[1], v_str, variable,bool_smoothing))
      except FileNotFoundError:
        print(f"No v{v_str} PSP data file found for the " + \
              f"interval beginning at {interval[0].strftime('%Y.%m.%d %H:%M')}")
        continue
  else:
    print('Enter a 0 or 1. Exitting.')
    return
  #Picking out LightCurves at the specified frequencies
  freqs = []
  input_freq = ''
  while input_freq != 'done':
    input_freq = input("Enter frequencies (in MHz) one at a time, or 'done' to move on\n")
    if input_freq != 'done':
      freqs.append(float(input_freq))
  lc_list = []
  #LightCurves at the given frequencies are returned for all time intervals specified earlier
  for lc_group in all_lcs:
    for freq in freqs:
      lc_list.append(get_lc_at_freq(lc_group, freq))
  #Splitting the LightCurves into subintervals if requested
  if split_info[0]:
    lc_list_split = split_lc_list(lc_list, split_info[1])
    lc_list = lc_list_split
  return lc_list

def prompt_aia(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on AIA data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  intervals, split_info = prompt_time()
  #Getting the channel(s) to be analyzed
  channels = []
  channel = ''
  while channel != 'done':
    channel = input("Enter wavelengths in A one at a time, or 'done' to move on\n")
    if channel != 'done':
      channels.append(int(channel))
  #Getting the regions(s) to be analyzed
  regions = []
  region = ''
  while region != 'done':
    region = input("Enter region numbers one at a time, or 'done' to move on\n")
    if region != 'done':
      regions.append(int(region))
  #Making LightCurves for all regions at all channels for each interval
  lc_list = []
  for interval in intervals:
    for channel in channels:
      for region in regions:
        try:
          lc_list.append(gf.gather_aia(interval[0], interval[1], channel, region,bool_smoothing))
        except FileNotFoundError:
          print(f"No AIA data file found for the {channel} A, region {region} " + \
                f"interval beginning at {interval[0].strftime('%Y.%m.%d %H:%M')}")
          continue
  #Splitting the LightCurves into subintervals if requested
  if split_info[0]:
    lc_list_split = split_lc_list(lc_list, split_info[1])
    lc_list = lc_list_split
  return lc_list

def prompt_HMI(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on AIA data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  dir = bool(int(input('Would you like to list the directory of the data file? Enter 0 for no, 1 for yes\n')))
  fname_list = []
  fname = ''
  lc_list = []
  if dir:
    while fname != 'done':
      fname = input("Enter a path for the files, or 'done' to move on\n")
      if fname != 'done':
        fname_list = os.listdir(fname)
    for file in fname_list:
      lc_list.append(gf.gather_hmi_fname(file,bool_smoothing))
  else:
    while fname != 'done':
      fname = input("Enter a filename (path not necessary), or 'done' to move on\n")
      if fname != 'done':
        fname_list.append(fname)
    for file in fname_list:
      lc_list.append(gf.gather_hmi_fname(file,bool_smoothing))
  return lc_list
    
def prompt_nustar(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on NuSTAR data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  fname = bool(int(input('Would you like to input filenames to skip questions? Enter 0 ' + \
                         'for no, 1 for yes\n')))
  if fname:
    fname_list = []
    while fname != 'done':
      fname = input("Enter a filename (path not necessary), or 'done' to move on\n")
      if fname != 'done':
        fname_list.append(fname)
    lc_list = []
    for file in fname_list:
      lc_list.append(gf.gather_nustar_fname(file,bool_smoothing))
    split_info = prompt_split(lc_list[0])
  else:
    intervals, split_info = prompt_time()
    fpm = input("Which detector? Enter 'A', 'B', or 'both'\n")
    if fpm == 'both':
      fpms = ['A', 'B']
    elif fpm == 'A':
      fpms = ['A']
    elif fpm == 'B':
      fpms = ['B']
    else:
      print("Please answer 'A', 'B', or 'both'. Exitting")
      return
    #Getting the regions(s) to be analyzed
    regions = []
    region = ''
    while region != 'done':
      region = input("Enter region number, or 'done' to move on\n")
      if region != 'done':
        regions.append(int(region))
    lc_list = []
    for interval in intervals:
      for fpm_val in fpms:
        for region in regions:
          try:
            new_lc = gf.gather_nustar(interval[0], interval[1], fpm_val, region)
            new_lc.detrend(51)
            lc_list.append(new_lc)
          except FileNotFoundError:
            print(f"No NuSTAR data file found for the FPM {fpm_val}, region {region} " + \
                  f"interval beginning at {interval[0].strftime('%Y.%m.%d %H:%M')}")
            continue
  #Splitting the LightCurves into subintervals if requested
  if split_info[0]:
    lc_list_split = split_lc_list(lc_list, split_info[1])
    lc_list = lc_list_split
  return lc_list

def prompt_wind(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on WIND data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  intervals, split_info = prompt_time()
  #Picking out LightCurves at the desired frequencies
  freqs = []
  input_freq = ''
  while input_freq != 'done':
    input_freq = input("Enter frequencies (in MHz) one at a time, or 'done' to move on\n")
    if input_freq != 'done':
      freqs.append(float(input_freq))
  lc_list = []
  for interval in intervals:
    lc_group = gf.gather_wind(interval[0], interval[1])
    for freq in freqs:
      lc_list.append(get_lc_at_freq(lc_group, freq))
  #Splitting the LightCurves into subintervals if requested
  new_lc = []
  if bool_smoothing:
    for lc in lc_list:
      lc.detrend(1000,bool_smoothing)
      new_lc.append(lc)
    lc_list = new_lc
  if split_info[0]:
    lc_list_split = split_lc_list(lc_list, split_info[1])
    lc_list = lc_list_split
  return lc_list

def prompt_stereo(bool_smoothing):
  '''
  Asks the flow of questions necessary to run routines on STEREO data, and acquires the requested
  LightCurves.
  --------
  Returns:
  --------
  lc_list : list of LightCurve
    A list containing the LightCurves specified by the user's answers.
  '''
  fname = ''
  lc_list = []
  fname_list = []
  lc_list_split = []
  bool_fname = bool(int(input('Would you like to input filenames to skip questions? Enter 0 ' + \
                              'for no, 1 for yes\n')))
  if bool_fname:
    while fname != 'done':
      fname = input("Enter a filename (path not necessary), or 'done' to move on\n")
      if fname != 'done':
        fname_list.append(fname)
    lc_group = []
    for fname in fname_list:
      lc_group.append(gf.gather_stereo_fname(fname,bool_smoothing)[0])
    freqs = []
    input_freq = ''
    while input_freq != 'done':
      input_freq = input("Enter frequencies (in kHz) one at a time, or 'done' to move on\n")
      if input_freq != 'done':
        freqs.append(float(input_freq))
    all_lcs = []
    #LightCurves at the given frequencies are returned for all time intervals specified earlier
    for lc in lc_group:
      for freq in freqs:
        all_lcs.append(get_lc_at_freq(lc, freq))
        split_info = prompt_split(all_lcs[0])
    #Splitting the LightCurves into subintervals if requested
    if split_info[0]:
      lc_list_split = split_lc_list(all_lcs, split_info[1])
      all_lcs = lc_list_split
  else:
    intervals, split_info = prompt_time()
    all_lcs = []
    #Checking which STEREO satellite to obtain data from
    ab_sel = input("Which STEREO satellite would you like data from? Enter 'A' or 'B'\n")
    if ab_sel == 'A':
      for interval in intervals:
        all_lcs = gf.gather_stereo_a(interval[0], interval[1],bool_smoothing)
    elif ab_sel == 'B':
      for interval in intervals:
        all_lcs.append(gf.gather_stereo_b(interval[0], interval[1],bool_smoothing))
    else:
      print("Enter 'A' or 'B'. Exitting.")
    if split_info[0]:
      lc_list_split = split_lc_list(all_lcs, split_info[1])
      all_lcs = lc_list_split
  return all_lcs

def prompt_methods():
  '''
  Asks the questions necessary to determine which methods to run, and how to run them.
  --------
  Returns:
  --------
  method_params : 2D array
    Array containing the instructions to run or not run a method, as well as the parameters
    for the run if necessary. 1st entry is peak finder, 2nd is wavelet, 3rd is FFT.
  bool_show : Boolean
    Toggles whether or not to show results.
  bool_save : Boolean
    Toggles whether or not to save results.
  '''
  method_params = [[], [], [], []]
  # First ask if the user wants to run the whole gamut of options.
  bool_all=bool(int(input('Do you want to run all analysis types? Enter 0 for no, 1 for yes\n')))
  if bool_all:
    method_params[0].append(bool(1))   # Yes to summary plot
    method_params[1].append(bool(1))   # Yes to peak finder
    bool_wavelet=bool(1)
    method_params[2].append(bool_wavelet)   # Yes to wavelet
    method_params[2].append(bool(0))   # No to pre-flattening in wavelet
    method_params[2].append(bool(1))   # Yes to saving text file of wavelet data
    bool_fft=bool(1)
    method_params[3].append(bool_fft)   # Yes to FFT
    method_params[3].append(bool(1))   # Yes to saving text file of FFT data
    bool_show=bool(0)   # Do not display (Lily's computer doesn't like this)
    bool_save=bool(1)   # Do save
  else:
    #First asks about a full summary plot
    bool_sum = bool(int(input('Do you want to generate a plot summarizing the results of all ' + \
                            'methods? Enter 0 for no, 1 for yes\n')))
    method_params[0].append(bool_sum)
    #Only asks about individual methods if a summary plot is not requested
    if not bool_sum:
      #Peak finder
      bool_pf = bool(int(input('Do you want to run the peak finder? Enter 0 for no, 1 for yes\n')))
      method_params[1].append(bool_pf)
      #Wavelet
      bool_wavelet = bool(int(input('Do you want to run the wavelet? Enter 0 for no, 1 for yes\n')))
      method_params[2].append(bool_wavelet)
      #FFT
      bool_fft = bool(int(input('Do you want to run the FFT? Enter 0 for no, 1 for yes\n')))
      method_params[3].append(bool_fft)
      # Wavelet-specific queries
      if bool_wavelet:
        bool_flatten = bool(int(input('WAVELETS: Do you want to perform flattening as a preprocessing ' + \
                                    'step? Enter 0 for no, 1 for yes\n')))
        bool_txt = bool(int(input('WAVELETS: Would you like to only? save the text file that summarizes' + \
                                'the results? Enter 0 for no, 1 for yes\n')))
        method_params[2].append(bool_flatten)
        method_params[2].append(bool_txt)
      # FFT-specific queries
      if bool_fft:
        bool_txt = bool(int(input('Would you like to only save the text file that summarizes' + \
                                'the results? Enter 0 for no, 1 for yes\n')))
        method_params[3].append(bool_txt)
    #Asking about showing and saving results (goes for all methods)
    bool_show = bool(int(input('Do you want to display results? Enter 0 for no, 1 for yes\n')))
    bool_save = bool(int(input('Do you want to save results? Enter 0 for no, 1 for yes\n')))
  return method_params, bool_show, bool_save

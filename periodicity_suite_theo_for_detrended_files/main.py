'''
This file defines and runs the function that performs periodicity analysis routines following the
parameters input via prompts given to the user.
'''

#Imports
import prompting
import os
import gather_functions as gf

def main():
  '''
  Prompts the user for the necessary information, and runs periodicity analysis routines
  accordingly.
  '''
  done_str = ''
  while done_str != 'done':
    #Getting the necessary information and creating LightCurves
    bool_fname = bool(int(input('Would you like to input filenames to skip questions? '+\
      'Enter 0 for no, 1 for yes\n')))
    if bool_fname:
      default_path='/home/sanakettu/research/umn/python/periodicity/data/lightcurve_examples/detrends_reformat/'
      print('The default directory within which to seek data is: ',default_path)
      #allowing the user to put in entire directories to analyze
      dir = bool(int(input('Would you like to specify a different directory? '+\
        'Enter 0 for no (=use default), 1 for yes\n')))
      fname_list = []
      fname = ''
      if dir:
        path = input("Enter the FULL PATH for the desired files.\n")
      else:
        path=default_path
#      print(path,'...',path.split('/'))
      path_last=path.split('/')[-2]
#      print(path_last)
      if path_last == 'regularized':
        data_format=1
      elif path_last == 'detrends_reformat':
        data_format=3
      else:
        data_format=int(input('Input data file format? (1) Yuni/Reed. (2) Cindy/Autoplot.\n')) 
#        data_format=2
      fname_list = os.listdir(path)
    else:
      source = input('Enter the instrument whose data you wish to analyze. '+\
        'psp, aia, nustar, wind, or stereo\n')
    #Asking which periodicity methods to run and the necessary follow-up questions
    method_params, bool_show, bool_save = prompting.prompt_methods()
    min_dist=180  #seconds: minimum spacing required between peak finder results
    if bool_fname:
      lc_list = []
      for file in fname_list:
        source=file.split('_')[0]
        lc_list.append(gf.gather_data_from_file(file,source,dir,path,
          input_dist=min_dist,format=data_format))
    else:
      if source.upper() == 'PSP':
        lc_list  = prompting.prompt_psp(input_dist=min_dist)
      elif source.upper() == 'AIA':
        lc_list = prompting.prompt_aia(input_dist=min_dist) 
      elif source.upper() == 'NUSTAR':
        lc_list = prompting.prompt_nustar(input_dist=min_dist)
      elif source.upper() == 'WIND':
        lc_list  = prompting.prompt_wind(input_dist=min_dist)
      elif source.upper() == 'STEREO':
        lc_list  = prompting.prompt_stereo(input_dist=min_dist)
      else:
        print("Enter a valid instrument name. Accepted values: "+\
          "'psp', 'aia', 'nustar', 'wind', and 'stereo'")
        return
    #Running the methods as specified on the LightCurves in lc_list
    for lightcurve in lc_list:
      print('\nProcessing ... ',lightcurve,' at ',lightcurve.get_time_str(),
        ', with '+lightcurve.get_param_msg(),'\n')
      if method_params[0][0]:
        lightcurve.summary_plot(bool_show, bool_save)
      #This is a parameter for the FFT that is changed if the wavelet is run
      period_ax_range = 0
      #Peak finder
      if method_params[1][0]:
        _, _, _, _ = lightcurve.peak_finder(min_dist,
          show_results=bool_show,save_results=bool_save)
        lightcurve.plot_time_series(show_peaks=True,
          input_dist=min_dist, show_plot=bool_show,save_plot=bool_save)
      #Wavelet
      if method_params[2][0]:
        period_ax_range = lightcurve.wavelet(bool_show, bool_save,
          'Morlet',method_params[2][1], method_params[2][2])
      #FFT
      if method_params[3][0]:
        lightcurve.fft_periodicity(bool_show, bool_save,
          period_ax_range, method_params[3][1])

    #Giving the option to do the whole process again
    done_str = input("Press Enter to repeat this process, or enter 'done' to exit\n")

if __name__ == '__main__':
  main()

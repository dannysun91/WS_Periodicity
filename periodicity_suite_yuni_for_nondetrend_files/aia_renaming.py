#  limb_ar_060000-070000_94.0 Angstrom_N21_lc.csv
# to 
# lc_20220906_090000_to_20220906_111000_171_N21_reg2.csv


from os import listdir
import glob
from shutil import copy2

base_dir='/home/sanakettu/research/umn/python/periodicity/data/lightcurve_examples/aia_from_reed_20250721/'
orig_dir=base_dir+'detrend_orig/'
dest_dir=base_dir+'detrend_renamed/'
#wildcard='20*/'

#for path in glob.glob(orig_dir+wildcard):
for path in glob.glob(orig_dir):
  path_parts=path.split('/')
  for fname in listdir(path):
##    print(fname)
    if fname.split('.')[-1]=='csv':
      try:
        parts=fname.split('_')
        regname=parts[0]
        if regname=='limb':
          regnum=1
        elif regname=='north':
          regnum=2
        elif regname=='south':
          regnum=3
        elif regname=='southeast':
          regnum=4
        elif regname=='northwest':
          regnum=5
        elif regname=='southwest':
          regnum=6
        else:
          regnum=7
        if regname != 'tamarpsp':
          idx=regname.find('psp')
          if idx==-1:
            source='aia'
          else:
            source='psp'
        else:
          source='aia'
##        print(parts,'...',regname,'...',source)
##        print(parts[-4],parts[-3])
#        date=path_parts[-2]
        date='20221208'
        t_start=parts[-4].split('-')[0]
        t_end=parts[-4].split('-')[1]
        wavelength=parts[-3].split('.')[0]
#        t_start=parts[-3].split('-')[0]
#        t_end=parts[-3].split('-')[1]
#        wavelength=parts[-2].split('.')[0]
##        print('WAVELENGTH',wavelength)
        new_fname=f'{source}_{date}_{t_start}_to_{date}_{t_end}_{wavelength}_N21_reg{regnum}_{regname}.csv'
        print('Saving file... ',new_fname,'\n')
        copy2(path+fname, dest_dir+new_fname)
      except:
        print('FAILED ON FILE',fname,'\n')
        pass


import os
import numpy as np
from computage.settings import ROOTDIR

#function for clock file retrieval
def get_clock_file(filename):
    clock_file_path = os.path.join(ROOTDIR, "models_library/raw_models/", filename)  
    return clock_file_path

#Horvath-specific ELU-like transform
def anti_trafo(x, adult_age=20):
    y = np.where(
        x < 0, (1 + adult_age) * np.exp(x) - 1, (1 + adult_age) * x + adult_age
    )
    return y

#identity transform if not given other
def identity(x):
    return x

### model definitions are taken from biolearn: https://github.com/bio-learn/biolearn/blob/master/ ###


models_path = os.path.join(ROOTDIR,'/models_library/raw_models/')
model_files = os.listdir(models_path)
model_names = list(map(lambda a: a.replace('.csv','').lower(), model_files))

model_definitions = dict([(key, value)
          for i, (key, value) in enumerate(zip(model_names, model_files))])
'''
{'epitoc2': 'EpiTOC2.csv', 'dunedinpoam': 'DunedinPoAm.csv', 
'mccartneyblood_2018_alcohol': 'McCartneyBlood_2018_Alcohol.csv', 
'mccartneyblood_2018_education': 'McCartneyBlood_2018_Education.csv', 
'linblood99cpg_2016': 'LinBlood99CpG_2016.csv', 'linblood3cpg_2016': 'LinBlood3CpG_2016.csv', 
'hannumlung_2013': 'HannumLung_2013.csv', 
'horvathmultishrunken_2013': 'HorvathMultiShrunken_2013.csv', 'hannumblood_2013': 'HannumBlood_2013.csv', 
'horvathmulti_2013': 'HorvathMulti_2013.csv', 'mccartneyblood_2018_bmi': 'McCartneyBlood_2018_BMI.csv', 
'mccartneyblood_2018_hdl': 'McCartneyBlood_2018_HDL.csv', 'mccartneyblood_2018_ldl': 'McCartneyBlood_2018_LDL.csv', 
'hannumbreast_2013': 'HannumBreast_2013.csv', 'knightblood_2016': 'KnightBlood_2016.csv', 'yingdamage': 'YingDamAge.csv', 
'mccartneyblood_2018_waisttipratio': 'McCartneyBlood_2018_WaistTipRatio.csv', 'hannumkidney_2013': 'HannumKidney_2013.csv',
 'mccartneyblood_2018_totalfat': 'McCartneyBlood_2018_TotalFat.csv', 'yingcausage': 'YingCausAge.csv', 
 'mccartneyblood_2018_tc': 'McCartneyBlood_2018_TC.csv', 'mccartneyblood_2018_smoking': 'McCartneyBlood_2018_Smoking.csv', 
 'yingadaptage': 'YingAdaptAge.csv', 'phenoage': 'PhenoAge.csv'}
'''

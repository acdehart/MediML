import os
import sys

import uproot
import pandas as pd

def root_to_df(root_path, column_names):

    pickle_path = f"./med_data_{root_path.split('.')[0]}.pkl"
    if os.path.exists(pickle_path):
        tree_pd = pd.read_pickle(pickle_path)
    else:
        file = uproot.open(root_path)
        tree_key = file.keys()[0]
        tree = file[tree_key]
        column_names = file['TVAERS;1'].keys()
        print(column_names)
        tree_pd = tree.arrays(column_names, library="pd")
        tree_pd.to_pickle(pickle_path)
        input("Done Pickling!")

    # Manage datedied and vax_date

    return tree_pd

###### List of all keys in TVAERS_ntuple.root TTree object
#['vaers_id', 'recvdate', 'state', 'age_yrs', 'sex', 'symptom_text', 'died', 'datedied', 'rpt_date', 'l_threat', 'er_visit', 'hospital', 'hospdays', 'x_stay', 'disable', 'recovd', 'vax_date', 'onset_date', 'numdays', 'lab_data', 'v_adminby', 'v_fundby', 'other_meds', 'cur_ill', 'history', 'prior_vax', 'splttype', 'form_vers', 'birth_defect', 'ofc_visit', 'er_ed_visit', 'allergies', 'vax_type', 'vax_manu', 'vax_lot', 'vax_dose_series', 'vax_route', 'vax_site', 'vax_name', 'symptoms', 'symptom_versions']

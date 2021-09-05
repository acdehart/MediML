import uproot
import pandas as pd

def root_to_df(root_path, column_names):
    file = uproot.open(root_path)
    tree_key = file.keys()[0]
    tree = file[tree_key]
    tree_pd = tree.arrays(column_names, library="pd")
    tree_pd.to_pickle("./med_data.pkl")

###### List of all keys in TVAERS_ntuple.root TTree object
#['vaers_id', 'recvdate', 'state', 'age_yrs', 'sex', 'symptom_text', 'died', 'datedied', 'rpt_date', 'l_threat', 'er_visit', 'hospital', 'hospdays', 'x_stay', 'disable', 'recovd', 'vax_date', 'onset_date', 'numdays', 'lab_data', 'v_adminby', 'v_fundby', 'other_meds', 'cur_ill', 'history', 'prior_vax', 'splttype', 'form_vers', 'birth_defect', 'ofc_visit', 'er_ed_visit', 'allergies', 'vax_type', 'vax_manu', 'vax_lot', 'vax_dose_series', 'vax_route', 'vax_site', 'vax_name', 'symptoms', 'symptom_versions']

#!/usr/bin/env python
# coding: utf-8

# In[132]:


#get_ipython().system('pip install --user uproot')
import sys
sys.path.append("/home/acraplet/Alie/UROP2021/AnalyserWork/RootFiles")
import uproot 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import xgboost as xgb
import matplotlib as mpl
#mpl.use('Agg')
mpl.use('TkAgg')
import matplotlib.pyplot as plt

#A code to train a BDTto distinguish CP-even from CP-Odd events using the available signal root files. The output is a model to be used on all datasets, therefore be careful about having the samevariables on all your files, run an example locally when in doubt 

# The signal datafiles that we are using in the training
tree = uproot.open("/vols/cms/ac4317/UROP_2021/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/output/outputBDT2/GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
tree2 = uproot.open("/vols/cms/ac4317/UROP_2021/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/output/outputBDT2/VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]

#Variables to import, some will be droped for the training
variables = [  "wt_cp_sm", "wt_cp_ps", "wt_cp_mm",
                "rand",
                "pt_1","pt_2",
                "met",
                "aco_angle_1", "aco_angle_5", "aco_angle_6",
                "y_1_1", "y_1_2",
                "ip_sig_1", "ip_sig_2",
                "mva_dm_1","mva_dm_2",
                "tau_decay_mode_1","tau_decay_mode_2",
                "deepTauVsJets_medium_1","deepTauVsJets_medium_2",
                "deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
                "deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
                "trg_doubletau", "pv_angle_new", "BDTOddEven22"
             ]


pi_1 = ["pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1"]
pi_2 = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2"]

pi0_1 = ["pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1"]
pi0_2 = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2"]

pi2_1 = ["pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1"]
pi2_2 = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2"]

pi3_1 = ["pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1"]
pi3_2 = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"]


pis = pi_1 + pi_2 + pi2_1 + pi2_2 + pi3_1 + pi3_2 

#Here the pion four-momenta is not included as input variable -> it was found that it didn't improve the training much
df = tree.pandas.df(variables)# + pis)
df2 = tree2.pandas.df(variables)# + pis)


#index, a bit sloppy, will need improvement later

df['index'] =  np.arange(0, len(df), 1)
df2['index'] =  np.arange(0, len(df2), 1)


import random
random.seed(123456)

#select the rhorho channel
df_1 = df[
      (df["tau_decay_mode_1"] == 1) 
    & (df["tau_decay_mode_2"] == 1) 
    & (df["mva_dm_1"] == 1) 
    & (df["mva_dm_2"] == 1)
   
]

df2_1 = df2[
      (df2["tau_decay_mode_1"] == 1) 
    & (df2["tau_decay_mode_2"] == 1) 
    & (df2["mva_dm_1"] == 1) 
    & (df2["mva_dm_2"] == 1)
   
]

#construct, using the weights the CP-even and CP-odd datasets
df_ps = df_1[
      (df_1["rand"]<df_1["wt_cp_ps"]/2)
]

df_sm = df_1[
      (df_1["rand"]<df_1["wt_cp_sm"]/2)
]


df2_ps = df2_1[
      (df2_1["rand"]<df2_1["wt_cp_ps"]/2)
]

df2_sm = df2_1[
      (df2_1["rand"]<df2_1["wt_cp_sm"]/2)
]


         
# prepare the target labels
y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))

y2_sm = pd.DataFrame(np.ones(df2_sm.shape[0]))
y2_ps = pd.DataFrame(np.zeros(df2_ps.shape[0]))

y = pd.concat([y_sm, y_ps,y2_sm, y2_ps])
y.columns = ["class"]


# prepare the dataframe to use in training
X = pd.concat([df_sm, df_ps, df2_sm, df2_ps])

# drop any other variables that aren't required in training

X2 = X.drop([
            "wt_cp_sm","wt_cp_ps","wt_cp_mm", "rand",
            "tau_decay_mode_1","tau_decay_mode_2","mva_dm_1","mva_dm_2",
            "deepTauVsJets_medium_1","deepTauVsJets_medium_2",
            "deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
            "deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
            "trg_doubletau",
           ], axis=1).reset_index(drop=True) 

#split the dataset into training an testing. We use 80-20% ratios
X1_train,X1_test, y1_train, y1_test  = train_test_split(
    X2,
    y,
    test_size=0.2,
    random_state=123456,
    stratify=y.values,
)


#Set up and run the XGboost algorithm 
xgb_params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "learning_rate": 0.02,
    "silent": 1,
    "n_estimators": 1000,
    "subsample": 0.9,
    "seed": 123451,
}


xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(
    X1_train,
    y1_train,
    early_stopping_rounds=200, # stops the training if doesn't improve after 200 iterations
    eval_set=[(X1_train, y1_train), (X1_test, y1_test)],
    eval_metric = "auc", # can use others
    verbose=True,
)


#Use the model to predict BDT score for the full dataset
y1_proba1 = xgb_clf.predict_proba(X2)
auc = roc_auc_score(y, y1_proba1[:,1])

print("This is the auc:", auc)
#Add the BDT weight to the dataset
X2['BDTweights'] = y1_proba1.T[0]



# Add back the items that we need: weights
array_wt_cp_sm = np.array(X["wt_cp_sm"])
X2 = X2.assign(wt_cp_sm=array_wt_cp_sm)

array_wt_cp_ps = np.array(X["wt_cp_ps"])
X2 = X2.assign(wt_cp_ps=array_wt_cp_ps)

array_wt_cp_mm = np.array(X["wt_cp_mm"])
X2 = X2.assign(wt_cp_mm=array_wt_cp_mm)

array_rand = np.array(X["rand"])
X2 = X2.assign(rand=array_rand)


# Form new sm/ps datasets for training checks
df_ps = X2[
      (X2["rand"]<X2["wt_cp_ps"]/2)
]

df_sm = X2[
      (X2["rand"]<X2["wt_cp_sm"]/2)
]

# Some plots to see the sm/ps separation in BDT score
plt.hist(df_ps['BDTweights'], bins = 100, alpha = 0.5, label = 'Pseudoscalar')
plt.hist(df_sm['BDTweights'], bins = 100, alpha = 0.5, label = 'Standard Model')
plt.xlabel('predict_proba')
plt.ylabel('Occurences')
plt.title('Probability of cp even for rho rho training without 4 vector\nwith reweighting')
plt.legend()
#plt.savefig('Test_with_weights_fullVBFGluGlu_2.png')
plt.close()


#  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
def plot_roc_curve(fpr, tpr, auc, string):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label = 'Auto-made')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'ROC AUC Score: {:.3f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #plt.savefig('%s'%string)

#and for a sanity check, a function to plot your own roc curves
def hand_made_roc(df_ps, df_sm):
    fraction_sm = []
    fraction_ps = []
    array = np.linspace(0, 1, 1000)
    for x in array:
        sup_ps = np.where(df_ps['BDTweights']>x, 0, 1)
        fraction_ps.append(np.sum(sup_ps)/len(df_ps))
        
        sup_sm = np.where(df_sm['BDTweights']>x, 0, 1)
        fraction_sm.append(np.sum(sup_sm)/len(df_sm))
    return fraction_ps, fraction_sm

#plot to check the roc curve
fraction_ps, fraction_sm = hand_made_roc(df_ps, df_sm)
plt.plot(fraction_ps, fraction_sm, label = 'Hand-made')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
array = np.linspace(0, 1, 100)
plt.plot(array, array, 'k--')
plt.legend()
#plt.savefig('Test_self_roc_5')
plt.close()

#Save the model 
import pickle

with open('tt_RhoRho_Model4_GluGluVBF_xgb.pkl', 'wb') as f:
        pickle.dump(xgb_clf, f, protocol=2)




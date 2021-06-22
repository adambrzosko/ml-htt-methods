#!/usr/bin/env python
# coding: utf-8

import sys

import ROOT
import uproot # can also use root_pandas or root_numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc

import matplotlib.pyplot as plt
import xgboost as xgb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--channel', '-c', help= 'channel to train for')
parser.add_argument('--year', '-y', help= 'year to train for')
parser.add_argument('--sign', '-s', help= 'Even or odd event number training?')
args = parser.parse_args()
year = str(args.year)
chan = str(args.channel)
sign = str(args.sign)

if not (sign == 'even' or sign == 'odd' ): 
  print 'ERROR must select sign = even or odd'
  exit()        

def GetCrossWeight(samp,year):
    filename='params_mssm_%(year)s.json' % vars()
    with open(filename, 'r') as f:
        info = json.load(f)

    lumi = info['SingleMuon']['lumi']
    xs = info[samp]['xs']
    evt = info[samp]['evt']
    print '(lumi, xs, et) = (%f,%f,%f)' % (lumi, xs, evt)
    return xs*lumi/evt
    


sel_vars = [
    "event",
    "wt",
    "wt_ff_mssm_1",
    "os", 
    "iso_1",
    "gen_match_1",
    "gen_match_2",
    "deepTauVsJets_medium_1",
    "deepTauVsJets_vvvloose_1",
    "deepTauVsEle_vvloose_1",
    "deepTauVsEle_tight_1",
    "deepTauVsMu_tight_1",
    "deepTauVsMu_vloose_1",
    "deepTauVsJets_medium_2",
    "deepTauVsJets_vvvloose_2",
    "deepTauVsEle_vvloose_2",
    "deepTauVsEle_tight_2",
    "deepTauVsMu_tight_2",
    "deepTauVsMu_vloose_2",
    "leptonveto",
    "trg_mutaucross",
    "trg_etaucross",
    "trg_singlemuon",
    "trg_singleelectron",
    "trg_doubletau",
    "trg_muonelectron",
    "n_bjets",
]

training_vars = [
  "pt_1",
  "pt_2",
  "m_vis",
  "pt_vis",
  "svfit_mass",
  "jpt_1",
  "n_jets",
  "jdeta",
  "mjj",
  "dijetpt",
  "pt_tt",
  "mt_1",
  "mt_tot",
  "mt_2",
  "mt_lep",
  "met",
  #"ME_q2v1",
  #"ME_q2v2",
  "jpt_2",
]

invars = training_vars+sel_vars

if year == '2016':
  ztt_samples = ['DYJetsToLL-LO-ext1','DYJetsToLL-LO-ext2','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO','EWKZ2Jets_ZToLL','EWKZ2Jets_ZToLL-ext1','EWKZ2Jets_ZToLL-ext2']
  top_samples = ['TT']
  vv_samples = ['T-tW', 'Tbar-tW','Tbar-t','T-t','WWTo1L1Nu2Q','WZJToLLLNu','VVTo2L2Nu','VVTo2L2Nu-ext1','ZZTo2L2Q','ZZTo4L-amcat','WZTo2L2Q','WZTo1L3Nu','WZTo1L1Nu2Q']
  wjets_samples = ['WJetsToLNu-LO', 'WJetsToLNu-LO-ext','W1JetsToLNu-LO','W2JetsToLNu-LO','W2JetsToLNu-LO-ext','W3JetsToLNu-LO','W3JetsToLNu-LO-ext','W4JetsToLNu-LO','W4JetsToLNu-LO-ext1','W4JetsToLNu-LO-ext2', 'EWKWMinus2Jets_WToLNu','EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1','EWKWPlus2Jets_WToLNu-ext2']

  sm_higgs = ['GluGluToHToTauTau_M-125','VBFHToTauTau_M-125','ZHToTauTau_M-125','WplusHToTauTau_M-125','WminusHToTauTau_M-125','VBFHToTauTau_M-95']


if year == '2017':

  ztt_samples = ['DYJetsToLL-LO','DYJetsToLL-LO-ext1','DY1JetsToLL-LO','DY1JetsToLL-LO-ext','DY2JetsToLL-LO','DY2JetsToLL-LO-ext','DY3JetsToLL-LO','DY3JetsToLL-LO-ext','DY4JetsToLL-LO','EWKZ2Jets']
  top_samples = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']
  vv_samples = ['T-tW', 'Tbar-tW','Tbar-t','T-t','WWToLNuQQ','WZTo2L2Q','WZTo1L1Nu2Q','WZTo1L3Nu','WZTo3LNu', 'WWTo2L2Nu']
  wjets_samples = ['WJetsToLNu-LO','WJetsToLNu-LO-ext','W1JetsToLNu-LO','W2JetsToLNu-LO','W3JetsToLNu-LO','W4JetsToLNu-LO','EWKWMinus2Jets','EWKWPlus2Jets']

  sm_higgs = ['GluGluHToTauTau_M-125','GluGluHToTauTau_M-125-ext','VBFHToTauTau_M-125','ZHToTauTau_M-125','WplusHToTauTau_M-125','WminusHToTauTau_M-125']

if year == '2018':

  ztt_samples = ['DYJetsToLL-LO','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO','EWKZ2Jets']
  top_samples = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']
  vv_samples = [
          'T-tW-ext1', 'Tbar-tW-ext1','Tbar-t','WWTo2L2Nu','T-t',
          'WWToLNuQQ','WZTo1L3Nu','WZTo3LNu','WZTo3LNu-ext1','WZTo2L2Q',
          'ZZTo2L2Nu-ext1','ZZTo2L2Nu-ext2','ZZTo2L2Q','ZZTo4L-ext','ZZTo4L'
          ]
  wjets_samples = ['WJetsToLNu-LO','W1JetsToLNu-LO','W2JetsToLNu-LO','W3JetsToLNu-LO','W4JetsToLNu-LO','EWKWMinus2Jets','EWKWPlus2Jets']

  sm_higgs = ['GluGluHToTauTau_M-125', 'VBFHToTauTau_M-125-ext1','ZHToTauTau_M-125','WplusHToTauTau_M-125','WminusHToTauTau_M-125']

# remove wjets samples because they come from FF
wjets_samples = []
backgrounds = ztt_samples+top_samples+vv_samples+wjets_samples+sm_higgs

data_name=''
if chan == 'mt': 
  data_name = 'SingleMuon'
if chan == 'et': 
  if year!='2018': data_name = 'SingleElectron'
  else: data_name = 'EGamma'
if chan == 'tt':
  data_name = 'Tau'
if chan == 'em':
  data_name = 'MuonEG'

#backgrounds = ['DYJetsToLL-LO-ext1','DYJetsToLL-LO-ext2','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO']
#backgrounds = ['DYJetsToLL-LO','DY1JetsToLL-LO','DY2JetsToLL-LO','DY3JetsToLL-LO','DY4JetsToLL-LO','EWKZ2Jets']

runs = []
if year == '2016':
  runs = ['B','C','D','E','F','G','H']
if year == '2017':
  runs = ['B','C','D','E','F']
if year == '2018':
  runs = ['A','B','C','D']

#runs=['C']
ff_files = []

for r in runs: ff_files.append(data_name+r)

vbf_signal = ['VBFHToTauTau_M-95']
ggh_signal = ['SUSYGluGluToHToTauTau_M-95_powheg']

file_ext='_%(chan)s_%(year)s.root' % vars()
sdir='./'

if year == '2016':
  sdir='/vols/cms/dw515/Offline/output/MSSM/mssm_2016_v2/'
if year == '2017':
  sdir='/vols/cms/dw515/Offline/output/MSSM/mssm_2017_v2/'
if year == '2018':
  sdir='/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/'

def apply_filters(df, chan='mt',FF=False):

  if chan == 'mt':
    if not FF:
      df = df[
           (df["gen_match_2"] < 6)
          &(df["os"] > 0.5)
          &(df["wt"] < 2)
          &(df["mt_1"] < 50)
          &(df["iso_1"] < 0.15)
          &(df["deepTauVsJets_medium_2"] > 0.5)
          &(df["deepTauVsEle_vvloose_2"] > 0.5)
          &(df["deepTauVsMu_tight_2"] > 0.5)
          &(df["leptonveto"] == 0)
          &(df["n_bjets"] == 0)
          &((df["trg_singlemuon"] > 0.5)  | (df["trg_mutaucross"] > 0.5))
          &(df["svfit_mass"] < 250)
      ]
    else:  
      df = df[
          (df["os"] > 0.5)
         &(df["wt"] < 2)
         &(df["mt_1"] < 50)
         &(df["iso_1"] < 0.15)
         &(df["deepTauVsJets_vvvloose_2"] > 0.5)
         &(df["deepTauVsJets_medium_2"] < 0.5)
         &(df["deepTauVsEle_vvloose_2"] > 0.5)
         &(df["deepTauVsMu_tight_2"] > 0.5)
         &(df["leptonveto"] == 0)
         &(df["n_bjets"] == 0)
         &((df["trg_singlemuon"] > 0.5)  | (df["trg_mutaucross"] > 0.5))
         &(df["svfit_mass"] < 250)
      ]
  if chan == 'et':
    if not FF:
      df = df[
           (df["gen_match_2"] < 6)
          &(df["os"] > 0.5)
          &(df["wt"] < 2)
          &(df["mt_1"] < 50)
          &(df["iso_1"] < 0.15)
          &(df["deepTauVsJets_medium_2"] > 0.5)
          &(df["deepTauVsEle_tight_2"] > 0.5)
          &(df["deepTauVsMu_vloose_2"] > 0.5)
          &(df["leptonveto"] == 0)
          &(df["n_bjets"] == 0)
          &((df["trg_singleelectron"] > 0.5)  | (df["trg_etaucross"] > 0.5))
          &(df["svfit_mass"] < 250)
      ]
    else:
      df = df[
            (df["os"] > 0.5)
           &(df["wt"] < 2)
           &(df["mt_1"] < 50)
           &(df["iso_1"] < 0.15)
           &(df["deepTauVsJets_vvvloose_2"] > 0.5)
           &(df["deepTauVsJets_medium_2"] < 0.5)
           &(df["deepTauVsEle_tight_2"] > 0.5)
           &(df["deepTauVsMu_vloose_2"] > 0.5)
           &(df["leptonveto"] == 0)
           &(df["n_bjets"] == 0)
           &((df["trg_singleelectron"] > 0.5)  | (df["trg_etaucross"] > 0.5))
           &(df["svfit_mass"] < 250)
      ]
  if chan == 'tt':
    if not FF:
      df = df[
           (df["gen_match_1"] < 6)
          &(df["os"] > 0.5)
          &(df["wt"] < 2)
          &(df["deepTauVsJets_medium_1"] > 0.5)
          &(df["deepTauVsEle_vvloose_1"] > 0.5)
          &(df["deepTauVsMu_vloose_1"] > 0.5)
          &(df["deepTauVsJets_medium_2"] > 0.5)
          &(df["deepTauVsEle_vvloose_2"] > 0.5)
          &(df["deepTauVsMu_vloose_2"] > 0.5)
          &(df["leptonveto"] == 0)
          &(df["n_bjets"] == 0)
          &(df["trg_doubletau"] > 0.5)
          &(df["svfit_mass"] < 250)
      ]
    else:    
      df = df[
          (df["os"] > 0.5)
          &(df["wt"] < 2)
          &(df["deepTauVsJets_vvvloose_1"] > 0.5)
          &(df["deepTauVsJets_medium_1"] < 0.5)
          &(df["deepTauVsEle_vvloose_1"] > 0.5)
          &(df["deepTauVsMu_vloose_1"] > 0.5)
          &(df["deepTauVsJets_medium_2"] > 0.5)
          &(df["deepTauVsEle_vvloose_2"] > 0.5)
          &(df["deepTauVsMu_vloose_2"] > 0.5)
          &(df["leptonveto"] == 0)
          &(df["n_bjets"] == 0)
          &(df["trg_doubletau"] > 0.5)
          &(df["svfit_mass"] < 250)
      ]
  return df

count=0
for f in backgrounds:
   print 'loading %(f)s' % vars() 
   print sdir+f+file_ext
   tree=uproot.open(sdir+f+file_ext)["ntuple"]
   df = tree.pandas.df(invars)
   sf = GetCrossWeight(f,year)
   print f, sf
   df.loc[:,"wt_2"] = df["wt"]*sf
   df=apply_filters(df,chan,False) 
   if count == 0: backgrounds_df = df
   else: backgrounds_df = pd.concat([backgrounds_df,df], ignore_index=True)
   count+=1 

count=0
for f in vbf_signal:
   print 'loading %(f)s' % vars() 
   tree=uproot.open(sdir+f+file_ext)["ntuple"]
   df = tree.pandas.df(invars)
   sf = GetCrossWeight(f,year)
   print f, sf
   df.loc[:,"wt_2"] = df["wt"]*sf
   df=apply_filters(df,chan,False) 
   if count == 0: vbf_df = df
   else: vbf_df = pd.concat([vbf_df,df], ignore_index=True)
   count+=1

count=0
for f in ggh_signal:
   print 'loading %(f)s' % vars() 
   tree=uproot.open(sdir+f+file_ext)["ntuple"]
   df = tree.pandas.df(invars)
   sf = GetCrossWeight(f,year)
   print f, sf
   df.loc[:,"wt_2"] = df["wt"]*sf
   df=apply_filters(df,chan,False) 
   if count == 0: ggh_df = df
   else: ggh_df = pd.concat([ggh_df,df], ignore_index=True)
   count+=1

count=0
for f in ff_files:
   print 'loading %(f)s' % vars()    
   tree=uproot.open(sdir+f+file_ext)["ntuple"]
   df = tree.pandas.df(invars)
   df.loc[:,"wt_2"] = df["wt_ff_mssm_1"]
   df=apply_filters(df,chan,True) 
   if count == 0: ff_df = df
   else: ff_df = pd.concat([ff_df,df], ignore_index=True)
   count+=1

print 'done'


if chan == 'mt':

  backgrounds_df = backgrounds_df[
       (backgrounds_df["gen_match_2"] < 6)
      &(backgrounds_df["os"] > 0.5)
      &(backgrounds_df["wt"] < 2)
      &(backgrounds_df["mt_1"] < 50)
      &(backgrounds_df["iso_1"] < 0.15)
      &(backgrounds_df["deepTauVsJets_medium_2"] > 0.5)
      &(backgrounds_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(backgrounds_df["deepTauVsMu_tight_2"] > 0.5)
      &(backgrounds_df["leptonveto"] == 0)
      &(backgrounds_df["n_bjets"] == 0)
      &((backgrounds_df["trg_singlemuon"] > 0.5)  | (backgrounds_df["trg_mutaucross"] > 0.5))
      &(backgrounds_df["svfit_mass"] < 250)
  ]
  
  vbf_df = vbf_df[
       (vbf_df["gen_match_2"] < 6)
      &(vbf_df["os"] > 0.5)
      &(vbf_df["wt"] < 2)
      &(vbf_df["mt_1"] < 50)
      &(vbf_df["iso_1"] < 0.15)
      &(vbf_df["deepTauVsJets_medium_2"] > 0.5)
      &(vbf_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(vbf_df["deepTauVsMu_tight_2"] > 0.5)
      &(vbf_df["leptonveto"] == 0)
      &(vbf_df["n_bjets"] == 0)
      &((vbf_df["trg_singlemuon"] > 0.5)  | (vbf_df["trg_mutaucross"] > 0.5))
      &(vbf_df["svfit_mass"] < 250)
  ]
  
  ggh_df = ggh_df[
       (ggh_df["gen_match_2"] < 6)
      &(ggh_df["os"] > 0.5)
      &(ggh_df["wt"] < 2)
      &(ggh_df["mt_1"] < 50)
      &(ggh_df["iso_1"] < 0.15)
      &(ggh_df["deepTauVsJets_medium_2"] > 0.5)
      &(ggh_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(ggh_df["deepTauVsMu_tight_2"] > 0.5)
      &(ggh_df["leptonveto"] == 0)
      &(ggh_df["n_bjets"] == 0)
      &((ggh_df["trg_singlemuon"] > 0.5)  | (ggh_df["trg_mutaucross"] > 0.5))
      &(ggh_df["svfit_mass"] < 250)
  ]

  ff_df = ff_df[
        (ff_df["os"] > 0.5)
       &(ff_df["wt"] < 2)
       &(ff_df["mt_1"] < 50)
       &(ff_df["iso_1"] < 0.15)
       &(ff_df["deepTauVsJets_vvvloose_2"] > 0.5)
       &(ff_df["deepTauVsJets_medium_2"] < 0.5)
       &(ff_df["deepTauVsEle_vvloose_2"] > 0.5)
       &(ff_df["deepTauVsMu_tight_2"] > 0.5)
       &(ff_df["leptonveto"] == 0)
       &(ff_df["n_bjets"] == 0)
       &((ff_df["trg_singlemuon"] > 0.5)  | (ff_df["trg_mutaucross"] > 0.5))
       &(ff_df["svfit_mass"] < 250)
   ]

if chan == 'et':

  backgrounds_df = backgrounds_df[
       (backgrounds_df["gen_match_2"] < 6)
      &(backgrounds_df["os"] > 0.5)
      &(backgrounds_df["wt"] < 2)
      &(backgrounds_df["mt_1"] < 50)
      &(backgrounds_df["iso_1"] < 0.15)
      &(backgrounds_df["deepTauVsJets_medium_2"] > 0.5)
      &(backgrounds_df["deepTauVsEle_tight_2"] > 0.5)
      &(backgrounds_df["deepTauVsMu_vloose_2"] > 0.5)
      &(backgrounds_df["leptonveto"] == 0)
      &(backgrounds_df["n_bjets"] == 0)
      &((backgrounds_df["trg_singleelectron"] > 0.5)  | (backgrounds_df["trg_etaucross"] > 0.5))
      &(backgrounds_df["svfit_mass"] < 250)
  ]

  vbf_df = vbf_df[
       (vbf_df["gen_match_2"] < 6)
      &(vbf_df["os"] > 0.5)
      &(vbf_df["wt"] < 2)
      &(vbf_df["mt_1"] < 50)
      &(vbf_df["iso_1"] < 0.15)
      &(vbf_df["deepTauVsJets_medium_2"] > 0.5)
      &(vbf_df["deepTauVsEle_tight_2"] > 0.5)
      &(vbf_df["deepTauVsMu_vloose_2"] > 0.5)
      &(vbf_df["leptonveto"] == 0)
      &(vbf_df["n_bjets"] == 0)
      &((vbf_df["trg_singleelectron"] > 0.5)  | (vbf_df["trg_etaucross"] > 0.5))
      &(vbf_df["svfit_mass"] < 250)
  ]

  ggh_df = ggh_df[
       (ggh_df["gen_match_2"] < 6)
      &(ggh_df["os"] > 0.5)
      &(ggh_df["wt"] < 2)
      &(ggh_df["mt_1"] < 50)
      &(ggh_df["iso_1"] < 0.15)
      &(ggh_df["deepTauVsJets_medium_2"] > 0.5)
      &(ggh_df["deepTauVsEle_tight_2"] > 0.5)
      &(ggh_df["deepTauVsMu_vloose_2"] > 0.5)
      &(ggh_df["leptonveto"] == 0)
      &(ggh_df["n_bjets"] == 0)
      &((ggh_df["trg_singleelectron"] > 0.5)  | (ggh_df["trg_etaucross"] > 0.5))
      &(ggh_df["svfit_mass"] < 250)
  ]

  ff_df = ff_df[
        (ff_df["os"] > 0.5)
       &(ff_df["wt"] < 2)
       &(ff_df["mt_1"] < 50)
       &(ff_df["iso_1"] < 0.15)
       &(ff_df["deepTauVsJets_vvvloose_2"] > 0.5)
       &(ff_df["deepTauVsJets_medium_2"] < 0.5)
       &(ff_df["deepTauVsEle_tight_2"] > 0.5)
       &(ff_df["deepTauVsMu_vloose_2"] > 0.5)
       &(ff_df["leptonveto"] == 0)
       &(ff_df["n_bjets"] == 0)
       &((ff_df["trg_singleelectron"] > 0.5)  | (ff_df["trg_etaucross"] > 0.5))
       &(ff_df["svfit_mass"] < 250)
   ]

if chan == 'tt':

  backgrounds_df = backgrounds_df[
       (backgrounds_df["gen_match_1"] < 6)
      &(backgrounds_df["os"] > 0.5)
      &(backgrounds_df["wt"] < 2)
      &(backgrounds_df["deepTauVsJets_medium_1"] > 0.5)
      &(backgrounds_df["deepTauVsEle_vvloose_1"] > 0.5)
      &(backgrounds_df["deepTauVsMu_vloose_1"] > 0.5)
      &(backgrounds_df["deepTauVsJets_medium_2"] > 0.5)
      &(backgrounds_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(backgrounds_df["deepTauVsMu_vloose_2"] > 0.5)
      &(backgrounds_df["leptonveto"] == 0)
      &(backgrounds_df["n_bjets"] == 0)
      &(backgrounds_df["trg_doubletau"] > 0.5)
      &(backgrounds_df["svfit_mass"] < 250)
  ]

  vbf_df = vbf_df[
       (vbf_df["gen_match_1"] < 6)
      &(vbf_df["os"] > 0.5)
      &(vbf_df["wt"] < 2)
      &(vbf_df["deepTauVsJets_medium_1"] > 0.5)
      &(vbf_df["deepTauVsEle_vvloose_1"] > 0.5)
      &(vbf_df["deepTauVsMu_vloose_1"] > 0.5)
      &(vbf_df["deepTauVsJets_medium_2"] > 0.5)
      &(vbf_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(vbf_df["deepTauVsMu_vloose_2"] > 0.5)
      &(vbf_df["leptonveto"] == 0)
      &(vbf_df["n_bjets"] == 0)
      &(vbf_df["trg_doubletau"] > 0.5)
      &(vbf_df["svfit_mass"] < 250)
  ]

  ggh_df = ggh_df[
       (ggh_df["gen_match_1"] < 6)
      &(ggh_df["os"] > 0.5)
      &(ggh_df["wt"] < 2)
      &(ggh_df["deepTauVsJets_medium_1"] > 0.5)
      &(ggh_df["deepTauVsEle_vvloose_1"] > 0.5)
      &(ggh_df["deepTauVsMu_vloose_1"] > 0.5)
      &(ggh_df["deepTauVsJets_medium_2"] > 0.5)
      &(ggh_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(ggh_df["deepTauVsMu_vloose_2"] > 0.5)
      &(ggh_df["leptonveto"] == 0)
      &(ggh_df["n_bjets"] == 0)
      &(ggh_df["trg_doubletau"] > 0.5)
      &(ggh_df["svfit_mass"] < 250)
  ]


  ff_df = ff_df[
      (ff_df["os"] > 0.5)
      &(ff_df["wt"] < 2)
      &(ff_df["deepTauVsJets_vvvloose_1"] > 0.5)
      &(ff_df["deepTauVsJets_medium_1"] < 0.5)
      &(ff_df["deepTauVsEle_vvloose_1"] > 0.5)
      &(ff_df["deepTauVsMu_vloose_1"] > 0.5)
      &(ff_df["deepTauVsJets_medium_2"] > 0.5)
      &(ff_df["deepTauVsEle_vvloose_2"] > 0.5)
      &(ff_df["deepTauVsMu_vloose_2"] > 0.5)
      &(ff_df["leptonveto"] == 0)
      &(ff_df["n_bjets"] == 0)
      &(ff_df["trg_doubletau"] > 0.5)
      &(ff_df["svfit_mass"] < 250)
  ]

backgrounds_df = pd.concat([backgrounds_df,ff_df], ignore_index=True)

if sign == 'even':
  backgrounds_df = backgrounds_df[(backgrounds_df["event"] % 2 == 0)]
  vbf_df=vbf_df[(vbf_df["event"] % 2 == 0)]
  ggh_df=ggh_df[(ggh_df["event"] % 2 == 0)]
if sign == 'odd':
  backgrounds_df = backgrounds_df[(backgrounds_df["event"] % 2 == 1)]
  vbf_df=vbf_df[(vbf_df["event"] % 2 == 1)]
  ggh_df=ggh_df[(ggh_df["event"] % 2 == 1)]

# normalize all classes to the background yield

sum_w_backgrounds = backgrounds_df['wt_2'].sum()
sum_w_vbf = vbf_df['wt_2'].sum()
sum_w_ggh = ggh_df['wt_2'].sum()

backgrounds_df.loc[:,"wt_3"] = backgrounds_df["wt_2"]*sum_w_backgrounds/sum_w_backgrounds
ggh_df.loc[:,"wt_3"] = ggh_df["wt_2"]*sum_w_backgrounds/sum_w_ggh
vbf_df.loc[:,"wt_3"] = vbf_df["wt_2"]*sum_w_backgrounds/sum_w_vbf

print 'yields of each class after rebalancing them:'
print 'background:', backgrounds_df['wt_3'].sum()
print 'vbf:', vbf_df['wt_3'].sum()
print 'ggh', ggh_df['wt_3'].sum()


# prepare the target labels
y_background = pd.DataFrame(np.zeros(backgrounds_df.shape[0]))
y_ggh = pd.DataFrame(np.zeros(ggh_df.shape[0]))
y_vbf = pd.DataFrame(np.zeros(vbf_df.shape[0]))

y_background+=0
y_ggh+=1
y_vbf+=2

frames = [backgrounds_df, ggh_df, vbf_df]

X = pd.concat(frames)
w = X["wt_3"]

y_frames = [y_background, y_ggh, y_vbf]

y = pd.concat(y_frames)
#.reset_index(drop=True)
y.columns = ["class"]

#make sure jpt_1 is defined only for Njets>0 evenst
X.loc[:,"jpt_1"] = (X["n_jets"]>0)*X["jpt_1"] 
# make sure mjj, jpt_2, jdeta, and dijetpt is only defined for N_jets>1 events
X.loc[:,"mjj"] = (X["n_jets"]>1)*X["mjj"] 
X.loc[:,"jdeta"] = (X["n_jets"]>1)*X["jdeta"] 
X.loc[:,"jpt_2"] = (X["n_jets"]>1)*X["jpt_2"] 
X.loc[:,"dijetpt"] = (X["n_jets"]>1)*X["dijetpt"] 



# drop all variables we dont want in the training
to_drop = sel_vars+[
           'wt_2',
           'wt_3']

print X['event'][:10]
X = X.drop(to_drop, axis=1).reset_index(drop=True)

print 'X variables after dropping some:'
print X[:10]


# function to plot 'signal' vs 'background' for a specified variable
# useful to check whether a variable gives some separation between
# signal and background states
def plot_signal_background(data1, data2, column, chan, year, sign,
                        bins=100, x_uplim=0, **kwargs):

    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5

    df1 = data1[column]
    df2 = data2[column]

    fig, ax = plt.subplots()
    df1 = df1.sample(2000, random_state=1234)
    df2 = df2.sample(2000, random_state=1234)
    low = max(min(df1.min(), df2.min()),-5)
    high = max(df1.max(), df2.max())
    if x_uplim != 0:
        ax.set_xlim(0,x_uplim)
    ax.hist(df1, bins=bins, range=(low,high), **kwargs)
    ax.hist(df2, bins=bins, range=(low,high), **kwargs)
    
    # ax.set_yscale('log')
    plt.savefig('plots/'+column+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')


# eg. here comparing mass:
for p in training_vars: plot_signal_background(backgrounds_df, vbf_df, p, chan, year, sign, bins=30)


# split into train and validation dataset 
X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
    X,
    y,
    w,
    test_size=0.2,
    random_state=123452,
    stratify=y.values,
)


# define some XGBoost parameters, unspecified will be default
# https://xgboost.readthedocs.io/en/latest////index.html
# not optimised at all, just playing by ear

xgb_params = {
    "objective": "multi:softprob",
    "max_depth": 3,
    "learning_rate": 0.05,
    "silent": 1,
    "n_estimators": 1000,
    "subsample": 0.9,
    "seed": 123451,
}

# run the training
num_round = 5 # the number of training iterations

xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(
    X_train,
    y_train,
    w_train,
    early_stopping_rounds=20, # stops the training if doesn't improve after N iterations
    eval_set=[(X_train, y_train, w_train), (X_test, y_test, w_test)],
    eval_metric = "mlogloss", # can use others
    verbose=True,
)

from xgboost import plot_importance

for i in ['weight', 'cover', 'gain']:
  plot_importance(xgb_clf,importance_type=i, xlabel=i)
  plt.savefig('plots/importance_%(i)s_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')

xgb_clf.get_booster().save_model('lowmass_training_%(year)s_%(chan)s_%(sign)s_training.model' % vars())
import pickle
with open ("lowmass_training_%(year)s_%(chan)s_%(sign)s_training.pkl" % vars(),'w') as f:
    pickle.dump(xgb_clf,f)

proba=xgb_clf.predict_proba(X_test)


from sklearn.preprocessing import label_binarize
onehot=label_binarize(y_test,classes=[0,1,2])
fpr=dict()
tpr=dict()
thresh = dict()
roc_auc=dict()
for i in range(3):
    fpr[i], tpr[i], thresh[i] = roc_curve(onehot[:,i], proba[:,i])#,None,w_test)
    roc_auc[i] = auc(fpr[i], tpr[i])#,reorder=True)
    

proba_train=xgb_clf.predict_proba(X_train)
onehot_train=label_binarize(y_train,classes=[0,1,2])
fpr_train=dict()
tpr_train=dict()
thresh_train = dict()
roc_auc_train=dict()
for i in range(3):
    fpr_train[i], tpr_train[i], thresh_train[i] = roc_curve(onehot_train[:,i], proba_train[:,i])#,None,w_test)
    roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])#,reorder=True)

print 'roc_scores:'
print roc_auc 
print roc_auc_train

# just makes the roc_curve look nicer than the default
def plot_roc_curve(fpr, tpr, auc, figname,chan,year,sign):
    fig, ax = plt.subplots(figsize=(13,8))
    ax.plot(fpr, tpr)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'ROC AUC Score: {:.3f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.savefig('plots/'+figname+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')

plot_roc_curve(fpr[0],tpr[0],roc_auc[0], 'ROC_background',chan,year,sign)
plot_roc_curve(fpr[1],tpr[1],roc_auc[1], 'ROC_ggH',chan,year,sign)
plot_roc_curve(fpr[2],tpr[2],roc_auc[2], 'ROC_VBF',chan,year,sign)

data_prob=pd.DataFrame(proba)
data_prob.columns=['Background','ggH','VBF']
data_prob["label"]=y_test.values
data_prob["weight"]=w_test.values
print data_prob[:10]

real_background=data_prob[data_prob["label"]==0]
real_ggh=data_prob[data_prob["label"]==1]
real_vbf=data_prob[data_prob["label"]==2]

print real_background[:5]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
nbins=20
#density='false'
real_background["Background"].hist(bins=nbins, histtype=u'step', lw=3, label='Background', weights = real_background["weight"])
real_ggh["Background"].hist(bins=nbins, histtype=u'step', lw=3, label='ggH', weights = real_ggh["weight"])
real_vbf["Background"].hist(bins=nbins, histtype=u'step', lw=3,  label='VBF', weights = real_vbf["weight"])

leg=plt.legend(loc=2)
plt.title("")
plt.savefig('plots/'+'BDT_score_Background'+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')
#plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
nbins=20
#density='false'
real_background["ggH"].hist(bins=nbins, histtype=u'step', lw=3, label='Background', weights = real_background["weight"])
real_ggh["ggH"].hist(bins=nbins, histtype=u'step', lw=3, label='ggH', weights = real_ggh["weight"])
real_vbf["ggH"].hist(bins=nbins, histtype=u'step', lw=3,  label='VBF', weights = real_vbf["weight"])

leg=plt.legend(loc=2)
plt.title("")
plt.savefig('plots/'+'BDT_score_ggH'+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')
#plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
nbins=20
#density='false'
real_background["VBF"].hist(bins=nbins, histtype=u'step', lw=3, label='Background', weights = real_background["weight"])
real_ggh["VBF"].hist(bins=nbins, histtype=u'step', lw=3, label='ggH', weights = real_ggh["weight"])
real_vbf["VBF"].hist(bins=nbins, histtype=u'step', lw=3,  label='VBF', weights = real_vbf["weight"])

leg=plt.legend(loc=2)
plt.title("")
plt.savefig('plots/'+'BDT_score_VBF'+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')
#plt.show()

predict=xgb_clf.predict(X_test)
print predict[:10]


def plot_confusion_matrix(y_test, y_pred, classes,
                    figname, chan, year, sign, w_test=None, normalise_by_col=False, normalise_by_row=False,
                    cmap=plt.cm.Blues):
    if w_test is not None: cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    else: cm = confusion_matrix(y_test, y_pred) 
    if normalise_by_col:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print 'Normalised efficiency confusion matrix'
    if normalise_by_row:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print 'Normalised purity confusion matrix'
    else:
        print 'Non-normalised confusion matrix'

    print cm

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='w' if cm[i, j] > thresh else 'k')

    plt.tight_layout(pad=1.4)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('plots/'+figname+'_%(chan)s_%(year)s_%(sign)s.pdf' % vars(),format='pdf')
    print 'Confusion matrix saved as {}'.format(figname)

    return None


#0 = background
#1 = ggH
#2 = VBF

from sklearn.metrics import confusion_matrix
import itertools
plot_confusion_matrix(y_test, predict, [0,1,2],
                    'purtiy_test', chan, year, sign, normalise_by_row=True, w_test=w_test)
plot_confusion_matrix(y_test, predict, [0,1,2],
                    'efficienct_test', chan, year, sign, normalise_by_col=True, w_test=w_test)


print 'printing vars:'
print X.columns

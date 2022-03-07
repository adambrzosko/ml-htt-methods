import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import logging
logger = logging.getLogger("annotate_file_inc.py")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

import numpy as np
import yaml
import os
import pickle
from array import array
import argparse
from sklearn.preprocessing import StandardScaler
    

def main(args, config, file_names):

    path = "/vols/cms/dw515/Offline/output/SM/cpprod_2018/" #check how this looks
    
    #Load models
    if full_model:
    	with open('{}/LSH_model_{}_{}_{}.pkl'
            	.format(args.model_folder, args.accuracy, args.date, args.time), 'r') as f: #ask fin about the format
            NN_model = pickle.load(f)
    else:
	with open('{}/LSH_model_{}_{}_{}.pkl' #1PRONG
            	.format(args.model_folder, args.accuracy, args.date, args.time), 'r') as f: 
            NN_model = pickle.load(f)
        with open('{}/LSH_model_{}_{}_{}.pkl' #3PRONG
            	.format(args.model_folder, args.accuracy, args.date, args.time), 'r') as f: 
            NN_model = pickle.load(f)
     
     #Open files 
     file_ = ROOT.TFile("{}/{}".format(path, sample), "UPDATE")
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception
            
     tree = file_.Get(args.tree)
     
     # Book branches for annotation
     values = []
     for variable in config["variables"]: #different values for variables
         if variable in ["dijetpt","eta_h","IC_binary_test_4_score","IC_binary_test_4_index","bpt_1",]:
             values.append(array("f", [-9999]))
         if variable in ["eta_1","eta_2","jdeta","jpt_1","jpt_2","svfit_mass","m_sv","m_vis","met","jeta_1","jeta_2","mt_tot","mt_sv",
                    "met_dphi_1","met_dphi_2","mjj","mt_1","mt_2","mt_lep","pt_1","pt_2","pt_h","pt_tt","pt_vis","pzeta","dR"]:
             values.append(array("d", [-9999]))
         if variable in ["n_jets","n_bjets","opp_sides"]:
             values.append(array("I", [0]))
         if variable not in ["zfeld","centrality","mjj_jdeta","dijetpt_pth","dijetpt_jpt1","dijetpt_pth_over_pt1",
                    "msv_mvis","msvsq_mvis","msv_sq","log_metsq_jeta2","met_jeta2","oppsides_centrality","pthsq_ptvis","msv_rec","dR_custom","rms_pt","rms_jpt","rec_sqrt_msv"]:
             tree.SetBranchAddress(variable, values[-1])
         # else:
         #     tree.SetBranchAddress("eta_h", values[-1])

     response_max_score = array("f", [-9999])
     branch_max_score = tree.Branch("{}_max_score".format(args.tag), response_max_score, "{}_max_score/F".format(args.tag))

     response_max_index = array("f", [-9999])
     branch_max_index = tree.Branch("{}_max_index".format(args.tag), response_max_index, "{}_max_index/F".format(args.tag))

     # Run the event loop
     for i_event in range(tree.GetEntries()):
         tree.GetEntry(i_event)

         # Get event number and compute response
         event = int(getattr(tree, "event"))
         m_sv = float(getattr(tree, "svfit_mass"))

         if m_sv > 0:

             values_stacked = np.hstack(values).reshape(1, len(values))
             response = classifier[event % 2].predict_proba(values_stacked, #what is this method on xgb???
                     ntree_limit=classifier[event % 2].best_iteration+1)	#need to understand how this works
             response = np.squeeze(response)

             # Find max score and index
             response_max_score[0] = -9999.0
             for i, r in enumerate(response):
                 if r > response_max_score[0]:
                     response_max_score[0] = r
                     response_max_index[0] = i
                 if i_event % 10000 == 0:
                     logger.debug('Currently on event {}'.format(i_event))


             # Fill branches
             branch_max_score.Fill()
             branch_max_index.Fill()

         else:
             response_max_score[0] = -9999.0
             response_max_index[0] = -9999.0

             # Fill branches
             branch_max_score.Fill()
             branch_max_index.Fill()

        logger.debug("Finished looping over events")

        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()

        logger.debug("Closed file")

        
            

#!/usr/bin/env python

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply XGB model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="lowmass_config.yaml",
        help="Path to training config file")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--output-folder",type=str, default="",
        help="If specified, chose where to save outputs with BDT scores"
    )
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--training", default="lowmass", help="Name of training to use.")
    parser.add_argument(
        "--channel", default="mt", help="Name of channel to annotate.")
    parser.add_argument(
        "--path", default="./", help="directory with the trees to have BDT scores added to.")
    parser.add_argument(
        "--era", default="", help="Year to use.")

    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def load_files(filelist):

    with open(filelist) as f:
        file_names = f.read().splitlines()

    return file_names


def main(args, config, file_names):

    path = args.path

    # Sanity checks
    for sample_ in file_names:
        sample = sample_.split()[0]
        if len(sample_.split())>1: 
            nsplit = int(sample_.split()[1])
        else: 
            nsplit = 0
        print(sample, nsplit)
        if not os.path.exists("{}/{}".format(path, sample)):
            logger.fatal("Input file %s does not exist.", sample)
            raise Exception

        logger.debug("Following mapping of classes to class numbers is used.")
        for i, class_ in enumerate(config["classes"]):
            logger.debug("%s : %s", i, class_)

        # Load models and preprocessing

        if args.training == "lowmass":
            with open('{}_training_{}_{}_odd_training.pkl'
                    .format(args.training, args.era,args.channel, args.training), 'r') as f:
                xgb_clf_fold1 = pickle.load(f)
            with open('{}_training_{}_{}_even_training.pkl'
                    .format(args.training, args.era,args.channel, args.training), 'r') as f:
                xgb_clf_fold0 = pickle.load(f)
        classifier = [xgb_clf_fold1, xgb_clf_fold0]


        # Open input file
        file_ = ROOT.TFile("{}/{}".format(path, sample))
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception

        tree = file_.Get(args.tree)

        # Book branches for annotation
        values = []
        var_names_list = {} # keep track of the variable indicies by name
        for variable in config["variables"]:
            if variable in ["dijetpt","eta_h","bpt_1"]:
                values.append(array("f", [-9999]))
                var_names_list[variable] = len(values)-1
            if variable in ["eta_1","eta_2","jdeta","jpt_1","jpt_2","svfit_mass","m_sv","m_vis","met","jeta_1","jeta_2","mt_tot","mt_sv",
                    "met_dphi_1","met_dphi_2","mjj","mt_1","mt_2","mt_lep","pt_1","pt_2","pt_h","pt_tt","pt_vis","pzeta","dR"]:
                values.append(array("d", [-9999]))
                var_names_list[variable] = len(values)-1
            if variable in ["n_jets","n_bjets","opp_sides","n_deepbjets"]:
                values.append(array("I", [0]))
                var_names_list[variable] = len(values)-1
            if variable not in ["zfeld","centrality","mjj_jdeta","dijetpt_pth","dijetpt_jpt1","dijetpt_pth_over_pt1",
                    "msv_mvis","msvsq_mvis","msv_sq","log_metsq_jeta2","met_jeta2","oppsides_centrality","pthsq_ptvis","msv_rec","dR_custom","rms_pt","rms_jpt","rec_sqrt_msv"]:
                tree.SetBranchAddress(variable, values[-1])
  

        response_max_score = array("f", [-9999])
        response_signal_score = array("f", [-9999])
        response_background_score = array("f", [-9999])
        response_ggh_score = array("f", [-9999])
        response_vbf_score = array("f", [-9999])
        response_max_index = array("f", [-9999])

        if tree.GetListOfBranches().FindObject("{}_max_score".format(args.tag)):
           branch_max_score = tree.GetBranch("{}_max_score".format(args.tag))
           tree.SetBranchAddress("{}_max_score".format(args.tag),response_max_score)
        else:
          branch_max_score = tree.Branch("{}_max_score".format(
              args.tag), response_max_score, "{}_max_score/F".format(
                  args.tag))
        if tree.GetListOfBranches().FindObject("{}_sig_score".format(args.tag)):
           branch_sig_score = tree.GetBranch("{}_sig_score".format(args.tag))
           tree.SetBranchAddress("{}_sig_score".format(args.tag),response_signal_score)
        else:
          branch_signal_score = tree.Branch("{}_sig_score".format(
              args.tag), response_signal_score, "{}_sig_score/F".format(
                  args.tag))

        if tree.GetListOfBranches().FindObject("{}_bkg_score".format(args.tag)):
           branch_background_score = tree.GetBranch("{}_bkg_score".format(args.tag))
           tree.SetBranchAddress("{}_bkg_score".format(args.tag),response_background_score)
        else:
          branch_background_score = tree.Branch("{}_bkg_score".format(
              args.tag), response_background_score, "{}_bkg_score/F".format(
                  args.tag))

        if tree.GetListOfBranches().FindObject("{}_ggh_score".format(args.tag)):
           branch_ggh_score = tree.GetBranch("{}_ggh_score".format(args.tag))
           tree.SetBranchAddress("{}_ggh_score".format(args.tag),response_ggh_score)
        else:
          branch_ggh_score = tree.Branch("{}_ggh_score".format(
              args.tag), response_ggh_score, "{}_ggh_score/F".format(
                  args.tag))

        if tree.GetListOfBranches().FindObject("{}_vbf_score".format(args.tag)):
           branch_vbf_score = tree.GetBranch("{}_vbf_score".format(args.tag))
           tree.SetBranchAddress("{}_vbf_score".format(args.tag),response_vbf_score)
        else:
          branch_vbf_score = tree.Branch("{}_vbf_score".format(
              args.tag), response_vbf_score, "{}_vbf_score/F".format(
                  args.tag))

        if tree.GetListOfBranches().FindObject("{}_max_index".format(args.tag)):
           branch_max_index = tree.GetBranch("{}_max_index".format(args.tag))
           tree.SetBranchAddress("{}_max_index".format(args.tag),response_max_index)
        else: 
          branch_max_index = tree.Branch("{}_max_index".format(
              args.tag), response_max_index, "{}_max_index/F".format(
                  args.tag))

###############################
        
        if args.output_folder != "":
            fileout_ = ROOT.TFile(
                "{}/{}".format(args.output_folder, sample.replace('.root','_{}.root'.format(nsplit))), 
                "RECREATE"
            )
        else:
            fileout_ = ROOT.TFile(
                "{}/{}".format(path, sample.replace('.root','_{}.root'.format(nsplit))), 
                "RECREATE"
            )
        newtree=tree.CloneTree(0)

        # Run the event loop

        perjobs=300000
        mini=nsplit*300000
        maxi=(nsplit+1)*300000
        entries=tree.GetEntries()
        if maxi > entries: maxi=entries


        for i_event in range(mini,maxi):

            tree.GetEntry(i_event)

            # Get event number and compute response
            event = int(getattr(tree, "event"))
            if tree.GetListOfBranches().FindObject("svfit_mass"):
              m_sv = float(getattr(tree, "svfit_mass"))
            else: m_sv=-1

            if m_sv > 0:

                values_stacked = np.hstack(values).reshape(1, len(values))

                # convert njets into int - would need to do this for both int variables e.g nbjets
                values_stacked[0][var_names_list['n_jets']] = int(values_stacked[0][var_names_list['n_jets']])

                # for all single jet variables we want to set their values to 0 if n_jets is < 1
                # and similar for 2jet variables if n_jets<2
                # note if you add additional variables then this needs updating
                vars_1jet = ['jpt_1']
                vars_2jet = ['jpt_2','mjj','dijetpt','jdeta']

                njets = values_stacked[0][var_names_list['n_jets']] 
  
                if njets < 1:
                  for v in vars_1jet: values_stacked[0][var_names_list[v]] = 0.               
                if njets < 2:
                  for v in vars_2jet: values_stacked[0][var_names_list[v]] = 0.               
 

                # even events use classifier[0] (trained on odd events), odd events use classifier[1] (trained on even events)
                response = classifier[event % 2].predict_proba(values_stacked,
                        ntree_limit=classifier[event % 2].best_iteration+1)
                response = np.squeeze(response)

                response_background_score[0] = response[0] # background score
                response_signal_score[0] = 1.-response[0] # overall signal score score
                response_ggh_score[0] = response[1] # background score
                response_vbf_score[0] = response[2] # background score
                # Find max score and index
                response_max_score[0] = -9999.0
                for i, r in enumerate(response):
                    if r > response_max_score[0]:
                        response_max_score[0] = r
                        response_max_index[0] = i
                if i_event % 10000 == 0:
                    logger.debug('Currently on event {}'.format(i_event))

                #print 'scores:'
                #print response_signal_score[0], response_background_score[0], response_ggh_score[0], response_vbf_score[0] 
                #print response_max_score[0],  response_max_index[0] 

                # Fill branches
                newtree.Fill()

            else:
                response_max_score[0] = -9999.0
                response_max_index[0] = -9999.0

                # Fill branches
                newtree.Fill()

        logger.debug("Finished looping over events")

        # Write everything to file
        fileout_.cd()
        newtree.Show(1)
        newtree.Write("ntuple",ROOT.TObject.kWriteDelete)
        fileout_.Close()
        file_.Close()

        logger.debug("Closed file")

if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    file_names = load_files(args.input)
    main(args, config, file_names)

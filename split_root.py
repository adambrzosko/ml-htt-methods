#!/usr/bin/env python
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
import sys

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
        default="mt_xgb_training_config.yaml",
        help="Path to training config file")
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default="ntuple",
        help="Prefix of directories in ROOT file to be annotated.")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["multi_fold1_cpsm_mt_JHU_xgb.pkl", "multi_fold0_cpsm_mt_JHU_xgb.pkl"],
        help=
        "Keras models to be used for the annotation. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the training is independent from the application."
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        nargs="+",
        default=[
            "tt_training_9May_tests/tt_fold1_scaler.pkl",
            "tt_training_9May_tests/tt_fold0_scaler.pkl"
        ],
        help=
        "Data preprocessing to be used. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the preprocessing is independent for the folds."
    )
    
    parser.add_argument(
        "--output-folder",type=str, default="",
        help="If specified, chose where to save outputs with BDT scores"
    )
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--training", default="JHU", help="Name of training to use.")
    parser.add_argument(
        "--mjj", default="high", help="mjj training to use.")
    parser.add_argument(
        "--channel", default="mt", help="Name of channel to annotate.")
    parser.add_argument(
        "--model_folder", default="mt_training_10May_mjj_jdeta_dijetpt/", help="Folder name where trained model is.")
    parser.add_argument(
        "--path", default="./", help="directory with the trees to have BDT scores added to.")
    parser.add_argument(
        "--era", default="", help="Year to use.")

    return parser.parse_args()

def main(args, file_names):

    path = args.path
    nsplit = 0
    file_ = ROOT.TFile("{}".format(file_names))
    if file_ == None:
        logger.fatal("File %s is not existent.", sample)
        raise Exception
    tree = file_.Get(args.tree)

    for sample_ in range(tree.GetEntries()//30000+1):
        file_ = ROOT.TFile("{}".format(file_names))
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception
        tree = file_.Get(args.tree)
     
        if args.output_folder != "":
            fileout_ = ROOT.TFile(
                "split/{}".format(file_names.replace('.root','_{}.root'.format(nsplit))), 
                "CREATE"
            )
        else:
            fileout_ = ROOT.TFile(
                "split/{}".format(file_names.replace('.root','_{}.root'.format(nsplit))), 
                "CREATE"
            )
        newtree=tree.CloneTree(0)

        # Run the event loop

        perjobs=30000
        mini=nsplit*30000
        maxi=(nsplit+1)*30000
        entries=tree.GetEntries()
        if maxi > entries: maxi=entries


        for i_event in range(mini,maxi):
            #print(i_event)
            tree.GetEntry(i_event)

            # Get event number and compute response
            event = int(getattr(tree, "event"))
            if tree.GetListOfBranches().FindObject("svfit_mass"):
              m_sv = float(getattr(tree, "svfit_mass"))
            else: m_sv=-1

            #print m_sv

            if m_sv > 0:
                newtree.Fill()

            else:
                newtree.Fill()


        logger.debug("Finished looping over events")

        # Write everything to file
        fileout_.cd()
        #newtree.Show(1)
        newtree.Write("ntuple",ROOT.TObject.kWriteDelete)
        fileout_.Close()
        file_.Close()
        nsplit = nsplit+1

        logger.debug("Closed file")
        
def parse_config(filename):
    return pickle.load(open(filename, "r"))

def load_files(filelist):
    with open(filelist) as f:
        file_names = f.read().splitlines()
    return file_names

        
args = parse_arguments()
file_names = "%s"%sys.argv[1]#load_files(args.input)
main(args, file_names)

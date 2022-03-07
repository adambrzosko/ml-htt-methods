"""
1. Load a ROOT file
2. Run the preprocessing on it - dataframeinit, dataframemod, imgen
3. Run analyse model pointing to this data
4. Annotate branches

Have a config file that sets configs
"""

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

from pipeline_object import pipeline 
import yaml
    
def parse_arguments():
    """Tool for easy argument calling in functions"""
    parser = argparse.ArgumentParser(
        description="Apply NN model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="nn_training_config.yaml",	#CREATE THIS
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
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--channel", default="mt", help="Name of channel to annotate.")
    parser.add_argument(
        "--model_folder", default="mt_training_10May_mjj_jdeta_dijetpt/", help="Folder name where trained model is.") #change this
    parser.add_argument(
        "--era", default="", help="Year to use.")
    parser.add_argument(
        "loadpath", default="/vols/cms/dw515/outputs/SM/MPhysNtuples", help="Path to files to use.")
    parser.add_argument(
        "savepath", default="/vols/cms/ab7018/Masters2021/Annotation", help="Path to save output files.")
    return parser.parse_args()

def parse_config(filename):
    """For loading the config file with yaml"""				
    return yaml.load(open(filename, "r"))
    
#def load_files(filelist):
#    """For loading the data files"""
#    with open(filelist) as f:
#        file_names = f.read().splitlines()
#        # file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
#    return file_names
    
    
def main(args, config):
    
    #create a jesmond
    jesmond = pipeline(args.loadpath, args.savepath) #ideally should take vars from config file
    
    #load root files for preprocessing
    jesmond.load_root_files() #change this in the pipeline so its flexible and takes from loadpath
    
    #do preprocessing
    """ Before I noticed the meta commands lol - still need edits to original
    split_full_by_dm(jesmond) 
    energyfinder_2(jesmond, config["momvariablenames1"])
    calc_mass(jesmond, config["momvariablenames1"])
    tau_eta(jesmond, config["momvariablenames1"])
    ang_var(jesmond, config["momvariablenames1"], config["momvariablenames2"], config["particlename"]) 
    get_fourmomenta_lists(jesmond)
    phi_eta_find(jesmond) 
    rotate_dataframe(jesmond) 
    drop_variables(jesmond)            
    drop_variables_2(jesmond)          
    create_featuredesc(jesmond)           
    generate_grids(jesmond)           
    """
    split_full_by_dm(jesmond)
    modify_dataframe(jesmond)
    imvar_jesmond = create_imvar_dataframe(jesmond)
    #jesmond.clear_dataframe()          
    generate_datasets(jesmond, imvar_jesmond, args.savepath)  #modify this not to save but create 
    
    #put this in the test form - may need to do it later, want to preserve the "event" variable (id)
    test_inputs = 
    
    #load our model
    model = keras.models.load_model(args.model_folder)
    
    #open original file
    file_ = ROOT.TFile("{}".format(args.loadpath), "UPDATE")
    if file_ == None:
        logger.fatal("File %s is not existent.", sample)
        raise Exception
    tree = file_.Get(args.tree)
    
    
    #Book branches for annotation
    response_scores = array("f", [0,0,0]) #want array instead of float - troubles??
    branch_scores = tree.Branch("{}_scores".format(
        args.tag), response_scores, "{}_scores/F".format(args.tag))

    
    # Run the event loop
    for i_event in range(tree.GetEntries()):
        tree.GetEntry(i_event)

        # Get event number and compute response
        event = int(getattr(tree, "event"))
        m_sv = float(getattr(tree, "svfit_mass"))

        if m_sv > 0:
            response_scores = model.predict_proba(test_inputs[test_inputs["event"]==event]) #have to properly feed the whole event with the images
            
            # Fill branches
            branch_max_score.Fill()

        else:
            response_scores[i] = [0,0,0]

            # Fill branches
            branch_scores.Fill()

        logger.debug("Finished looping over events")

        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()

        logger.debug("Closed file")        
    
    
                      
if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    #file_names = load_files(args.input)
    main(args, config)



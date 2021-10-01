#!/cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw/CMSSW_10_2_19/external/slc7_amd64_gcc700/bin/python

import fit_functions as ff
import sys

file_name='%s'%sys.argv[1]
ff.AddBDT_EvenOdd_RhoRho(file_name, 'BDTOddEven')

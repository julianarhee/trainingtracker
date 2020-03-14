#!/usr/bin/env python2

import matplotlib as mpl
mpl.use('agg')
import os
import glob
import json
import pymworks
import re
import datautils
import copy
import math
import time
import optparse
import sys

import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import cPickle as pkl
from cPickle import PicklingError

import utils as util
import process_datafiles as processd

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/behavior-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/behavior-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default=None, 
                      help='Animal ID')
    parser.add_option('-c', '--cohort', action='store', dest='cohort', default=None, 
                      help='cohort, e.g, AL)')

    parser.add_option('-p', '--paradigm', action='store', dest='paradigm', default='threeport', 
            help='paradigm used (default:  threeport')

    parser.add_option('-n', '--nproc', action='store', dest='n_processes', default=1, 
            help='N processes to use (default:  1)')


    parser.add_option('--plot-session', action='store_true', dest='plot_each_session', default=False, 
            help='flag to plot performance for each sessions')
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
            help='flag to reprocess sessions anew')
    parser.add_option('--meta', action='store_true', dest='create_meta', default=False, 
            help='flag to recreate metadata (if adding new datafiles)')
 
    parser.add_option('--trials', action='store_true', dest='process_trials', default=false, 
            help='flag to do trial-parsing step for each session (if true, also makes dataframes)')

    parser.add_option('--dfs', action='store_true', dest='make_dataframe', default=false, 
            help='flag to do only create dataframe from processed session data')

    (options, args) = parser.parse_args(options)

    return options


def main(options):

    opts = extract_options(options)
    rootdir = opts.rootdir
    paradigm = opts.paradigm
    animalid = opts.animalid
    cohort = opts.cohort
    if animalid in [None, 'None']:
        assert cohort in os.listdir(os.path.join(rootdir, paradigm, 'cohort_data')), "COHORT NOT FOUND: %s" % str(cohort)
        print("processing all animald in cohort: %s" % cohort)
        process_cohort = True
    else:
        process_cohort = False

    n_processes = int(opts.n_processes)
    print("***running on %i processes" % n_processes)
 
    create_new = opts.create_new
    create_meta = opts.create_meta 
    plot_each_session = opts.plot_each_session
    process_trials = opts.process_trials
    make_dataframe = opts.make_dataframe

    # Set all output dirs
    cohort_dirs = [os.path.split(p)[0] for p in glob.glob(os.path.join(rootdir, paradigm, 'cohort_data', 'A*', 'raw'))]
    cohort_list = [re.search('(\D{2})', os.path.split(pf)[-1]).group(0) for pf in cohort_dirs]

    for curr_cohort in cohort_list:
        working_dir = os.path.join(rootdir, paradigm, 'cohort_data', curr_cohort)   
        output_dir = os.path.join(working_dir, 'processed')
        output_figdir = os.path.join(output_dir, 'figures')
        output_datadir = os.path.join(output_dir, 'data')
        if not os.path.exists(output_figdir): os.makedirs(output_figdir)
        if not os.path.exists(output_datadir): os.makedirs(output_datadir)
            
    #### Load metadata
    metadata = util.get_metadata(paradigm, create_meta=create_meta)

    if parse_trials:
        make_dataframe = False # already doing this by default
        A = None
        if process_cohort:
            print("-- processing cohort: %s" % cohort)
            #### Process current animal
            for animalid, session_meta in metadata[metadata['cohort']==cohort].groupby(['animalid']):
                print('[%s] - starting processing...' % animalid)
                A = processd.process_sessions_for_animal(animalid, session_meta, paradigm=paradigm, n_processes=n_processes,
                                              create_new=create_new, plot_each_session=plot_each_session)
                print("[%s] - done processing! -" % animalid)

                print("[%s] - creating dataframe" % animalid)
                df, new_s = processd.get_animal_df(animalid, paradigm, metadata, create_new=True, rootdir=rootdir)     
        else:
            print('[%s] - starting processing...' % animalid)
            A = processd.process_sessions_for_animal(animalid, metadata, paradigm=paradigm, n_processes=n_processes,
                                              create_new=create_new, plot_each_session=plot_each_session)
            print("[%s] - done processing! -" % animalid)
            
            print("[%s] - creating dataframe" % animalid)
            df, new_s = processd.get_animal_df(animalid, paradigm, metadata, create_new=True, rootdir=rootdir)    
      
    if make_dataframe:
        if process_cohort:
            #### Process current animal
            for animalid, session_meta in metadata[metadata['cohort']==cohort].groupby(['animalid']):
                print("[%s] - creating dataframe" % animalid)
                df, new_s = processd.get_animal_df(animalid, paradigm, session_meta, create_new=True, rootdir=rootdir)     
        else:            
            print("[%s] - creating dataframe" % animalid)
            df, new_s = processd.get_animal_df(animalid, paradigm, metadata, create_new=True, rootdir=rootdir)    
 

    print("~~~ done! ~~~")
     
    return A

if __name__ == '__main__':
    main(sys.argv[1:])


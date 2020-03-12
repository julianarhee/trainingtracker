#!/usr/bin/env python2

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

from scipy import stats
import parse_behavior as pb

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def get_metadata(paradigm, root='/n/coxfs01/behavior-data', create_meta=False):
    meta_datafile = os.path.join(root, paradigm, 'metadata.pkl')

    reload_meta = False
    if os.path.exists(meta_datafile):
        print("Loading existing metadata...")
        with open(meta_datafile, 'rb') as f:
            metadata = pkl.load(f)
    else:
        reload_meta = True

    if create_meta or reload_meta:
        print("Creating new metadata...")
        ### All raw datafiles
        raw_fns = glob.glob(os.path.join(root, paradigm, 'cohort_data', 'A*', 'raw', '*.mwk'))

        #### Get all animals and sessions
        metadata = pd.concat([pd.DataFrame({'animalid': pb.parse_datafile_name(fn)[0],
                                          'session': int(pb.parse_datafile_name(fn)[1]),
                                          'datasource': fn, 
                                          'cohort': pb.parse_datafile_name(fn)[0][0:2]}, index=[i]) \
                                           for i, fn in enumerate(raw_fns)], axis=0)

        with open(meta_datafile, 'wb') as f:
            pkl.dump(metadata, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    return metadata


def parse_datafiles_for_animal(animalid, metadata, n_processes=1,
                               paradigm='threeport', create_new=False,
                               root='/n/coxfs01/behavior-data'):

    # Set experiment parsing vars and params:
    response_types = ['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore']
    outcome_types = outcome_types = ['success', 'ignore', 'failure']
    ignore_flags = None

    print("***********************************************")
    print("ANIMAL:  %s" % animalid)
    print("***********************************************")
    
    cohort = metadata[metadata.animalid==animalid]['cohort'].unique()[0]

    # Get current animal session info:
    A = pb.Animal(animalid=animalid, experiment=paradigm) #, output_datadir=output_datadir)
    curr_processed_dir = os.path.join(root, paradigm, 'cohort_data', cohort, 'processed')

    # --- Create or load animal datafile:
    animal_datafile = os.path.join(curr_processed_dir, 'data', '%s.pkl' % animalid)
    print("outfile: %s" % animal_datafile)

    # --- Check if processed file exists -- load or create new.
    create_new = False
    if os.path.exists(animal_datafile):
        try:
            with open(animal_datafile, 'rb') as f:
                A = pkl.load(f)   
        except EOFError:
            create_new = True

    # --- Process new datafiles / sessions:
    all_sessions = metadata[metadata.animalid==animalid]['session'].values
    old_sessions = [skey for skey, sobject in A.sessions.items() if sobject is not None]
    print("[%s]: Found %i processed sessions." % (animalid, len(old_sessions)))
    new_sessions = [s for s in all_sessions if s not in old_sessions]
    print("[%s]: There are %i out of %i found session datafiles to process." % (A.animalid, len(new_sessions), len(all_sessions)))

    # Process all new sessions:
    session_meta = metadata[(metadata.animalid==animalid)] # & (metadata.session==session)]
    processed_sessions = pb.process_sessions_mp(new_sessions, session_meta,
                                             #dst_dir=output_figdir,
                                             nprocesses=n_processes,
                                             ignore_flags=ignore_flags,
                                             response_types=response_types,
                                             outcome_types=outcome_types, create_new=create_new)

    # Update animal sessions dict:
    for datestr, S in processed_sessions.items():
        A.sessions.update({datestr: S})

    # Save to disk:
    try:
        with open(animal_datafile, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)
    except PicklingError:
        print("Unable to pkl: New sessions are not the same class as old sessions.")
        print("Reprocessing %i old sessions..." % len(processed_sessions))
        for session in old_sessions:
            curr_sessionmeta = session_meta[session_meta.session==session] #session_info[datestr]
            S = pd.process_session(curr_sessionmeta)
            A.sessions[session] = S

        with open(animal_datafile, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("[%s] ~~~ processing complete! ~~~" % A.animalid)

    return A

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/behavior-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/behavior-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', 
                      help='Animal ID')
    parser.add_option('-p', '--paradigm', action='store', dest='paradigm', default='threeport', 
            help='paradigm used (default:  threeport')

    (options, args) = parser.parse_args(options)

    return options


def main(options):

    opts = extract_options(options)
    root = opts.rootdir
    paradigm = opts.paradigm
    animalid = opts.animalid
    n_processes = int(opt.n_processes)
    create_new = opts.create_new

    # Set all output dirs
    cohort_dirs = [os.path.split(p)[0] for p in glob.glob(os.path.join(root, paradigm, 'cohort_data', 'A*', 'raw'))]
    cohort_list = [re.search('(\D{2})', os.path.split(pf)[-1]).group(0) for pf in cohort_dirs]

    for cohort in cohort_list:
        working_dir = os.path.join(root, paradigm, 'cohort_data', cohort)   
        output_dir = os.path.join(working_dir, 'processed')
        output_figdir = os.path.join(output_dir, 'figures')
        output_datadir = os.path.join(output_dir, 'data')
        if not os.path.exists(output_figdir): os.makedirs(output_figdir)
        if not os.path.exists(output_datadir): os.makedirs(output_datadir)

            
    #### Load metadata
    metadata = get_metadata(paradigm)

    A = parse_datafiles_for_animal(animalid, metadata, paradigm=paradigm, n_processes=n_processes,
                                  create_new=create_new)

    print("[%s] - done processing! -" % animalid)

    return A

if __name__ == '__main__':
    main(sys.arvg[1:])


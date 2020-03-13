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

from scipy import stats
#import parse_behavior as pb
import utils as util

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def get_metadata(paradigm, rootdir='/n/coxfs01/behavior-data', create_meta=False):
    meta_datafile = os.path.join(rootdir, paradigm, 'metadata.pkl')

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
        raw_fns = glob.glob(os.path.join(rootdir, paradigm, 'cohort_data', 'A*', 'raw', '*.mwk'))

        #### Get all animals and sessions
        metadata = pd.concat([pd.DataFrame({'animalid': util.parse_datafile_name(fn)[0],
                                          'session': int(util.parse_datafile_name(fn)[1]),
                                          'datasource': fn, 
                                          'cohort': util.parse_datafile_name(fn)[0][0:2]}, index=[i]) \
                                           for i, fn in enumerate(raw_fns)], axis=0)

        with open(meta_datafile, 'wb') as f:
            pkl.dump(metadata, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    return metadata

def load_animal_data(animalid, paradigm, metadata, rootdir='/n/coxfs01/behavior-data'):

    # --- Create or load animal datafile:
#     cohort = str(re.findall('(\D+)', aimalid)[0])
#     curr_processed_dir = os.path.join(root, paradigm, 'cohort_data', cohort, 'processed')
#     animal_datafile = os.path.join(curr_processed_dir, 'data', '%s.pkl' % animalid)
#     print("outfile: %s" % animal_datafile)

    # --- Check if processed file exists -- load or create new.
    A = util.Animal(animalid=animalid, experiment=paradigm, rootdir=rootdir)
    create_new = False
    reload_data = False
    if os.path.exists(A.path):
        try:
            with open(A.path, 'rb') as f:
                A = pkl.load(f)   
        except EOFError:
            create_new = True
        except ImportError:
            reload_data = True
            create_new = False
    print(create_new, reload_data)

    print("outfile: %s" % A.path)
    
    # --- Process new datafiles / sessions:
    all_sessions = metadata[metadata.animalid==animalid]['session'].unique() #.values
    old_sessions = [int(skey) for skey, sobject in A.sessions.items() if sobject is not None]
    none_sessions = [int(skey) for skey, sobject in A.sessions.items() if sobject is None]
    print("[%s]: Loaded %i processed sessions (+%i are None)." % (animalid, len(old_sessions), len(none_sessions)))
    new_sessions = [s for s in all_sessions if s not in old_sessions and s not in none_sessions]
    print("[%s]: Found %i out of %i sessions to process." % (A.animalid, len(new_sessions), len(all_sessions)))
    
    return A, new_sessions


def process_sessions_for_animal(animalid, metadata, n_processes=1, plot_each_session=True,
                               paradigm='threeport', create_new=False,
                               rootdir='/n/coxfs01/behavior-data'):

    # Set experiment parsing vars and params:
    response_types = ['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore']
    outcome_types = outcome_types = ['success', 'ignore', 'failure']
    ignore_flags = None

    print("***********************************************")
    print("ANIMAL:  %s" % animalid)
    print("***********************************************")
    
    # --- Get current animal session info:
    session_meta = metadata[metadata.animalid==animalid].copy()
    A, new_sessions = load_animal_data(animalid, paradigm, session_meta, rootdir=rootdir)

    # --- process.
    start_t = time.time()
    session_meta = metadata[(metadata.animalid==animalid)] # & (metadata.session==session)]
    processed_sessions = util.process_sessions_mp(new_sessions, session_meta,
                                             nprocesses=n_processes, plot_each_session=plot_each_session,
                                             ignore_flags=ignore_flags,
                                             response_types=response_types,
                                             outcome_types=outcome_types, create_new=create_new)
    end_t = time.time() - start_t
    print("--> Elapsed time: {0:.2f}sec".format(end_t))

    # --- Update animal sessions dict:
    for datestr, S in processed_sessions.items():
        A.sessions.update({datestr: S})

    # --- Save to disk:
    try:
        with open(A.path, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)
    except PicklingError:
        print("Unable to pkl: New sessions are not the same class as old sessions.")
        print("Reprocessing %i old sessions..." % len(processed_sessions))
        for session in old_sessions:
            curr_sessionmeta = session_meta[session_meta.session==session] #session_info[datestr]
            S = pd.process_session(curr_sessionmeta)
            A.sessions[session] = S
        with open(A.path, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("[%s] ~~~ processing complete! ~~~" % A.animalid)

    return A



def get_animal_df(animalid, paradigm, metadata, create_new=False, rootdir='/n/coxfs01/behavior-data'):
    
    # Check for dataframe
    outdir = os.path.join(rootdir, paradigm, 'processed', 'data')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(outdir)
    d_outfile = os.path.join(outdir, 'df_%s.pkl' % animalid)
    print(d_outfile)
    
    reload_df = False
    if os.path.exists(d_outfile) and create_new is False:
        print("... loading existing df")
        with open(d_outfile, 'rb') as f:
            df = pkl.load(f)
        new_sessions=[]
    else:
        reload_df = True

    if reload_df:
        print("... making dataframe for %s" % animalid)
        df, new_sessions, no_trials = sessiondata_to_df(animalid, paradigm, metadata, rootdir=rootdir)
        with open(d_outfile, 'wb') as f:
            pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)

    return df, new_sessions, no_trials


def sessiondata_to_df(animalid, paradigm, metadata, rootdir='/n/coxfs01/behavior-data',
                      event_names=['outcome', 'time', 'response', 'no_feedback', 'response_time', 'duration', 
                                    'pos_x', 'pos_y', 'rotation', 'size', 'depth_rotation', 'light_position', 'name']):
    
    A, new_sessions = load_animal_data(animalid, paradigm, metadata, rootdir=rootdir)
    if len(new_sessions)> 0:
        print("[%s] There are %i new sessions to analyze..." % (animalid, len(new_sessions)))

    dflist = []
    no_trials = []
    for si, (sess, s) in enumerate(A.sessions.items()):
        if si % 20 == 0:
            print("... adding %i of %i sessions." % (int(si+1), len(A.sessions)))

        if s is None or s.trials is None or len(s.trials)==0:
            no_trials.append(sess)
            continue

        tmpd=[]
        for ti, trial in enumerate(s.trials):
            currvalues = dict((k, 0) for k in event_names)
            got_keys = [k for k, v in trial.items() if k in event_names]
            for k in got_keys:
                if isinstance(trial[k], tuple): #len(trial[k])>1:
                    currvalues[k] = '_'.join([str(i) for i in trial[k]])
                else:
                    currvalues[k] = trial[k]
            tmpd.append(pd.DataFrame(currvalues, index=[ti]))
        tmpdf = pd.concat(tmpd, axis=0)
        tmpdf['response_time'] = (tmpdf['response_time']-tmpdf['time']) / 1E6
        tmpdf['session'] = [sess for _ in np.arange(0, len(tmpd))]
        dflist.append(tmpdf)

    df = pd.concat(dflist, axis=0)
    print('%i sessions have no trials' % len(no_trials))


    return df, new_sessions, no_trials

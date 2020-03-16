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


def get_sessions_for_animal(animalid, metadata, n_processes=1, plot_each_session=False,
                               paradigm='threeport', create_new=False,
                               response_types=['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore'],
                               outcome_types=['success', 'ignore', 'failure'],
                               ignore_flags=None,
                               rootdir='/n/coxfs01/behavior-data'):

    # Set experiment parsing vars and params:
#    response_types = ['Announce_AcquirePort1', 'Announce_AcquirePort3', 'ignore']
#    outcome_types = outcome_types = ['success', 'ignore', 'failure']
#    ignore_flags = None
#
    print("***********************************************")
    print("ANIMAL:  %s" % animalid)
    print("***********************************************")
    
    # --- Get current animal session info:
    session_meta = metadata[metadata['animalid']==animalid].copy()
    A, new_sessions = util.load_animal_data(animalid, paradigm, session_meta, rootdir=rootdir)

    # --- process.
    start_t = time.time()
    session_meta = metadata[(metadata['animalid']==animalid)] # & (metadata.session==session)]
    if create_new:
        print("!!! (Re)processing all sessions.")
        new_sessions = session_meta['session'].unique()
    processed_sessions = util.get_sessions_mp(new_sessions, session_meta,
                                             n_processes=n_processes, plot_each_session=plot_each_session,
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
            S = util.get_session_data(curr_sessionmeta)
            A.sessions[session] = S
        with open(A.path, 'wb') as f:
            pkl.dump(A, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("[%s] ~~~ processing complete! ~~~" % A.animalid)

    return A



def get_animal_df(animalid, paradigm, metadata, create_new=False, rootdir='/n/coxfs01/behavior-data'):
   
    new_sessions=[]
    no_trials = []
    df = None

    # Check for dataframe
    outdir = os.path.join(rootdir, paradigm, 'processed', 'data')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    d_outfile = os.path.join(outdir, 'df_%s.pkl' % animalid)
    
    reload_df = False
    if os.path.exists(d_outfile) and create_new is False:
        print("... loading existing df")
        with open(d_outfile, 'rb') as f:
            df = pkl.load(f)
    else:
        reload_df = True

    if reload_df:
        print("... making dataframe for %s" % animalid)
        A, new_sessions = util.format_animal_data(animalid, paradigm, metadata, rootdir=rootdir)
        df = util.animal_data_to_dataframe(A)
        #df, new_sessions, no_trials = sessiondata_to_df(animalid, paradigm, metadata, rootdir=rootdir)
        if df is None:
            return None, None
        else:
            with open(d_outfile, 'wb') as f:
                pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)

    return df, new_sessions


def sessiondata_to_df(animalid, paradigm, metadata, rootdir='/n/coxfs01/behavior-data',
                      event_names=['outcome', 'time', 'response', 'no_feedback', 'response_time', 'duration', 
                                    'pos_x', 'pos_y', 'rotation', 'size', 'depth_rotation', 'light_position', 'name']):
    
    A, new_sessions = util.load_animal_data(animalid, paradigm, metadata, rootdir=rootdir)
    if len(new_sessions)> 0:
        print("[%s] There are %i new sessions to analyze..." % (animalid, len(new_sessions)))

    df = None
    sessionlist = []
    no_trials = []
    for si, (session, sessionobj) in enumerate(A.sessions.items()):
        if si % 20 == 0:
            print("... adding %i of %i sessions." % (int(si+1), len(A.sessions)))

        if sessionobj is None or sessionobj.trials is None or len(sessionobj.trials)==0:
            no_trials.append(session)
            continue

        trialdf = util.trialdict_to_dataframe(sessionobj.trials, session=session)
        if trialdf is not None:
            sessionlist.append(trialdf)
            
    if len(sessionlist) > 0:
        df = pd.concat(sessionlist, axis=0)
    print('%i sessions have no trials' % len(no_trials))


    return df, new_sessions, no_trials


#!/usr/bin/env python
# simple data analysis attempt for within-session
# Nov 24, 2015 - jyr
# Go/NoGo task, new protocol with optimizers.

# un-crap-ify the crap code from 2013...

# For "logging" mode during debugging/testing new code:
# (These need to be done BEFORE importing pymworks, etc.)

# In [1]: import logging
# In [2]: logging.basicConfig(level=logging.DEBUG)
# In [3]: logging.debug('foo')
# DEBUG:root:foo


import os
import sys

import pymworks
import cPickle as pkl

import itertools
import functools

import logging
#import datautils
import numpy as np

import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


###############################################################################
# GENERALLY USEFUL FUNCTIONS
###############################################################################

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def save_dict(var, savepath, fname):
    '''
    Check if dict exists. Append and save, if so. Otherwise, just save.

    var : dict
        dict to save

    savepath : string
        path to save dict (or to look for existent dict)

    fname: string
        name of .pkl dict
    '''
    fn = os.path.join(savepath, fname)
    if not os.path.exists(fn):
        P = var
    else:
        f = open(fn,'rb')
        P = pkl.load(f)
        P.update(var)
        f.close()

    f = open(fn, 'wb')
    pkl.dump(P, f)
    f.close()


###############################################################################
# SESSION ANALYSIS FUNCTIONS:
###############################################################################

def get_session_list(input_path):
    '''
    Returns a dict with animal names as keys, all found sessions as values
        e.g., {'JR003E': ['JR003E_20151120.mwk', 'JR003E_20151123.mwk']}

    input_path : string 
        path to dict containing data files (organized by subject)
        each subject folder should contain .mwk files for all sessions.
    '''
    animals = os.listdir(input_path)
    F = dict()
    for a in animals:
        F[a] = os.listdir(os.path.join(input_path, a))
        F[a] = sorted([f for f in F[a] if os.path.splitext(f)[1]=='.mwk'])
    print "Session list: ", F

    return F


def get_new_sessions(outpath, session_dict):
    '''
    Returns a dict with animal names as keys, all *new* sessions as values
        e.g., {'JR003E': ['JR003E_20151120.mwk', 'JR003E_20151123.mwk']}

    outpath : string 
        path to dict containing previously saved processed data for each subject

    session_dict : dict
        dict containing all found sessions (values) for all animals (keys)
        output of get_new_sessions()
    '''
    new_sessions = dict()                                                               
    if not os.path.exists(outpath):                                             # initiate dict if data from this expt has never been processed before
        print "First analysis!"
        os.makedirs(outpath)
        for animal in session_dict.keys():
            new_sessions[animal] = session_dict[animal]
            print "Adding %s to animal-session dict." % animal                         
    else:                                                                       # only chopse new stuff to be analyzed (animals and/or sessions)
        for animal in session_dict.keys():
            outfiles = [os.path.join(outpath, i) for i in os.listdir(outpath)\
                            if animal in i]                                     # get processed files for loading
            print "Existent outfiles: ", outfiles
            if outfiles:                                                        # only use new sessions if animal has been processed
                fn_summ = open([i for i in outfiles if '_summary.pkl' in i][0],'rb')            
                summary = pkl.load(fn_summ)
                new_sessions[animal] = [s for s in session_dict[animal] if s not in summary.keys()]
                if new_sessions[animal]:
                    print "Analyzing sessions: %s,\n%s" % (animal, new_sessions[animal])
                else:
                    print "No new sessions found."
                fn_summ.close()
            else:
                print "New animal %s detected..." % animal                      # get all sessions if new animal
                new_sessions[animal] = session_dict[animal]

    return new_sessions



###############################################################################
# SESSION PARSING FUNCTIONS:
###############################################################################

def get_session_stats():
    stats = {}
    # for n in ('session_correct_lick', 'session_bad_lick', 'session_bad_ignore', \
    #         'session_correct_ignore', 'session_target_trialcounter', \
    #         'session_distractor_trialcounter', \
    #         'trials_in_session'): #, 'aborted_counter'):
    #     stats[n] = stat(functools.partial(get_session_range, name=n))
    for n in ('targetprob', 'curr_contrast', 'phases_completed', \
            'targetprob_lower'):
        stats[n] = stat(functools.partial(get_last, name=n), \
                lambda a, b: b)
    stats['engagement'] = stat(measure_engagement, \
            lambda a, b: (a[0] + b[0], a[1] + b[1]))
    return stats


def process_datafiles(dfns, stats):
    data = {}
    try:
        for dfn in dfns:
            logging.debug("Datafile: %s" % dfn)
            df = None
            df = pymworks.open(dfn)
            sname = os.path.split(dfn)[1]
            r = {}
            for s in stats:
                r[s] = stats[s][0](df)
                logging.debug("\t%s: %s" % (s, r[s]))
            df.close()
            # data[dfn] = r
            data[sname] = r
        return data
    except KeyError:
        print "NO KEY FOUND: %s", dfn


def session_incremented(evs, test=lambda a, b: b.value != (a.value - 1),
                   recurse=True):
    """
    Remove events where evs[i].value != (evs[i-1].value - 1)

    evs : list 
        mworks events
        evs[i-1] occurs later in physical time than evs[i]. Re-reverse at end.
    """
    ## remove evs[i] if NEXT trial value (evs[i+1]) is more than CURR trial value ev[i] - 1.

    events = sorted(evs, key=lambda e: e.time, reverse=True)
    bad_indices = []
    bw = events
    for i in xrange(1, len(events)):
        if test(events[i - 1], events[i]):
            bad_indices.append(i)
    for i in bad_indices[::-1]: #[::-1]
        del(events[i])
    if recurse and len(bad_indices):
        bw = session_incremented(session_incremented(events, test=test, recurse=recurse))
    return sorted(bw, key=lambda e: e.time, reverse=False)


def get_session_range(df, name=''):
    evs = df.get_events(name)
    for ev in evs:
        if type(ev.value) != float:
            ev.value = float(ev.value)
    gevs = session_incremented(evs)
    return pymworks.stats.events.valuerange(gevs)


def measure_engagement(df, name='head_input', threshold=500):
    return pymworks.stats.events.time_in_state(df.get_events(name), \
            test=lambda e: e.value < threshold)


def get_max(df, name=''):
    return pymworks.stats.events.valuemax(df.get_events(name)).value


def get_last(df, name=''):
    return sorted(df.get_events(name), key=lambda e: e.time)[-1].value


def stat(extract_function=get_session_range, \
        combine_function=lambda a, b: a + b):
    return (extract_function, combine_function)


def parse_trials(dfns, remove_orphans=True):
    """
    Parse session .mwk files.
    Key is session name values are lists of dicts for each trial in session.
    Looks for all response and display events that occur within session.

    dfns : list of strings
        contains paths to each .mwk file to be parsed
    
    remove_orphans : boolean
        for each response event, best matching display update event
        set this to 'True' to remove display events with unknown outcome events
    """

    trialdata = {}                                                              # initiate output dict
    
    for dfn in dfns:
        df = None
        df = pymworks.open(dfn)                                                 # open the datafile

        sname = os.path.split(dfn)[1]
        trialdata[sname] = []

        modes = df.get_events('#state_system_mode')                             # find timestamps for run-time start and end (2=run)
        run_idxs = np.where(np.diff([i['time'] for i in modes])<20)             # 20 is kind of arbitray, but mode is updated twice for "run"
        bounds = []
        for r in run_idxs[0]:
            try:
                stop_ev = next(i for i in modes[r:] if i['value']==0 or i['value']==1)
            except StopIteration:
                end_event_name = 'trial_end'
                print "NO STOP DETECTED IN STATE MODES. Using alternative timestamp: %s." % end_event_name
                stop_ev = df.get_events(end_event_name)[-1]
                print stop_ev
            bounds.append([modes[r]['time'], stop_ev['time']])

        # print "................................................................"
        print "****************************************************************"
        print "Parsing file\n%s... " % dfn
        print "Found %i start events in session." % len(bounds)
        print "****************************************************************"


        for bidx,boundary in enumerate(bounds):
            # print "................................................................"
            print "SECTION %i" % bidx
            print "................................................................"
            tmp_devs = df.get_events('#stimDisplayUpdate')                      # get *all* display update events
            tmp_devs = [i for i in tmp_devs if i['time']<= boundary[1] and\
                        i['time']>=boundary[0]]                                 # only grab events within run-time bounds (see above)

            dimmed = df.get_events('flag_dim_target')
            dimmed = np.mean([i.value for i in dimmed])                         # set dim flag to separate trials based on initial stim presentation

            resptypes = ['session_correct_lick','session_correct_ignore',\
                            'session_bad_lick','session_bad_ignore']
                             #,\ # 'aborted_counter']                           # list of response variables to track...
            
            outevs = df.get_events(resptypes)                                   # get all events with these vars
            outevs = [i for i in outevs if i['time']<= boundary[1] and\
                        i['time']>=boundary[0]]
            R = sorted([r for r in outevs if r.value > 0], key=lambda e: e.time)

            T = pymworks.events.display.to_trials(tmp_devs, R,\
                        remove_unknown=False)                                   # match stim events to response events

            if dimmed:                                                          # don't double count stimulus for "faded" target stimuli
                trials = [i for i in T if i['alpha_multiplier']==1.0]
            else:
                trials = T

            print "N total response events: ", len(trials)
            if remove_orphans:                                                  # this should always be true, generally...
                orphans = [(i,x) for i,x in enumerate(trials) if\
                            x['outcome']=='unknown']
                trials = [t for t in trials if not t['outcome']=='unknown']     # get rid of display events without known outcome within 'duration_multiplier' time
                print "Found and removed %i orphan stimulus events in file %s"\
                            % (len(orphans), dfn)
                print "N valid trials: %i" % len(trials)

            trialdata[sname].append(trials)

    return trialdata


def analyze_sessions(datadir):
    F = get_session_list(datadir)                                               # list of all existent datafiles

    outpath = os.path.join(os.path.split(datadir)[0], 'info')                   # set up output path for session info
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    save_dict(F, outpath, 'sessions.pkl')

    processed_path = os.path.join(os.path.split(outpath)[0], 'processed')           # path to previously processed data
    new_sessions = get_new_sessions(processed_path, F)                              # get dict of NEW sessions (and animals)


    # analyze and parse new datafiles:
    for animal in new_sessions.keys():
        tmp_new = new_sessions[animal]
        dfns = [os.path.join(datadir, animal, i) for i in tmp_new]

        # Parse trials for each session:
        curr_trials = parse_trials(dfns)

        outfiles = [os.path.join(processed_path, i)\
                    for i in os.listdir(processed_path) if animal in i]

        if '_trials' in outfiles:
            fn_trials = open([i for i in outfiles if '_trials.pkl' in i][0], 'rb')   
            trials = pkl.load(fn_trials)                                             # if summary file exists for current animal, just append 
            trials.update(curr_trials)
        else:
            trials = curr_trials                                                  # otherwise, create new summary dict for this animal

        fname = '%s_trials.pkl' % animal 
        save_dict(trials, processed_path, fname)                                   # save/overwrite summary file with new session data

        # Get summary info for each session:
        stats = get_session_stats()
        curr_summary = process_datafiles(dfns, stats)

        if outfiles:
            fn_summ = open([i for i in outfiles if '_summary.pkl' in i][0], 'rb')
            summary = pkl.load(fn_summ)                                             # if summary file exists for current animal, just append 
            summary.update(curr_summary)
        else:
            summary = curr_summary 

        fname = '%s_summary.pkl' % animal 
        save_dict(summary, processed_path, fname)                                   # save/overwrite summary file with new session data



###############################################################################
# do stuff...
###############################################################################

def main(datadir):
    plot = 1
    analyze_sessions(datadir)   

if __name__ == "__main__":
    #datadir = sys.argv[1]  # data INPUT folder (all others are soft-coded relative to this one)
    main(sys.argv[1])

